// Fast exact identical-set edge construction (see frnn/fast_path.hpp).
#include <frnn/fast_path.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace frnn {
namespace {

constexpr int kLeaf = 32;
constexpr int kWarpBuffer = 256;
constexpr int kWarpsPerBlock = 4;
constexpr int kChunks = 16;

// Host result storage handed to the caller (freed with std::free); move-only,
// freed on destruction unless ownership is taken by nulling ptr.
struct HostBuffer {
  std::int64_t* ptr = nullptr;
  std::size_t bytes = 0;
  HostBuffer() = default;
  HostBuffer(HostBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
  HostBuffer& operator=(HostBuffer&& o) noexcept {
    if (this != &o) { release(); ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr; o.bytes = 0; }
    return *this;
  }
  ~HostBuffer() { release(); }
  static HostBuffer make(std::size_t elements) {
    HostBuffer h;
    const std::size_t align = std::size_t(1) << 21;
    std::size_t bytes = elements * sizeof(std::int64_t);
    bytes = (bytes + align - 1) / align * align;
    if (bytes == 0) bytes = align;
    h.ptr = static_cast<std::int64_t*>(std::aligned_alloc(align, bytes));
    h.bytes = h.ptr ? bytes : 0;
    return h;
  }
  void release() { std::free(ptr); ptr = nullptr; bytes = 0; }
};

void check(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("frnn fast path: ") + what + ": " +
                             cudaGetErrorString(error));
  }
}

__device__ __forceinline__ unsigned orderedBits(float value) {
  const unsigned bits = __float_as_uint(value);
  return bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}
__device__ __forceinline__ float unorderedBits(unsigned bits) {
  const unsigned raw = (bits & 0x80000000u) ? (bits ^ 0x80000000u) : ~bits;
  return __uint_as_float(raw);
}

template <int D>
__global__ void boundsKernel(const float* points, int n, unsigned* minimum,
                             unsigned* maximum, int* non_finite) {
  float lo[D], hi[D];
  bool bad = false;
#pragma unroll
  for (int d = 0; d < D; ++d) { lo[d] = INFINITY; hi[d] = -INFINITY; }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const float v = points[static_cast<long long>(i) * D + d];
      bad |= !isfinite(v);
      lo[d] = fminf(lo[d], v);
      hi[d] = fmaxf(hi[d], v);
    }
  }
  if (__any_sync(0xffffffffu, bad) && (threadIdx.x & 31) == 0) atomicExch(non_finite, 1);
#pragma unroll
  for (int d = 0; d < D; ++d) {
    for (int o = 16; o; o >>= 1) {
      lo[d] = fminf(lo[d], __shfl_xor_sync(0xffffffffu, lo[d], o));
      hi[d] = fmaxf(hi[d], __shfl_xor_sync(0xffffffffu, hi[d], o));
    }
    if ((threadIdx.x & 31) == 0) {
      atomicMin(minimum + d, orderedBits(lo[d]));
      atomicMax(maximum + d, orderedBits(hi[d]));
    }
  }
}

// Split axes cycle through dimensions ordered by decreasing extent.
__global__ void axesKernel(const unsigned* minimum, const unsigned* maximum,
                           int dimension, int* axes) {
  float extent[32];
  int order[32];
  for (int d = 0; d < dimension; ++d) {
    extent[d] = unorderedBits(maximum[d]) - unorderedBits(minimum[d]);
    order[d] = d;
  }
  for (int i = 0; i < dimension; ++i)
    for (int j = i + 1; j < dimension; ++j)
      if (extent[order[j]] > extent[order[i]]) {
        const int t = order[i]; order[i] = order[j]; order[j] = t;
      }
  for (int d = 0; d < dimension; ++d) axes[d] = order[d];
}

__global__ void iotaKernel(int* values, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) values[i] = i;
}

// A node is a contiguous run of leaves [a, b) split at a + ceil((b-a)/2).
// Sorting by (node, coordinate) orders points along the level axis per node.
template <int D>
__global__ void keysKernel(const float* points, const int* permutation, int n,
                           int leaves, int level, const int* axes,
                           unsigned long long* keys) {
  const int axis = axes[level % D];
  for (int position = blockIdx.x * blockDim.x + threadIdx.x; position < n;
       position += gridDim.x * blockDim.x) {
    const int leaf = position / kLeaf;
    int a = 0, b = leaves;
    for (int t = 0; t < level; ++t) {
      const int mid = a + (b - a + 1) / 2;
      if (leaf < mid) b = mid; else a = mid;
    }
    keys[position] = (static_cast<unsigned long long>(a) << 32) |
        orderedBits(points[static_cast<long long>(permutation[position]) * D + axis]);
  }
}

template <int D>
__global__ void gatherKernel(const float* points, const int* permutation,
                             int n, int padded, float* sorted_points,
                             int* sorted_indices) {
  for (int position = blockIdx.x * blockDim.x + threadIdx.x;
       position < padded; position += gridDim.x * blockDim.x) {
    if (position < n) {
      const int source = permutation[position];
#pragma unroll
      for (int d = 0; d < D; ++d)
        sorted_points[static_cast<long long>(position) * D + d] =
            points[static_cast<long long>(source) * D + d];
      sorted_indices[position] = source;
    } else {
#pragma unroll
      for (int d = 0; d < D; ++d)  // NaN padding never matches.
        sorted_points[static_cast<long long>(position) * D + d] = __int_as_float(0x7fc00000);
      sorted_indices[position] = -1;
    }
  }
}

template <int D>
__global__ void leafBoxKernel(const float* sorted_points, int leaves,
                              float* box_lo, float* box_hi) {
  const int leaf = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (leaf >= leaves) return;
#pragma unroll
  for (int d = 0; d < D; ++d) {
    const float v = sorted_points[(static_cast<long long>(leaf) * kLeaf + lane) * D + d];
    float lo = v, hi = v;  // fminf/fmaxf ignore NaN padding.
    for (int o = 16; o; o >>= 1) {
      lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, o));
      hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    if (lane == 0) {  // dimension-major layout
      box_lo[static_cast<long long>(d) * leaves + leaf] = lo;
      box_hi[static_cast<long long>(d) * leaves + leaf] = hi;
    }
  }
}

// Symmetric warp-per-leaf search.  Leaf L handles candidate leaves >= L; for
// a pair from different leaves both directed edges are emitted, which is exact
// because fl(a-b) = -fl(b-a) makes the distance bitwise symmetric.
// Pruning soundness: for q in [lo, hi], fl(lo - c) <= fl(q - c) and
// fl(c - hi) <= fl(c - q) (monotone rounding), so each clamped gap is at most
// |fl(c - q)| and the monotone fma accumulation never exceeds the distance.
template <int D>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    searchKernel(const float* __restrict__ sp, const int* __restrict__ sidx,
                 const float* __restrict__ blo, const float* __restrict__ bhi,
                 int M, float r2, float thr, bool exclude_self, bool undirected,
                 int* __restrict__ osrc, int* __restrict__ otgt,
                 unsigned long long* counter, unsigned long long cap,
                 int* __restrict__ degree) {
  constexpr int DP = (D + 3) / 4;
  constexpr int SB = 64;
  __shared__ int bs[kWarpsPerBlock][kWarpBuffer], bt[kWarpsPerBlock][kWarpBuffer];
  __shared__ float4 sc[kWarpsPerBlock][SB][DP];
  __shared__ int sid[kWarpsPerBlock][SB];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int leaf = blockIdx.x * kWarpsPerBlock + w;
  if (leaf >= M) return;
  float q[DP * 4], ql[D], qh[D];
  const long long qpos = static_cast<long long>(leaf) * kLeaf + lane;
#pragma unroll
  for (int d = 0; d < D; ++d) {
    q[d] = sp[qpos * D + d];
    ql[d] = blo[static_cast<long long>(d) * M + leaf];
    qh[d] = bhi[static_cast<long long>(d) * M + leaf];
  }
#pragma unroll
  for (int d = D; d < DP * 4; ++d) q[d] = 0.f;
  const int qid = sidx[qpos];
  const bool qv = qid >= 0;
  int cnt = 0, nb = 0, ns = 0;
  const unsigned lt = (1u << lane) - 1;
  auto flushOut = [&]() {
    __syncwarp();
    unsigned long long bi = 0;
    if (lane == 0 && nb) bi = atomicAdd(counter, static_cast<unsigned long long>(nb));
    bi = __shfl_sync(0xffffffffu, bi, 0);
    for (int i = lane; i < nb; i += 32)
      if (bi + i < cap) { osrc[bi + i] = bs[w][i]; otgt[bi + i] = bt[w][i]; }
    __syncwarp();
    nb = 0;
  };
  auto emit2 = [&](float dd, int raw) {
    const bool same_leaf = raw < 0;
    const int t = same_leaf ? ~raw : raw;
    const bool hit = qv && t >= 0 && dd <= r2 && !(exclude_self && t == qid);
    const bool rev = hit && !same_leaf;
    if (hit) cnt++;
    if (rev) atomicAdd(degree + t, 1);
    const bool e1 = hit && (!undirected || qid > t);
    const bool e2 = rev && (!undirected || t > qid);
    const unsigned m1 = __ballot_sync(0xffffffffu, e1), m2 = __ballot_sync(0xffffffffu, e2);
    if (m1 | m2) {
      int o = nb + __popc(m1 & lt) + __popc(m2 & lt);
      if (e1) { bs[w][o] = qid; bt[w][o] = t; ++o; }
      if (e2) { bs[w][o] = t; bt[w][o] = qid; }
      nb += __popc(m1) + __popc(m2);
      if (nb > kWarpBuffer - 64) flushOut();
    }
  };
  auto dist = [&](int j) -> float {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < DP; ++k) {
      const float4 v = sc[w][j][k];
      const float cc[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int d = k * 4 + t;
        if (d < D) {  // reference order: (database - query), mul then fma
          const float df = cc[t] - q[d];
          acc = d == 0 ? __fmul_rn(df, df) : __fmaf_rn(df, df, acc);
        }
      }
    }
    return acc;
  };
  auto process = [&]() {
    __syncwarp();
    int j = 0;
    for (; j + 1 < ns; j += 2) {
      const float d0 = dist(j), d1 = dist(j + 1);
      const int t0 = sid[w][j], t1 = sid[w][j + 1];
      emit2(d0, t0);
      emit2(d1, t1);
    }
    if (j < ns) { const float d0 = dist(j); emit2(d0, sid[w][j]); }
    __syncwarp();
    ns = 0;
  };
  for (int base = leaf & ~31; base < M; base += 32) {
    const int j = base + lane;
    bool ok = false;
    if (j < M && j >= leaf) {
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const float g = fmaxf(fmaxf(blo[static_cast<long long>(d) * M + j] - qh[d],
                                    ql[d] - bhi[static_cast<long long>(d) * M + j]), 0.f);
        acc = fmaf(g, g, acc);
      }
      ok = acc <= thr;
    }
    unsigned m = __ballot_sync(0xffffffffu, ok);
    while (m) {
      const int b = __ffs(m) - 1;
      m &= m - 1;
      const int cl = base + b;
      {
        float pa = 0.f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const float g = fmaxf(fmaxf(blo[static_cast<long long>(d) * M + cl] - q[d],
                                      q[d] - bhi[static_cast<long long>(d) * M + cl]), 0.f);
          pa = fmaf(g, g, pa);
        }
        if (!__any_sync(0xffffffffu, qv && pa <= thr)) continue;
      }
      const long long cpos = static_cast<long long>(cl) * kLeaf + lane;
      float c[DP * 4];
#pragma unroll
      for (int d = 0; d < D; ++d) c[d] = sp[cpos * D + d];
#pragma unroll
      for (int d = D; d < DP * 4; ++d) c[d] = 0.f;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) {
        const float g = fmaxf(fmaxf(ql[d] - c[d], c[d] - qh[d]), 0.f);
        acc = fmaf(g, g, acc);
      }
      const bool sv = acc <= thr;  // NaN padding fails this test
      const unsigned sm = __ballot_sync(0xffffffffu, sv);
      if (sv) {
        const int o = ns + __popc(sm & lt);
#pragma unroll
        for (int k = 0; k < DP; ++k) sc[w][o][k] = make_float4(c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]);
        const int id = sidx[cpos];
        sid[w][o] = (cl == leaf) ? ~id : id;
      }
      ns += __popc(sm);
      if (ns >= 32) process();
    }
  }
  process();
  flushOut();
  if (qv && cnt) atomicAdd(degree + qid, cnt);
}

__global__ void maxDegreeKernel(const int* degree, int n, int* max_count) {
  int best = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    best = max(best, degree[i]);
  for (int o = 16; o; o >>= 1) best = max(best, __shfl_xor_sync(0xffffffffu, best, o));
  if ((threadIdx.x & 31) == 0) atomicMax(max_count, best);
}

struct Buffers {
  cudaStream_t stream = nullptr;
  float* points = nullptr; std::size_t points_cap = 0;
  float* sorted_points = nullptr; std::size_t sorted_cap = 0;
  int* perm = nullptr; std::size_t perm_cap = 0;
  int* perm_alt = nullptr; std::size_t perm_alt_cap = 0;
  int* sorted_indices = nullptr; std::size_t sidx_cap = 0;
  unsigned long long* keys = nullptr; std::size_t keys_cap = 0;
  unsigned long long* keys_alt = nullptr; std::size_t keys_alt_cap = 0;
  float* box_lo = nullptr; std::size_t box_lo_cap = 0;
  float* box_hi = nullptr; std::size_t box_hi_cap = 0;
  char* scratch = nullptr; std::size_t scratch_cap = 0;
  int* edges = nullptr; std::size_t edges_cap = 0;  // bytes
  int* degree = nullptr; std::size_t degree_cap = 0;
  unsigned* small = nullptr;       // device: 0-31 min, 32-63 max, 64-95 axes, 96-97 counter, 98 maxcnt, 99 nonfinite
  unsigned* host_small = nullptr;  // pinned
  int* host_edges = nullptr; std::size_t host_edges_cap = 0;  // elements
  std::size_t last_count = 0;
  cudaEvent_t events[32] = {};
};

// One cache per CUDA device: streams and allocations are device-bound.
Buffers& buffers() {
  int device = 0;
  check(cudaGetDevice(&device), "get device");
  static auto* instances = new std::map<int, Buffers>();  // intentionally leaked at exit
  return (*instances)[device];
}
std::mutex& fastMutex() {
  static std::mutex m;
  return m;
}

template <typename T>
void grow(T*& pointer, std::size_t& capacity, std::size_t bytes, const char* what) {
  if (pointer != nullptr && bytes <= capacity) return;
  if (pointer != nullptr) cudaFree(pointer);
  pointer = nullptr;
  capacity = 0;
  const std::size_t want = bytes + bytes / 2 + 256;
  check(cudaMalloc(reinterpret_cast<void**>(&pointer), want), what);
  capacity = want;
}

// Joins every started thread on scope exit, including exception paths.
struct Joiner {
  std::vector<std::thread> list;
  ~Joiner() { for (auto& t : list) t.join(); }
};

// Runs body(t) for t in [0, count); t = 0 runs on the calling thread.
void parallel(int count, const std::function<void(int)>& body) {
  Joiner threads;
  for (int t = 1; t < count; ++t) threads.list.emplace_back(body, t);
  body(0);
}

void ensureSmall(Buffers& b) {
  if (b.small != nullptr) return;
  check(cudaMalloc(&b.small, 128 * sizeof(unsigned)), "alloc small");
  check(cudaMallocHost(&b.host_small, 128 * sizeof(unsigned)), "alloc host small");
}

// Builds the kd ordering (sorted_points, sorted_indices, leaf boxes) of the n
// device points on stream s, and resets the search counters.
template <int D>
void buildTree(Buffers& b, const float* points, int n, cudaStream_t s) {
  const int leaves = (n + kLeaf - 1) / kLeaf;
  const int padded = leaves * kLeaf;
  int levels = 0;
  while ((1 << levels) < leaves) ++levels;
  const std::size_t sz = static_cast<std::size_t>(padded);
  grow(b.sorted_points, b.sorted_cap, sizeof(float) * sz * D, "alloc sorted");
  grow(b.perm, b.perm_cap, sizeof(int) * sz, "alloc perm");
  grow(b.perm_alt, b.perm_alt_cap, sizeof(int) * sz, "alloc perm");
  grow(b.sorted_indices, b.sidx_cap, sizeof(int) * sz, "alloc idx");
  grow(b.keys, b.keys_cap, 8 * sz, "alloc keys");
  grow(b.keys_alt, b.keys_alt_cap, 8 * sz, "alloc keys");
  grow(b.degree, b.degree_cap, sizeof(int) * n, "alloc degree");
  grow(b.box_lo, b.box_lo_cap, sizeof(float) * leaves * D, "alloc boxes");
  grow(b.box_hi, b.box_hi_cap, sizeof(float) * leaves * D, "alloc boxes");
  std::size_t sort_bytes = 0;
  check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, b.keys, b.keys_alt, b.perm,
                                        b.perm_alt, n, 0, 32 + levels, s), "sort size");
  grow(b.scratch, b.scratch_cap, sort_bytes, "alloc scratch");

  unsigned* minimum = b.small;
  unsigned* maximum = b.small + 32;
  int* axes = reinterpret_cast<int*>(b.small + 64);
  int* non_finite = reinterpret_cast<int*>(b.small + 99);
  check(cudaMemsetAsync(minimum, 0xff, 32 * sizeof(unsigned), s), "memset");
  check(cudaMemsetAsync(maximum, 0, 32 * sizeof(unsigned), s), "memset");
  check(cudaMemsetAsync(b.small + 96, 0, 4 * sizeof(unsigned), s), "memset");
  const int threads = 256;
  const int blocks = std::max(1, std::min(1024, (n + threads - 1) / threads));
  boundsKernel<D><<<std::min(blocks, 512), threads, 0, s>>>(points, n, minimum, maximum, non_finite);
  axesKernel<<<1, 1, 0, s>>>(minimum, maximum, D, axes);
  iotaKernel<<<blocks, threads, 0, s>>>(b.perm, n);
  int* perm = b.perm;
  int* perm_alt = b.perm_alt;
  for (int level = 0; level < levels; ++level) {
    keysKernel<D><<<(n + threads - 1) / threads, threads, 0, s>>>(points, perm, n, leaves,
                                                                  level, axes, b.keys);
    std::size_t bytes = b.scratch_cap;
    check(cub::DeviceRadixSort::SortPairs(b.scratch, bytes, b.keys, b.keys_alt, perm, perm_alt,
                                          n, 0, 32 + levels, s), "radix sort");
    std::swap(perm, perm_alt);
  }
  gatherKernel<D><<<(padded + threads - 1) / threads, threads, 0, s>>>(
      points, perm, n, padded, b.sorted_points, b.sorted_indices);
  leafBoxKernel<D><<<(padded + 127) / 128, 128, 0, s>>>(b.sorted_points, leaves, b.box_lo, b.box_hi);
  check(cudaGetLastError(), "launch build");
}

template <int D>
bool run(const float* host_points, int n, float radius, int max_neighbors,
         bool exclude_self, bool undirected, std::int64_t** result,
         std::int64_t* result_count) {
  const bool timing = std::getenv("FRNN_FAST_TIMING") != nullptr;
  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&](const char* name) { if (timing) { auto t = std::chrono::steady_clock::now(); std::fprintf(stderr, "%s %.3f ms\n", name, std::chrono::duration<double, std::milli>(t - t0).count()); t0 = t; } };
  Buffers& b = buffers();
  if (b.stream == nullptr) {
    check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), "create stream");
  }
  ensureSmall(b);
  cudaStream_t s = b.stream;
  grow(b.points, b.points_cap, sizeof(float) * n * D, "alloc points");
  check(cudaMemcpyAsync(b.points, host_points, sizeof(float) * n * D,
                        cudaMemcpyHostToDevice, s), "copy points");
  buildTree<D>(b, b.points, n, s);
  const int leaves = (n + kLeaf - 1) / kLeaf;
  const int threads = 256;
  const int blocks = std::max(1, std::min(1024, (n + threads - 1) / threads));
  unsigned long long* counter = reinterpret_cast<unsigned long long*>(b.small + 96);
  int* max_count = reinterpret_cast<int*>(b.small + 98);

  const float radius_squared = radius * radius;
  const float prune_threshold = radius_squared * 1.0001F;
  auto ensureHost = [&](std::size_t elements) {
    if (b.host_edges != nullptr && b.host_edges_cap >= elements) return;
    if (b.host_edges) cudaFreeHost(b.host_edges);
    b.host_edges = nullptr;
    b.host_edges_cap = 0;
    const std::size_t want = elements + elements / 8 + 1024;
    check(cudaHostAlloc(&b.host_edges, want * sizeof(int), cudaHostAllocMapped), "alloc mapped edges");
    b.host_edges_cap = want;
  };
  // Successful output is at most n * max_neighbors edges; larger guesses retry.
  ensureHost(static_cast<std::size_t>(n) * std::min(max_neighbors, 64) * 2);
  unsigned long long capacity = b.host_edges_cap / 2;
  int* device_edges = nullptr;
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_edges), b.host_edges, 0), "mapped pointer");
  if (b.events[0] == nullptr) {
    for (auto& e : b.events) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    // Reset counter and max degree only; word 99 holds the non-finite flag.
    check(cudaMemsetAsync(b.small + 96, 0, 3 * sizeof(unsigned), s), "memset");
    check(cudaMemsetAsync(b.degree, 0, sizeof(int) * n, s), "memset degree");
    searchKernel<D><<<(leaves + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32, 0, s>>>(
        b.sorted_points, b.sorted_indices, b.box_lo, b.box_hi, leaves, radius_squared,
        prune_threshold, exclude_self, undirected, device_edges, device_edges + capacity, counter,
        capacity, b.degree);
    maxDegreeKernel<<<std::min(blocks, 256), threads, 0, s>>>(b.degree, n, max_count);
    check(cudaGetLastError(), "launch search");
    check(cudaMemcpyAsync(b.host_small + 96, b.small + 96, 4 * sizeof(unsigned),
                          cudaMemcpyDeviceToHost, s), "copy counters");
    // Allocate and prefault the host result while the GPU works.
    const std::size_t estimate = b.last_count ? b.last_count + b.last_count / 4 + 4096
                                              : static_cast<std::size_t>(n) * 48 + 4096;
    HostBuffer out = HostBuffer::make(estimate * 2);
    {
      Joiner faulters;  // declared after `out`, so joined before it is freed
      if (out.ptr) {
        const int nt = 8;
        const std::size_t bytes = out.bytes;
        char* base = reinterpret_cast<char*>(out.ptr);
        for (int t = 0; t < nt; ++t) {
          faulters.list.emplace_back([=]() {
            const std::size_t begin = bytes * t / nt, end = bytes * (t + 1) / nt;
            for (std::size_t o = begin & ~std::size_t(4095); o < end; o += 4096) base[o] = 0;
          });
        }
      }
      check(cudaStreamSynchronize(s), "synchronize search");
    }
    lap("gpu (h2d+build+search)");
    if (b.host_small[99] != 0 || static_cast<int>(b.host_small[98]) > max_neighbors) {
      return false;
    }
    unsigned long long edge_count;
    std::memcpy(&edge_count, b.host_small + 96, sizeof(edge_count));
    if (edge_count > capacity) {
      ensureHost(static_cast<std::size_t>(edge_count) * 2);
      capacity = b.host_edges_cap / 2;
      check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_edges), b.host_edges, 0), "mapped pointer");
      continue;
    }
    const std::int64_t count = static_cast<std::int64_t>(edge_count);
    b.last_count = static_cast<std::size_t>(count);
    if (out.ptr == nullptr || out.bytes < static_cast<std::size_t>(count) * 2 * sizeof(std::int64_t)) {
      out = HostBuffer::make(static_cast<std::size_t>(count) * 2);
      if (out.ptr == nullptr) throw std::bad_alloc();
    }
    std::int64_t* output = out.ptr;
    const int* sources = b.host_edges;
    const int* targets = b.host_edges + capacity;
    // Widen to int64 and group by ascending source (the buildEdges contract)
    // with a per-chunk counting sort; order within a source stays unspecified.
    const int chunks = static_cast<int>(std::min<std::int64_t>(
        std::min<unsigned>(24, std::max(1u, std::thread::hardware_concurrency())),
        std::max<std::int64_t>(1, count >> 17)));
    const std::int64_t step = (count + chunks - 1) / chunks;
    const std::size_t un = static_cast<std::size_t>(n);
    std::vector<int> rel(un * chunks);  // per-(chunk, source) count, then offset within source
    std::vector<std::int64_t> start(un);
    parallel(chunks, [&](int c) {
      int* h = rel.data() + un * c;
      for (std::int64_t i = c * step, e = std::min(count, i + step); i < e; ++i) ++h[sources[i]];
    });
    parallel(chunks, [&](int c) {
      for (std::size_t v = un * c / chunks, e = un * (c + 1) / chunks; v < e; ++v) {
        int sum = 0;
        for (int k = 0; k < chunks; ++k) { const int x = rel[un * k + v]; rel[un * k + v] = sum; sum += x; }
        start[v] = sum;
      }
    });
    std::int64_t total = 0;
    for (auto& v : start) { const std::int64_t x = v; v = total; total += x; }
    parallel(chunks, [&](int c) {
      int* h = rel.data() + un * c;
      for (std::int64_t i = c * step, e = std::min(count, i + step); i < e; ++i) {
        const int v = sources[i];
        const std::int64_t p = start[v] + h[v]++;
        output[p] = v;
        output[count + p] = targets[i];
      }
    });
    lap("copy+widen+group");
    *result = out.ptr;
    out.ptr = nullptr;  // ownership passes to the caller
    *result_count = count;
    return true;
  }
  return false;
}

// (source, target) as one 64-bit key, so one radix sort orders by source, then target.
__global__ void packEdgesKernel(const int* __restrict__ sources, const int* __restrict__ targets,
                                long long count, unsigned long long* __restrict__ keys) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * blockDim.x)
    keys[i] = (static_cast<unsigned long long>(sources[i]) << 32) |
              static_cast<unsigned>(targets[i]);
}

__global__ void unpackEdgesKernel(const unsigned long long* __restrict__ keys, long long count,
                                  std::int64_t* __restrict__ out) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    out[i] = static_cast<std::int64_t>(keys[i] >> 32);
    out[count + i] = static_cast<std::int64_t>(keys[i] & 0xffffffffull);
  }
}

// Device-resident fast path: same tree and search as run<D>, but the points are
// read in place, the edges stay in device memory, and they are grouped on the
// GPU (radix sort by source, then target) instead of on host threads.
template <int D>
bool runDevice(const float* points, int n, float radius, int max_neighbors, bool exclude_self,
               bool undirected, cudaStream_t s,
               const std::function<std::int64_t*(std::int64_t)>& allocate,
               std::int64_t* result_count) {
  Buffers& b = buffers();
  ensureSmall(b);
  buildTree<D>(b, points, n, s);
  const int leaves = (n + kLeaf - 1) / kLeaf;
  const int threads = 256;
  const int blocks = std::max(1, std::min(1024, (n + threads - 1) / threads));
  unsigned long long* counter = reinterpret_cast<unsigned long long*>(b.small + 96);
  int* max_count = reinterpret_cast<int*>(b.small + 98);
  const float radius_squared = radius * radius;
  const float prune_threshold = radius_squared * 1.0001F;
  // Successful output is at most n * max_neighbors edges; a larger count retries once.
  unsigned long long capacity =
      b.last_count ? b.last_count + b.last_count / 4 + 4096
                   : static_cast<unsigned long long>(n) * std::min(max_neighbors, 64);
  for (int attempt = 0; attempt < 2; ++attempt) {
    grow(b.edges, b.edges_cap, 2 * capacity * sizeof(int), "alloc device edges");
    capacity = b.edges_cap / (2 * sizeof(int));
    int* sources = b.edges;
    int* targets = b.edges + capacity;
    // Reset counter and max degree only; word 99 holds the non-finite flag.
    check(cudaMemsetAsync(b.small + 96, 0, 3 * sizeof(unsigned), s), "memset");
    check(cudaMemsetAsync(b.degree, 0, sizeof(int) * n, s), "memset degree");
    searchKernel<D><<<(leaves + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32, 0, s>>>(
        b.sorted_points, b.sorted_indices, b.box_lo, b.box_hi, leaves, radius_squared,
        prune_threshold, exclude_self, undirected, sources, targets, counter, capacity,
        b.degree);
    maxDegreeKernel<<<std::min(blocks, 256), threads, 0, s>>>(b.degree, n, max_count);
    check(cudaGetLastError(), "launch search");
    check(cudaMemcpyAsync(b.host_small + 96, b.small + 96, 4 * sizeof(unsigned),
                          cudaMemcpyDeviceToHost, s), "copy counters");
    check(cudaStreamSynchronize(s), "synchronize search");
    if (b.host_small[99] != 0 || static_cast<int>(b.host_small[98]) > max_neighbors) {
      return false;
    }
    unsigned long long edge_count;
    std::memcpy(&edge_count, b.host_small + 96, sizeof(edge_count));
    if (edge_count > capacity) {
      capacity = edge_count + edge_count / 8 + 1024;
      continue;
    }
    if (edge_count > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
      return false;  // beyond cub's int item count; the general path handles it
    }
    const int count = static_cast<int>(edge_count);
    b.last_count = edge_count;
    int index_bits = 1;
    while ((1ll << index_bits) < n) ++index_bits;
    grow(b.keys, b.keys_cap, 8 * static_cast<std::size_t>(count) + 8, "alloc edge keys");
    grow(b.keys_alt, b.keys_alt_cap, 8 * static_cast<std::size_t>(count) + 8, "alloc edge keys");
    std::size_t sort_bytes = 0;
    check(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, b.keys, b.keys_alt, count, 0,
                                         32 + index_bits, s), "edge sort size");
    grow(b.scratch, b.scratch_cap, sort_bytes, "alloc scratch");
    std::int64_t* out = allocate(count);
    if (count > 0) {
      const int edge_blocks = std::max(1, std::min(4096, (count + threads - 1) / threads));
      packEdgesKernel<<<edge_blocks, threads, 0, s>>>(sources, targets, count, b.keys);
      std::size_t bytes = b.scratch_cap;
      check(cub::DeviceRadixSort::SortKeys(b.scratch, bytes, b.keys, b.keys_alt, count, 0,
                                           32 + index_bits, s), "edge sort");
      unpackEdgesKernel<<<edge_blocks, threads, 0, s>>>(b.keys_alt, count, out);
      check(cudaGetLastError(), "launch edge output");
    }
    // The buffers are shared with the next call, which may use another stream.
    check(cudaStreamSynchronize(s), "synchronize edge output");
    *result_count = count;
    return true;
  }
  return false;
}

}  // namespace

bool buildEdgesFast(const float* points, std::int64_t size, int dimension, float radius,
                    int max_neighbors, bool exclude_self, bool undirected,
                    std::int64_t** result, std::int64_t* result_count) {
  *result = nullptr;
  *result_count = 0;
  if (points == nullptr || size < 64 || size > (1 << 30) || max_neighbors <= 0 ||
      !std::isfinite(radius) || radius <= 0.0F) {
    return false;
  }
  std::lock_guard<std::mutex> lock(fastMutex());
  const int n = static_cast<int>(size);
  switch (dimension) {
#define FRNN_FAST_CASE(D) \
  case D:                 \
    return run<D>(points, n, radius, max_neighbors, exclude_self, undirected, result, result_count);
    FRNN_FAST_CASE(2)
    FRNN_FAST_CASE(3)
    FRNN_FAST_CASE(4)
    FRNN_FAST_CASE(6)
    FRNN_FAST_CASE(8)
    FRNN_FAST_CASE(12)
    FRNN_FAST_CASE(16)
#undef FRNN_FAST_CASE
    default:
      return false;
  }
}

bool buildEdgesFastDevice(const float* points, std::int64_t size, int dimension, float radius,
                          int max_neighbors, bool exclude_self, bool undirected,
                          cudaStream_t stream,
                          const std::function<std::int64_t*(std::int64_t)>& allocate,
                          std::int64_t* result_count) {
  *result_count = 0;
  if (points == nullptr || size < 64 || size > (1 << 30) || max_neighbors <= 0 ||
      !std::isfinite(radius) || radius <= 0.0F) {
    return false;
  }
  std::lock_guard<std::mutex> lock(fastMutex());
  const int n = static_cast<int>(size);
  switch (dimension) {
#define FRNN_FAST_CASE(D) \
  case D:                 \
    return runDevice<D>(points, n, radius, max_neighbors, exclude_self, undirected, stream, \
                        allocate, result_count);
    FRNN_FAST_CASE(2)
    FRNN_FAST_CASE(3)
    FRNN_FAST_CASE(4)
    FRNN_FAST_CASE(6)
    FRNN_FAST_CASE(8)
    FRNN_FAST_CASE(12)
    FRNN_FAST_CASE(16)
#undef FRNN_FAST_CASE
    default:
      return false;
  }
}

}  // namespace frnn
