#include <frnn/frnn.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace frnn {
namespace {

#if defined(FRNN_ENABLE_STAGE_PROFILING)
constexpr int kProfileEventCount = 10;
thread_local cudaEvent_t* active_profile_events = nullptr;
#endif

constexpr int kGridDimensions = 4;
constexpr int kMaximumGridResolution = 128;
constexpr int kMaximumGridCells3D = 128 * 1024;
constexpr int kMaximumGridCells4D = 128 * 128 * 128;
constexpr int kThreads = 256;
constexpr int kSearchThreads = 128;
constexpr std::int64_t kBruteForceCoordinateWork = 2'000'000;

struct GridParameters {
  float minimum[kGridDimensions];
  float inverse_cell_size;
  int resolution[kGridDimensions];
  int total_cells;
  int grid_dimensions;
};

[[noreturn]] void throwCuda(cudaError_t error, const char* operation) {
  std::ostringstream message;
  message << operation << ": " << cudaGetErrorString(error);
  throw std::runtime_error(message.str());
}

void checkCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throwCuda(error, operation);
  }
}

#if defined(FRNN_ENABLE_STAGE_PROFILING)
void recordProfileEvent(int index, cudaStream_t stream) {
  if (active_profile_events != nullptr) {
    checkCuda(cudaEventRecord(active_profile_events[index], stream),
              "record profiling event");
  }
}
#else
void recordProfileEvent(int, cudaStream_t) {}
#endif

int blocksFor(std::int64_t count) {
  if (count <= 0) {
    return 1;
  }
  return static_cast<int>(
      std::min<std::int64_t>((count + kThreads - 1) / kThreads, 65535));
}

int gridDimensionCount(int dimension) {
  if (dimension <= 4) return std::min(dimension, 3);
  return 4;
}

int maximumGridCells(int dimension) {
  const int gd = gridDimensionCount(dimension);
  if (gd <= 3) return kMaximumGridCells3D;
  return kMaximumGridCells4D;
}

void validateView(std::int64_t size, int dimension, const float* data,
                  const char* name) {
  if (size < 0) {
    throw std::invalid_argument(std::string(name) + ".size must be non-negative");
  }
  if (dimension <= 0) {
    throw std::invalid_argument(std::string(name) +
                                ".dimension must be positive");
  }
  if (dimension > 32) {
    throw std::invalid_argument(std::string(name) +
                                ".dimension must not exceed 32");
  }
  if (size > 0 && data == nullptr) {
    throw std::invalid_argument(std::string(name) +
                                ".data must not be null for non-empty input");
  }
  if (size > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) +
                                ".size exceeds the supported CUDA index range");
  }
}

void validateConfiguration(float radius, int max_neighbors) {
  if (!std::isfinite(radius) || radius <= 0.0F) {
    throw std::invalid_argument("radius must be finite and greater than zero");
  }
  if (max_neighbors < 0) {
    throw std::invalid_argument("max_neighbors must be non-negative");
  }
}

void validateHostValues(PointView view, const char* name) {
  validateView(view.size, view.dimension, view.data, name);
  const std::int64_t values = view.size * view.dimension;
  for (std::int64_t index = 0; index < values; ++index) {
    if (!std::isfinite(view.data[index])) {
      throw std::invalid_argument(std::string(name) +
                                  " contains a non-finite coordinate");
    }
  }
}

__device__ float atomicMinimum(float* address, float value) {
  int* integer_address = reinterpret_cast<int*>(address);
  int old = *integer_address;
  while (value < __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(integer_address, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
  return __int_as_float(old);
}

__device__ float atomicMaximum(float* address, float value) {
  int* integer_address = reinterpret_cast<int*>(address);
  int old = *integer_address;
  while (value > __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(integer_address, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
  return __int_as_float(old);
}

__global__ void initializeBounds(float* bounds) {
  if (threadIdx.x < kGridDimensions) {
    bounds[threadIdx.x] = INFINITY;
    bounds[kGridDimensions + threadIdx.x] = -INFINITY;
  }
}

__global__ void computeBounds(const float* points, int point_count,
                              int dimension, int grid_dimensions,
                              float* bounds) {
  for (int point = blockIdx.x * blockDim.x + threadIdx.x;
       point < point_count; point += blockDim.x * gridDim.x) {
    for (int axis = 0; axis < grid_dimensions; ++axis) {
      const float value = points[point * dimension + axis];
      atomicMinimum(&bounds[axis], value);
      atomicMaximum(&bounds[kGridDimensions + axis], value);
    }
  }
}

__global__ void finalizeGrid(const float* bounds, int grid_dimensions,
                             int maximum_cells, float radius,
                             GridParameters* parameters) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float largest_extent = 0.0F;
  for (int axis = 0; axis < grid_dimensions; ++axis) {
    largest_extent =
        fmaxf(largest_extent,
              bounds[kGridDimensions + axis] - bounds[axis]);
  }
  const float radius_cell_size =
      radius * (grid_dimensions == 4 ? 0.5F : 1.0F);
  float cell_size = fmaxf(radius_cell_size,
                          largest_extent /
                              static_cast<float>(kMaximumGridResolution - 1));
  if (!(cell_size > 0.0F) || !isfinite(cell_size)) {
    cell_size = radius_cell_size;
  }

  int resolution[kGridDimensions] = {1, 1, 1, 1};
  int total_cells = 1;
  for (int attempt = 0; attempt < 8; ++attempt) {
    long long candidate_cells = 1;
    for (int axis = 0; axis < grid_dimensions; ++axis) {
      int axis_resolution = static_cast<int>(
                                floorf((bounds[kGridDimensions + axis] -
                                        bounds[axis]) /
                                       cell_size)) +
                            1;
      axis_resolution =
          max(1, min(kMaximumGridResolution, axis_resolution));
      resolution[axis] = axis_resolution;
      candidate_cells *= axis_resolution;
    }
    if (candidate_cells <= maximum_cells) {
      total_cells = static_cast<int>(candidate_cells);
      break;
    }
    const float scale =
        powf(static_cast<float>(candidate_cells) /
                 static_cast<float>(maximum_cells),
             1.0F / static_cast<float>(grid_dimensions));
    cell_size *= fmaxf(1.01F, scale * 1.01F);
  }

  parameters->inverse_cell_size = 1.0F / cell_size;
  parameters->grid_dimensions = grid_dimensions;
  parameters->total_cells = total_cells;
  for (int axis = 0; axis < kGridDimensions; ++axis) {
    if (axis < grid_dimensions) {
      parameters->minimum[axis] = bounds[axis];
      parameters->resolution[axis] = resolution[axis];
    } else {
      parameters->minimum[axis] = 0.0F;
      parameters->resolution[axis] = 1;
    }
  }
}

__global__ void insertPoints(const float* points, int point_count,
                             int dimension,
                             const GridParameters* parameters,
                             int* grid_counts, int* point_cells,
                             int* point_cell_indices) {
  for (int point = blockIdx.x * blockDim.x + threadIdx.x;
       point < point_count; point += blockDim.x * gridDim.x) {
    int coordinate[kGridDimensions] = {0, 0, 0, 0};
    for (int axis = 0; axis < parameters->grid_dimensions; ++axis) {
      coordinate[axis] = static_cast<int>(
          floorf((points[point * dimension + axis] -
                  parameters->minimum[axis]) *
                 parameters->inverse_cell_size));
      coordinate[axis] =
          max(0, min(parameters->resolution[axis] - 1, coordinate[axis]));
    }
    const int cell =
        ((coordinate[0] * parameters->resolution[1] + coordinate[1]) *
             parameters->resolution[2] +
         coordinate[2]) *
            parameters->resolution[3] +
        coordinate[3];
    point_cells[point] = cell;
    point_cell_indices[point] = atomicAdd(&grid_counts[cell], 1);
  }
}

// sorted_points uses SoA layout: sorted_points[axis * point_count + dest].
// All dimension axes are stored so squaredDistanceSoA can check all dims.
__global__ void countingSort(const float* points, int point_count,
                             int dimension, int grid_dimensions,
                             const int* point_cells,
                             const int* point_cell_indices,
                             const int* grid_offsets, float* sorted_points,
                             int* sorted_indices) {
  for (int point = blockIdx.x * blockDim.x + threadIdx.x;
       point < point_count; point += blockDim.x * gridDim.x) {
    const int destination =
        grid_offsets[point_cells[point]] + point_cell_indices[point];
    for (int axis = 0; axis < dimension; ++axis) {
      sorted_points[axis * point_count + destination] =
          points[point * dimension + axis];
    }
    sorted_indices[destination] = point;
  }
}


__device__ bool precedes(float lhs_distance, std::int64_t lhs_index,
                         float rhs_distance, std::int64_t rhs_index) {
  return lhs_distance < rhs_distance ||
         (lhs_distance == rhs_distance && lhs_index < rhs_index);
}

__device__ void insertNeighbor(float distance, int database_index,
                               int max_neighbors, int& found,
                               float* distances, int* indices) {
  int insertion = min(found, max_neighbors);
  while (insertion > 0 &&
         precedes(distance, database_index, distances[insertion - 1],
                  indices[insertion - 1])) {
    --insertion;
  }
  if (insertion >= max_neighbors) {
    return;
  }
  const int last = min(found, max_neighbors - 1);
  for (int position = last; position > insertion; --position) {
    distances[position] = distances[position - 1];
    indices[position] = indices[position - 1];
  }
  distances[insertion] = distance;
  indices[insertion] = database_index;
  found = min(found + 1, max_neighbors);
}

__device__ bool follows(float lhs_distance, int lhs_index,
                        float rhs_distance, int rhs_index) {
  return precedes(rhs_distance, rhs_index, lhs_distance, lhs_index);
}

__device__ void swapNeighbor(float* distances, int* indices, int lhs,
                             int rhs) {
  const float distance = distances[lhs];
  const int index = indices[lhs];
  distances[lhs] = distances[rhs];
  indices[lhs] = indices[rhs];
  distances[rhs] = distance;
  indices[rhs] = index;
}

__device__ void siftNeighborDown(float* distances, int* indices, int count,
                                 int root) {
  while (true) {
    const int left = root * 2 + 1;
    if (left >= count) {
      return;
    }
    int worst = left;
    const int right = left + 1;
    if (right < count &&
        follows(distances[right], indices[right], distances[left],
                indices[left])) {
      worst = right;
    }
    if (!follows(distances[worst], indices[worst], distances[root],
                 indices[root])) {
      return;
    }
    swapNeighbor(distances, indices, root, worst);
    root = worst;
  }
}

__device__ void insertNeighborHeap(float distance, int database_index,
                                   int max_neighbors, int& found,
                                   float* distances, int* indices) {
  if (found < max_neighbors) {
    int position = found++;
    distances[position] = distance;
    indices[position] = database_index;
    while (position > 0) {
      const int parent = (position - 1) / 2;
      if (!follows(distances[position], indices[position],
                   distances[parent], indices[parent])) {
        break;
      }
      swapNeighbor(distances, indices, position, parent);
      position = parent;
    }
    return;
  }
  if (!precedes(distance, database_index, distances[0], indices[0])) {
    return;
  }
  distances[0] = distance;
  indices[0] = database_index;
  siftNeighborDown(distances, indices, found, 0);
}

__device__ void sortNeighborHeap(float* distances, int* indices, int count) {
  for (int end = count - 1; end > 0; --end) {
    swapNeighbor(distances, indices, 0, end);
    siftNeighborDown(distances, indices, end, 0);
  }
}

__device__ void buildNeighborHeap(float* distances, int* indices, int count) {
  for (int root = count / 2 - 1; root >= 0; --root) {
    siftNeighborDown(distances, indices, count, root);
  }
}

__device__ void selectNeighbor(float distance, int database_index,
                               int max_neighbors, int& found, bool& heap_mode,
                               float* distances, int* indices) {
  if (max_neighbors >= 64 && (heap_mode || found >= 24)) {
    if (!heap_mode) {
      buildNeighborHeap(distances, indices, found);
      heap_mode = true;
    }
    insertNeighborHeap(distance, database_index, max_neighbors, found,
                       distances, indices);
  } else {
    insertNeighbor(distance, database_index, max_neighbors, found, distances,
                   indices);
  }
}

template <int StaticDimension>
__device__ float squaredDistance(const float* lhs, const float* rhs,
                                 int runtime_dimension, float limit) {
  const int dimension =
      StaticDimension == 0 ? runtime_dimension : StaticDimension;
  float distance = 0.0F;
#pragma unroll
  for (int axis = 0; axis < dimension; ++axis) {
    const float difference = lhs[axis] - rhs[axis];
    distance += difference * difference;
    if (distance > limit) {
      break;
    }
  }
  return distance;
}

// SoA variant: sorted_database stored as soa[axis * database_count + idx].
// Each dimension's values are contiguous: 32 candidates per cache line vs
// 2.67 for AoS, reducing DRAM pressure on the early-exit filter path.
//
// For D a multiple of 4 and > 4: check axes in groups of 4, interleaving the
// loads so all 4 in a group are issued before any result is used.  The serial
// load chain (each break-check gates the next load) limits the current approach
// to ~1 axis per 576-cycle DRAM latency.  With 4-way interleaving, a group of
// 4 loads completes in ~576 cycles (one DRAM round-trip) instead of 4×576.
// Group order: unindexed dims (4..D-1) first, grid dims (0..3) last, so most
// failing candidates are eliminated in group 0 without entering group 1.
template <int StaticDimension>
__device__ float squaredDistanceSoA(const float* soa, int idx,
                                    int database_count,
                                    const float* query_point,
                                    int runtime_dimension, float limit) {
  const int dimension =
      StaticDimension == 0 ? runtime_dimension : StaticDimension;
  // kGD: number of grid dimensions. Grid dims are put in the last group.
  constexpr int kGD =
      StaticDimension == 0 ? 0
      : StaticDimension > 4 ? 4
      : StaticDimension <= 3 ? StaticDimension
      : 3;
  float distance = 0.0F;
  if constexpr (StaticDimension != 0 && StaticDimension % 4 == 0) {
    // 4-way interleaved: D/4 groups, each issuing 4 independent loads.
    constexpr int kGroups = StaticDimension / 4;
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      // Axes within this group (compile-time constants after unroll):
      const int ax0 = (g * 4 + kGD) % StaticDimension;
      const int ax1 = (g * 4 + kGD + 1) % StaticDimension;
      const int ax2 = (g * 4 + kGD + 2) % StaticDimension;
      const int ax3 = (g * 4 + kGD + 3) % StaticDimension;
      // All 4 loads issued before any computation (no inter-load dependency):
      const float d0 = soa[ax0 * database_count + idx] - query_point[ax0];
      const float d1 = soa[ax1 * database_count + idx] - query_point[ax1];
      const float d2 = soa[ax2 * database_count + idx] - query_point[ax2];
      const float d3 = soa[ax3 * database_count + idx] - query_point[ax3];
      distance += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
      if (distance > limit) {
        break;
      }
    }
  } else if constexpr (StaticDimension != 0) {
    // Non-multiple-of-4 compile-time D: reorder axes and check one at a time.
#pragma unroll
    for (int a = 0; a < StaticDimension; ++a) {
      const int axis = (StaticDimension > 4) ? (a + kGD) % StaticDimension : a;
      const float difference =
          soa[axis * database_count + idx] - query_point[axis];
      distance += difference * difference;
      if (distance > limit) {
        break;
      }
    }
  } else {
    // Runtime dimension: 1 axis per iteration with early exit.
    for (int axis = 0; axis < dimension; ++axis) {
      const float difference =
          soa[axis * database_count + idx] - query_point[axis];
      distance += difference * difference;
      if (distance > limit) {
        break;
      }
    }
  }
  return distance;
}

template <int StaticDimension>
__global__ void findNeighborsBruteForce(
    const float* query, int query_count, const float* database,
    int database_count, int dimension, float radius, int max_neighbors,
    bool exclude_self, bool inputs_are_same, float* neighbor_distances,
    int* neighbor_indices) {
  const float radius_squared = radius * radius;
  for (int query_index = blockIdx.x * blockDim.x + threadIdx.x;
       query_index < query_count;
       query_index += blockDim.x * gridDim.x) {
    const float* query_point = query + query_index * dimension;
    float cached_query[StaticDimension == 0 ? 1 : StaticDimension];
    if constexpr (StaticDimension != 0) {
#pragma unroll
      for (int axis = 0; axis < StaticDimension; ++axis) {
        cached_query[axis] = query_point[axis];
      }
      query_point = cached_query;
    }
    float* distances =
        neighbor_distances + static_cast<std::int64_t>(query_index) *
                                 max_neighbors;
    int* indices =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    int found = 0;
    bool heap_mode = false;

    for (int database_index = 0; database_index < database_count;
         ++database_index) {
      if (exclude_self && inputs_are_same &&
          database_index == query_index) {
        continue;
      }
      const float distance = squaredDistance<StaticDimension>(
          database + database_index * dimension,
          query_point, dimension, radius_squared);
      if (distance <= radius_squared) {
        selectNeighbor(distance, database_index, max_neighbors, found,
                       heap_mode, distances, indices);
      }
    }
    if (heap_mode) {
      sortNeighborHeap(distances, indices, found);
    }
    if (found < max_neighbors) {
      indices[found] = -1;
    }
  }
}


template <int StaticDimension>
__global__ __launch_bounds__(kSearchThreads, 8)
void findNeighbors(
    const float* query, int query_count,
    const float* sorted_database,
    const int* sorted_database_indices, int database_count, int dimension,
    const GridParameters* parameters, const int* grid_offsets,
    float radius, int max_neighbors, bool exclude_self, bool inputs_are_same,
    const int* query_indices, float* neighbor_distances,
    int* neighbor_indices) {
  const float radius_squared = radius * radius;
  for (int work_index = blockIdx.x * blockDim.x + threadIdx.x;
       work_index < query_count;
       work_index += blockDim.x * gridDim.x) {
    const int query_index =
        query_indices == nullptr ? work_index : query_indices[work_index];
    // Read query coordinates from AoS query.data using query_index.
    // For identical-set, query_index is the original index (via sorted_database_indices),
    // so this reads the correct original-order coordinates from AoS query.data.
    const float* query_point = query + query_index * dimension;
    float cached_query[StaticDimension == 0 ? 1 : StaticDimension];
    if constexpr (StaticDimension != 0) {
#pragma unroll
      for (int axis = 0; axis < StaticDimension; ++axis) {
        cached_query[axis] = query_point[axis];
      }
      query_point = cached_query;
    }
    float* distances =
        neighbor_distances + static_cast<std::int64_t>(query_index) *
                                 max_neighbors;
    int* indices =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    int found = 0;
    bool heap_mode = false;

    int minimum_cell[kGridDimensions] = {0, 0, 0, 0};
    int maximum_cell[kGridDimensions] = {0, 0, 0, 0};
    for (int axis = 0; axis < parameters->grid_dimensions; ++axis) {
      const float coordinate = query_point[axis];
      minimum_cell[axis] = max(
          0, static_cast<int>(floorf(
                 (coordinate - parameters->minimum[axis] - radius) *
                 parameters->inverse_cell_size)));
      maximum_cell[axis] =
          min(parameters->resolution[axis] - 1,
              static_cast<int>(floorf(
                  (coordinate - parameters->minimum[axis] + radius) *
                  parameters->inverse_cell_size)));
    }

    const float cell_size = 1.0f / parameters->inverse_cell_size;
    for (int x = minimum_cell[0]; x <= maximum_cell[0]; ++x) {
      for (int y = minimum_cell[1]; y <= maximum_cell[1]; ++y) {
        for (int z = minimum_cell[2]; z <= maximum_cell[2]; ++z) {
          for (int w = minimum_cell[3]; w <= maximum_cell[3]; ++w) {
            // Skip cell if its minimum 4D distance to the query already
            // exceeds the search radius (no candidate in the cell can pass).
            {
              const float cn0 = parameters->minimum[0] + x * cell_size;
              const float g0 = fmaxf(0.0f, fmaxf(cn0 - query_point[0],
                                                   query_point[0] - cn0 - cell_size));
              float sq = g0 * g0;
              const float cn1 = parameters->minimum[1] + y * cell_size;
              const float g1 = fmaxf(0.0f, fmaxf(cn1 - query_point[1],
                                                   query_point[1] - cn1 - cell_size));
              sq += g1 * g1;
              if (sq > radius_squared) continue;
              const float cn2 = parameters->minimum[2] + z * cell_size;
              const float g2 = fmaxf(0.0f, fmaxf(cn2 - query_point[2],
                                                   query_point[2] - cn2 - cell_size));
              sq += g2 * g2;
              if (sq > radius_squared) continue;
              if (parameters->grid_dimensions >= 4) {
                const float cn3 = parameters->minimum[3] + w * cell_size;
                const float g3 = fmaxf(0.0f, fmaxf(cn3 - query_point[3],
                                                     query_point[3] - cn3 - cell_size));
                sq += g3 * g3;
                if (sq > radius_squared) continue;
              }
            }
            const int cell =
                ((x * parameters->resolution[1] + y) *
                     parameters->resolution[2] +
                 z) *
                    parameters->resolution[3] +
                w;
            const int begin = grid_offsets[cell];
            const int end =
                cell + 1 < parameters->total_cells
                    ? grid_offsets[cell + 1]
                    : database_count;
            for (int sorted_index = begin; sorted_index < end;
                 ++sorted_index) {
              const int database_index =
                  sorted_database_indices[sorted_index];
              if (exclude_self && inputs_are_same &&
                  database_index == query_index) {
                continue;
              }
              const float distance = squaredDistanceSoA<StaticDimension>(
                  sorted_database, sorted_index, database_count, query_point,
                  dimension, radius_squared);
              if (distance > radius_squared) {
                continue;
              }
              selectNeighbor(distance, database_index, max_neighbors, found,
                             heap_mode, distances, indices);
            }
          }
        }
      }
    }
    if (heap_mode) {
      sortNeighborHeap(distances, indices, found);
    }
    if (found < max_neighbors) {
      indices[found] = -1;
    }
  }
}

// Warp-cooperative neighbor search: 32 threads share one query.
// Each lane strides over candidates (begin+lane, begin+lane+32, …) so 32
// independent memory requests are in flight simultaneously.  Passing
// candidates are appended to the output row via a fast shared-memory
// atomic counter; lane 0 does a single insertion sort of the small final
// list.  Dispatched when max_neighbors >= 64 to avoid overhead on tiny K.
template <int StaticDimension>
__global__ void findNeighborsWarpCoop(
    const float* query, int query_count, const float* sorted_database,
    const int* sorted_database_indices, int database_count, int dimension,
    const GridParameters* parameters, const int* grid_offsets, float radius,
    int max_neighbors, bool exclude_self, bool inputs_are_same,
    const int* query_indices, float* neighbor_distances,
    int* neighbor_indices) {
  constexpr int kWarpsPerBlock = kThreads / 32;
  const float radius_squared = radius * radius;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int work_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (work_index >= query_count) {
    return;
  }

  const int query_index =
      query_indices == nullptr ? work_index : query_indices[work_index];
  const float* query_point = query + query_index * dimension;
  float cached_query[StaticDimension == 0 ? 1 : StaticDimension];
  if constexpr (StaticDimension != 0) {
#pragma unroll
    for (int axis = 0; axis < StaticDimension; ++axis) {
      cached_query[axis] = query_point[axis];
    }
    query_point = cached_query;
  }

  float* distances =
      neighbor_distances +
      static_cast<std::int64_t>(query_index) * max_neighbors;
  int* indices =
      neighbor_indices +
      static_cast<std::int64_t>(query_index) * max_neighbors;

  // One counter per warp tracks how many candidates passed the radius test.
  __shared__ int warp_found[kWarpsPerBlock];
  if (lane == 0) {
    warp_found[warp_in_block] = 0;
  }
  __syncwarp();

  int minimum_cell[kGridDimensions] = {0, 0, 0, 0};
  int maximum_cell[kGridDimensions] = {0, 0, 0, 0};
  for (int axis = 0; axis < parameters->grid_dimensions; ++axis) {
    const float coordinate = query_point[axis];
    minimum_cell[axis] = max(
        0, static_cast<int>(floorf(
               (coordinate - parameters->minimum[axis] - radius) *
               parameters->inverse_cell_size)));
    maximum_cell[axis] =
        min(parameters->resolution[axis] - 1,
            static_cast<int>(floorf(
                (coordinate - parameters->minimum[axis] + radius) *
                parameters->inverse_cell_size)));
  }
  const float cell_size = 1.0f / parameters->inverse_cell_size;

  for (int x = minimum_cell[0]; x <= maximum_cell[0]; ++x) {
    for (int y = minimum_cell[1]; y <= maximum_cell[1]; ++y) {
      for (int z = minimum_cell[2]; z <= maximum_cell[2]; ++z) {
        for (int w = minimum_cell[3]; w <= maximum_cell[3]; ++w) {
          {
            const float cn0 = parameters->minimum[0] + x * cell_size;
            const float g0 = fmaxf(0.0f, fmaxf(cn0 - query_point[0],
                                                 query_point[0] - cn0 - cell_size));
            float sq = g0 * g0;
            const float cn1 = parameters->minimum[1] + y * cell_size;
            const float g1 = fmaxf(0.0f, fmaxf(cn1 - query_point[1],
                                                 query_point[1] - cn1 - cell_size));
            sq += g1 * g1;
            if (sq > radius_squared) continue;
            const float cn2 = parameters->minimum[2] + z * cell_size;
            const float g2 = fmaxf(0.0f, fmaxf(cn2 - query_point[2],
                                                 query_point[2] - cn2 - cell_size));
            sq += g2 * g2;
            if (sq > radius_squared) continue;
            if (parameters->grid_dimensions >= 4) {
              const float cn3 = parameters->minimum[3] + w * cell_size;
              const float g3 = fmaxf(0.0f, fmaxf(cn3 - query_point[3],
                                                   query_point[3] - cn3 - cell_size));
              sq += g3 * g3;
              if (sq > radius_squared) continue;
            }
          }
          const int cell =
              ((x * parameters->resolution[1] + y) *
                   parameters->resolution[2] +
               z) *
                  parameters->resolution[3] +
              w;
          const int begin = grid_offsets[cell];
          const int end = cell + 1 < parameters->total_cells
                              ? grid_offsets[cell + 1]
                              : database_count;
          for (int si = begin + lane; si < end; si += 32) {
            const int database_index = sorted_database_indices[si];
            if (exclude_self && inputs_are_same &&
                database_index == query_index) {
              continue;
            }
            const float distance = squaredDistanceSoA<StaticDimension>(
                sorted_database, si, database_count, query_point, dimension,
                radius_squared);
            if (distance <= radius_squared) {
              const int pos =
                  atomicAdd(&warp_found[warp_in_block], 1);
              if (pos < max_neighbors) {
                distances[pos] = distance;
                indices[pos] = database_index;
              }
            }
          }
        }
      }
    }
  }
  // Ensure all lanes have finished writing before lane 0 reads.
  __syncwarp();

  if (lane == 0) {
    const int found = min(warp_found[warp_in_block], max_neighbors);
    // Insertion sort in global memory: found is small (avg ~34 for real workload).
    for (int i = 1; i < found; ++i) {
      const float d = distances[i];
      const int idx = indices[i];
      int j = i;
      while (j > 0 &&
             precedes(d, idx, distances[j - 1], indices[j - 1])) {
        distances[j] = distances[j - 1];
        indices[j] = indices[j - 1];
        --j;
      }
      distances[j] = d;
      indices[j] = idx;
    }
    if (found < max_neighbors) {
      indices[found] = -1;
    }
  }
}

void launchBruteForceNeighbors(
    const float* query, int query_count, const float* database,
    int database_count, int dimension, float radius, int max_neighbors,
    bool exclude_self, bool inputs_are_same, float* neighbor_distances,
    int* neighbor_indices, cudaStream_t stream) {
  const int blocks = blocksFor(query_count);
#define FRNN_LAUNCH_BRUTE_FORCE(Dimension)                                  \
  findNeighborsBruteForce<Dimension>                                       \
      <<<blocks, kThreads, 0, stream>>>(                                   \
          query, query_count, database, database_count, dimension, radius, \
          max_neighbors, exclude_self, inputs_are_same,                    \
          neighbor_distances, neighbor_indices)
  switch (dimension) {
    case 1:
      FRNN_LAUNCH_BRUTE_FORCE(1);
      break;
    case 2:
      FRNN_LAUNCH_BRUTE_FORCE(2);
      break;
    case 3:
      FRNN_LAUNCH_BRUTE_FORCE(3);
      break;
    case 4:
      FRNN_LAUNCH_BRUTE_FORCE(4);
      break;
    case 8:
      FRNN_LAUNCH_BRUTE_FORCE(8);
      break;
    case 12:
      FRNN_LAUNCH_BRUTE_FORCE(12);
      break;
    case 16:
      FRNN_LAUNCH_BRUTE_FORCE(16);
      break;
    default:
      FRNN_LAUNCH_BRUTE_FORCE(0);
      break;
  }
#undef FRNN_LAUNCH_BRUTE_FORCE
}

void launchGridNeighbors(
    const float* query, int query_count, const float* sorted_database,
    const int* sorted_database_indices, int database_count, int dimension,
    const GridParameters* parameters, const int* grid_offsets,
    float radius, int max_neighbors, bool exclude_self, bool inputs_are_same,
    const int* query_indices, float* neighbor_distances,
    int* neighbor_indices, cudaStream_t stream) {
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (query_count + kSearchThreads - 1) / kSearchThreads, 65535));
#define FRNN_LAUNCH_GRID(Dimension)                                        \
  findNeighbors<Dimension><<<blocks, kSearchThreads, 0, stream>>>(         \
      query, query_count, sorted_database, sorted_database_indices,          \
      database_count, dimension, parameters, grid_offsets, radius,           \
      max_neighbors, exclude_self, inputs_are_same, query_indices,           \
      neighbor_distances, neighbor_indices)
  switch (dimension) {
    case 1:
      FRNN_LAUNCH_GRID(1);
      break;
    case 2:
      FRNN_LAUNCH_GRID(2);
      break;
    case 3:
      FRNN_LAUNCH_GRID(3);
      break;
    case 4:
      FRNN_LAUNCH_GRID(4);
      break;
    case 8:
      FRNN_LAUNCH_GRID(8);
      break;
    case 12:
      FRNN_LAUNCH_GRID(12);
      break;
    case 16:
      FRNN_LAUNCH_GRID(16);
      break;
    default:
      FRNN_LAUNCH_GRID(0);
      break;
  }
#undef FRNN_LAUNCH_GRID
}

void launchWarpCoopNeighbors(
    const float* query, int query_count, const float* sorted_database,
    const int* sorted_database_indices, int database_count, int dimension,
    const GridParameters* parameters, const int* grid_offsets, float radius,
    int max_neighbors, bool exclude_self, bool inputs_are_same,
    const int* query_indices, float* neighbor_distances,
    int* neighbor_indices, cudaStream_t stream) {
  // Each block contains kThreads/32 warps, one warp per query.
  constexpr int kQueriesPerBlock = kThreads / 32;
  const int blocks = static_cast<int>(
      std::min<int64_t>((query_count + kQueriesPerBlock - 1) / kQueriesPerBlock,
                        65535));
#define FRNN_LAUNCH_WARP_COOP(Dimension)                                    \
  findNeighborsWarpCoop<Dimension><<<blocks, kThreads, 0, stream>>>(        \
      query, query_count, sorted_database, sorted_database_indices,         \
      database_count, dimension, parameters, grid_offsets, radius,          \
      max_neighbors, exclude_self, inputs_are_same, query_indices,          \
      neighbor_distances, neighbor_indices)
  switch (dimension) {
    case 1:
      FRNN_LAUNCH_WARP_COOP(1);
      break;
    case 2:
      FRNN_LAUNCH_WARP_COOP(2);
      break;
    case 3:
      FRNN_LAUNCH_WARP_COOP(3);
      break;
    case 4:
      FRNN_LAUNCH_WARP_COOP(4);
      break;
    case 8:
      FRNN_LAUNCH_WARP_COOP(8);
      break;
    case 12:
      FRNN_LAUNCH_WARP_COOP(12);
      break;
    case 16:
      FRNN_LAUNCH_WARP_COOP(16);
      break;
    default:
      FRNN_LAUNCH_WARP_COOP(0);
      break;
  }
#undef FRNN_LAUNCH_WARP_COOP
}

__global__ void countEdges(const int* neighbor_indices,
                           int query_count, int max_neighbors,
                           bool undirected, std::int64_t* edge_counts) {
  for (int query_index = blockIdx.x * blockDim.x + threadIdx.x;
       query_index < query_count;
       query_index += blockDim.x * gridDim.x) {
    std::int64_t count = 0;
    const int* neighbors =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
      const int target = neighbors[neighbor];
      if (target < 0) {
        break;
      }
      if (!undirected || query_index > target) {
        ++count;
      }
    }
    edge_counts[query_index] = count;
  }
}

__global__ void writeEdges(const int* neighbor_indices,
                           int query_count, int max_neighbors,
                           bool undirected,
                           const std::int64_t* edge_offsets,
                           std::int64_t* edges) {
  for (int query_index = blockIdx.x * blockDim.x + threadIdx.x;
       query_index < query_count;
       query_index += blockDim.x * gridDim.x) {
    std::int64_t output_index = edge_offsets[query_index];
    const int* neighbors =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
      const int target = neighbors[neighbor];
      if (target < 0) {
        break;
      }
      if (!undirected || query_index > target) {
        edges[output_index * 2] = query_index;
        edges[output_index * 2 + 1] = target;
        ++output_index;
      }
    }
  }
}

template <typename T>
void allocateDevice(T** pointer, std::size_t count, const char* name) {
  if (count == 0) {
    *pointer = nullptr;
    return;
  }
  checkCuda(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)),
            name);
}

template <typename T>
void freeDevice(T*& pointer) noexcept {
  if (pointer != nullptr) {
    cudaFree(pointer);
    pointer = nullptr;
  }
}

SearchAlgorithm resolveAlgorithm(SearchAlgorithm requested,
                                 std::int64_t query_count,
                                 std::int64_t database_count, int dimension,
                                 int max_neighbors) {
  if (requested != SearchAlgorithm::automatic) {
    return requested;
  }
  if (query_count == 0 || database_count == 0) {
    return SearchAlgorithm::brute_force;
  }
  const std::int64_t database_dimensions =
      database_count > kBruteForceCoordinateWork / dimension
          ? kBruteForceCoordinateWork + 1
          : database_count * dimension;
  const std::int64_t selection_factor =
      std::max<std::int64_t>(1, (max_neighbors + 7) / 8);
  if (database_dimensions > kBruteForceCoordinateWork / selection_factor) {
    return SearchAlgorithm::grid;
  }
  const std::int64_t work_per_query =
      database_dimensions * selection_factor;
  const bool fits =
      query_count <= kBruteForceCoordinateWork / work_per_query;
  return fits ? SearchAlgorithm::brute_force : SearchAlgorithm::grid;
}

}  // namespace

struct Workspace::Impl {
  int device = -1;
  std::int64_t query_capacity = 0;
  std::int64_t database_capacity = 0;
  int dimension_capacity = 0;
  int neighbor_capacity = 0;
  bool has_grid_storage = false;
  int grid_cell_capacity = 0;

  float* bounds = nullptr;
  GridParameters* grid_parameters = nullptr;
  int* grid_counts = nullptr;
  int* grid_offsets = nullptr;
  int* point_cells = nullptr;
  int* point_cell_indices = nullptr;
  float* sorted_database = nullptr;
  int* sorted_database_indices = nullptr;
  float* neighbor_distances = nullptr;
  int* neighbor_indices = nullptr;
  std::int64_t* edge_counts = nullptr;
  std::int64_t* edge_offsets = nullptr;
  void* scan_temporary = nullptr;
  std::size_t scan_temporary_bytes = 0;
};

Workspace::Workspace() : impl_(std::make_unique<Impl>()) {}
Workspace::~Workspace() { clear(); }
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&& other) noexcept {
  if (this != &other) {
    clear();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void Workspace::clear() noexcept {
  if (!impl_) {
    return;
  }
  int original_device = -1;
  cudaGetDevice(&original_device);
  if (impl_->device >= 0 && impl_->device != original_device) {
    cudaSetDevice(impl_->device);
  }
  freeDevice(impl_->bounds);
  freeDevice(impl_->grid_parameters);
  freeDevice(impl_->grid_counts);
  freeDevice(impl_->grid_offsets);
  freeDevice(impl_->point_cells);
  freeDevice(impl_->point_cell_indices);
  freeDevice(impl_->sorted_database);
  freeDevice(impl_->sorted_database_indices);
  freeDevice(impl_->neighbor_distances);
  freeDevice(impl_->neighbor_indices);
  freeDevice(impl_->edge_counts);
  freeDevice(impl_->edge_offsets);
  if (impl_->scan_temporary != nullptr) {
    cudaFree(impl_->scan_temporary);
    impl_->scan_temporary = nullptr;
  }
  if (impl_->device >= 0 && impl_->device != original_device &&
      original_device >= 0) {
    cudaSetDevice(original_device);
  }
  *impl_ = Impl{};
}

void Workspace::reserve(std::int64_t max_query_points,
                        std::int64_t max_database_points, int dimension,
                        int max_neighbors, SearchAlgorithm algorithm) {
  validateView(max_query_points, dimension,
               max_query_points == 0 ? nullptr
                                     : reinterpret_cast<const float*>(1),
               "max_query_points");
  validateView(max_database_points, dimension,
               max_database_points == 0 ? nullptr
                                        : reinterpret_cast<const float*>(1),
               "max_database_points");
  if (max_neighbors < 0) {
    throw std::invalid_argument("max_neighbors must be non-negative");
  }
  requiredEdgeCapacity(max_query_points, max_neighbors);
  const SearchAlgorithm resolved_algorithm =
      resolveAlgorithm(algorithm, max_query_points, max_database_points,
                       dimension, max_neighbors);
  const bool needs_grid_storage =
      resolved_algorithm == SearchAlgorithm::grid;
  const int required_grid_cells =
      needs_grid_storage ? maximumGridCells(dimension) : 0;

  int current_device = -1;
  checkCuda(cudaGetDevice(&current_device), "cudaGetDevice");
  if (impl_->device == current_device &&
      impl_->query_capacity >= max_query_points &&
      impl_->database_capacity >= max_database_points &&
      impl_->dimension_capacity >= dimension &&
      impl_->neighbor_capacity >= max_neighbors &&
      (!needs_grid_storage ||
       (impl_->has_grid_storage &&
        impl_->grid_cell_capacity >= required_grid_cells))) {
    return;
  }

  const std::int64_t query_capacity =
      std::max(impl_->query_capacity, max_query_points);
  const std::int64_t database_capacity =
      std::max(impl_->database_capacity, max_database_points);
  const int dimension_capacity =
      std::max(impl_->dimension_capacity, dimension);
  const int neighbor_capacity =
      std::max(impl_->neighbor_capacity, max_neighbors);
  clear();
  impl_->device = current_device;
  impl_->query_capacity = query_capacity;
  impl_->database_capacity = database_capacity;
  impl_->dimension_capacity = dimension_capacity;
  impl_->neighbor_capacity = neighbor_capacity;
  impl_->has_grid_storage = needs_grid_storage;
  impl_->grid_cell_capacity = required_grid_cells;

  try {
    if (needs_grid_storage) {
      allocateDevice(&impl_->bounds, 2 * kGridDimensions, "allocate bounds");
      allocateDevice(&impl_->grid_parameters, 1, "allocate grid parameters");
      allocateDevice(&impl_->grid_counts, required_grid_cells,
                     "allocate grid counts");
      allocateDevice(&impl_->grid_offsets, required_grid_cells,
                     "allocate grid offsets");
      allocateDevice(&impl_->point_cells,
                     static_cast<std::size_t>(database_capacity),
                     "allocate point cells");
      allocateDevice(&impl_->point_cell_indices,
                     static_cast<std::size_t>(database_capacity),
                     "allocate point cell indices");
      allocateDevice(
          &impl_->sorted_database,
          static_cast<std::size_t>(database_capacity) * dimension_capacity,
          "allocate sorted database");
      allocateDevice(&impl_->sorted_database_indices,
                     static_cast<std::size_t>(database_capacity),
                     "allocate sorted database indices");
    }
    const std::size_t neighbor_values =
        static_cast<std::size_t>(query_capacity) * neighbor_capacity;
    allocateDevice(&impl_->neighbor_distances, neighbor_values,
                   "allocate neighbor distances");
    allocateDevice(&impl_->neighbor_indices, neighbor_values,
                   "allocate neighbor indices");
    allocateDevice(&impl_->edge_counts,
                   static_cast<std::size_t>(query_capacity) + 1,
                   "allocate edge counts");
    allocateDevice(&impl_->edge_offsets,
                   static_cast<std::size_t>(query_capacity) + 1,
                   "allocate edge offsets");

    std::size_t grid_scan_bytes = 0;
    if (needs_grid_storage) {
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, grid_scan_bytes, impl_->grid_counts,
                    impl_->grid_offsets, required_grid_cells),
                "size grid prefix-sum workspace");
    }
    std::size_t edge_scan_bytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  nullptr, edge_scan_bytes, impl_->edge_counts,
                  impl_->edge_offsets,
                  static_cast<int>(query_capacity + 1)),
              "size edge prefix-sum workspace");
    impl_->scan_temporary_bytes =
        std::max(grid_scan_bytes, edge_scan_bytes);
    checkCuda(cudaMalloc(&impl_->scan_temporary,
                         impl_->scan_temporary_bytes),
              "allocate prefix-sum workspace");
  } catch (...) {
    clear();
    throw;
  }
}

std::int64_t requiredEdgeCapacity(std::int64_t query_points,
                                  int max_neighbors) {
  if (query_points < 0 || max_neighbors < 0) {
    throw std::invalid_argument(
        "query_points and max_neighbors must be non-negative");
  }
  if (max_neighbors != 0 &&
      query_points >
          std::numeric_limits<std::int64_t>::max() / max_neighbors) {
    throw std::overflow_error("edge capacity exceeds int64 range");
  }
  return query_points * max_neighbors;
}

void buildEdgesAsync(DevicePointView query, DevicePointView database,
                     DeviceEdgeBuffer output, float radius, int max_neighbors,
                     BuildOptions options, Workspace& workspace,
                     cudaStream_t stream) {
  validateView(query.size, query.dimension, query.data, "query");
  validateView(database.size, database.dimension, database.data, "database");
  if (query.dimension != database.dimension) {
    throw std::invalid_argument(
        "query and database dimensions must be identical");
  }
  validateConfiguration(radius, max_neighbors);
  const std::int64_t required_capacity =
      requiredEdgeCapacity(query.size, max_neighbors);
  if (output.edge_count == nullptr) {
    throw std::invalid_argument("output.edge_count must not be null");
  }
  const bool count_only = output.edges == nullptr;
  if (!count_only && output.capacity < required_capacity) {
    throw std::invalid_argument(
        "output.capacity is smaller than requiredEdgeCapacity");
  }
  if (options.undirected && !options.inputs_are_same) {
    throw std::invalid_argument(
        "undirected output requires inputs_are_same=true");
  }
  if (options.inputs_are_same && query.size != database.size) {
    throw std::invalid_argument(
        "inputs_are_same requires equal query and database sizes");
  }

  if (query.size == 0 || database.size == 0 || max_neighbors == 0) {
    checkCuda(cudaMemsetAsync(output.edge_count, 0, sizeof(std::int64_t),
                              stream),
              "clear edge count");
    return;
  }

  const SearchAlgorithm algorithm =
      resolveAlgorithm(options.algorithm, query.size, database.size,
                       query.dimension, max_neighbors);
  workspace.reserve(query.size, database.size, query.dimension,
                    max_neighbors, algorithm);
  Workspace::Impl& memory = *workspace.impl_;
  const int query_count = static_cast<int>(query.size);
  const int database_count = static_cast<int>(database.size);
  const int grid_cell_capacity = maximumGridCells(query.dimension);
  recordProfileEvent(0, stream);
  if (algorithm == SearchAlgorithm::brute_force) {
    for (int event = 1; event <= 4; ++event) {
      recordProfileEvent(event, stream);
    }
    launchBruteForceNeighbors(
        query.data, query_count, database.data, database_count,
        query.dimension, radius, max_neighbors, options.exclude_self,
        options.inputs_are_same, memory.neighbor_distances,
        memory.neighbor_indices, stream);
    checkCuda(cudaPeekAtLastError(), "launch findNeighborsBruteForce");
  } else {
    const int grid_dimensions = gridDimensionCount(query.dimension);
    initializeBounds<<<1, kGridDimensions, 0, stream>>>(memory.bounds);
    checkCuda(cudaPeekAtLastError(), "launch initializeBounds");
    computeBounds<<<blocksFor(database.size), kThreads, 0, stream>>>(
        database.data, database_count, database.dimension, grid_dimensions,
        memory.bounds);
    checkCuda(cudaPeekAtLastError(), "launch computeBounds");
    finalizeGrid<<<1, 1, 0, stream>>>(
        memory.bounds, grid_dimensions, grid_cell_capacity, radius,
        memory.grid_parameters);
    checkCuda(cudaPeekAtLastError(), "launch finalizeGrid");
    recordProfileEvent(1, stream);

    checkCuda(cudaMemsetAsync(memory.grid_counts, 0,
                              grid_cell_capacity * sizeof(int), stream),
              "clear grid counts");
    insertPoints<<<blocksFor(database.size), kThreads, 0, stream>>>(
        database.data, database_count, database.dimension,
        memory.grid_parameters, memory.grid_counts, memory.point_cells,
        memory.point_cell_indices);
    checkCuda(cudaPeekAtLastError(), "launch insertPoints");
    recordProfileEvent(2, stream);
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  memory.scan_temporary, memory.scan_temporary_bytes,
                  memory.grid_counts, memory.grid_offsets,
                  grid_cell_capacity,
                  stream),
              "prefix sum grid counts");
    recordProfileEvent(3, stream);
    countingSort<<<blocksFor(database.size), kThreads, 0, stream>>>(
        database.data, database_count, database.dimension, grid_dimensions,
        memory.point_cells,
        memory.point_cell_indices, memory.grid_offsets,
        memory.sorted_database, memory.sorted_database_indices);
    checkCuda(cudaPeekAtLastError(), "launch countingSort");
    recordProfileEvent(4, stream);

    // For identical-set, iterate queries in spatial order for better cache
    // locality on the sorted_database SoA access.
    const float* search_query = query.data;
    const int* search_query_indices =
        options.inputs_are_same ? memory.sorted_database_indices : nullptr;
    launchGridNeighbors(
        search_query, query_count, memory.sorted_database,
        memory.sorted_database_indices, database_count, query.dimension,
        memory.grid_parameters, memory.grid_offsets,
        radius, max_neighbors,
        options.exclude_self, options.inputs_are_same,
        search_query_indices, memory.neighbor_distances,
        memory.neighbor_indices, stream);
    checkCuda(cudaPeekAtLastError(), "launch findNeighbors");
  }
  recordProfileEvent(5, stream);
  checkCuda(cudaMemsetAsync(memory.edge_counts, 0,
                            (query.size + 1) * sizeof(std::int64_t), stream),
            "clear edge counts");
  countEdges<<<blocksFor(query.size), kThreads, 0, stream>>>(
      memory.neighbor_indices, query_count, max_neighbors, options.undirected,
      memory.edge_counts);
  checkCuda(cudaPeekAtLastError(), "launch countEdges");
  recordProfileEvent(6, stream);
  checkCuda(cub::DeviceScan::ExclusiveSum(
                memory.scan_temporary, memory.scan_temporary_bytes,
                memory.edge_counts, memory.edge_offsets, query_count + 1,
                stream),
            "prefix sum edge counts");
  recordProfileEvent(7, stream);
  if (!count_only) {
    writeEdges<<<blocksFor(query.size), kThreads, 0, stream>>>(
        memory.neighbor_indices, query_count, max_neighbors,
        options.undirected, memory.edge_offsets, output.edges);
    checkCuda(cudaPeekAtLastError(), "launch writeEdges");
  }
  recordProfileEvent(8, stream);
  checkCuda(cudaMemcpyAsync(
                output.edge_count, memory.edge_offsets + query_count,
                sizeof(std::int64_t), cudaMemcpyDeviceToDevice, stream),
            "copy edge count");
  recordProfileEvent(9, stream);
}

#if defined(FRNN_ENABLE_STAGE_PROFILING)
namespace detail {

void profileBuildEdgesAsync(DevicePointView query, DevicePointView database,
                            DeviceEdgeBuffer output, float radius,
                            int max_neighbors, BuildOptions options,
                            Workspace& workspace, cudaStream_t stream,
                            float* stage_milliseconds, int stage_count) {
  if (stage_milliseconds == nullptr ||
      stage_count != kProfileEventCount - 1) {
    throw std::invalid_argument("invalid stage profiling output");
  }
  cudaEvent_t events[kProfileEventCount] = {};
  for (cudaEvent_t& event : events) {
    checkCuda(cudaEventCreate(&event), "create profiling event");
  }
  try {
    active_profile_events = events;
    buildEdgesAsync(query, database, output, radius, max_neighbors, options,
                    workspace, stream);
    active_profile_events = nullptr;
    checkCuda(cudaEventSynchronize(events[kProfileEventCount - 1]),
              "synchronize profiling events");
    for (int stage = 0; stage < stage_count; ++stage) {
      checkCuda(cudaEventElapsedTime(&stage_milliseconds[stage], events[stage],
                                     events[stage + 1]),
                "calculate stage elapsed time");
    }
  } catch (...) {
    active_profile_events = nullptr;
    for (cudaEvent_t event : events) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    throw;
  }
  for (cudaEvent_t event : events) {
    cudaEventDestroy(event);
  }
}

}  // namespace detail
#endif

std::vector<Edge> buildEdges(PointView query, PointView database, float radius,
                             int max_neighbors, BuildOptions options) {
  validateHostValues(query, "query");
  validateHostValues(database, "database");
  if (query.dimension != database.dimension) {
    throw std::invalid_argument(
        "query and database dimensions must be identical");
  }
  validateConfiguration(radius, max_neighbors);
  const std::int64_t capacity =
      requiredEdgeCapacity(query.size, max_neighbors);
  if (capacity == 0 || database.size == 0) {
    return {};
  }

  float* device_query = nullptr;
  float* device_database = nullptr;
  std::int64_t* device_edges = nullptr;
  std::int64_t* device_count = nullptr;
  cudaStream_t stream = nullptr;
  try {
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create CUDA stream");
    allocateDevice(
        &device_query,
        static_cast<std::size_t>(query.size) * query.dimension,
        "allocate device query");
    const bool alias_inputs =
        options.inputs_are_same && query.data == database.data &&
        query.size == database.size;
    if (alias_inputs) {
      device_database = device_query;
    } else {
      allocateDevice(
          &device_database,
          static_cast<std::size_t>(database.size) * database.dimension,
          "allocate device database");
    }
    allocateDevice(&device_count, 1, "allocate device edge count");

    checkCuda(cudaMemcpyAsync(
                  device_query, query.data,
                  static_cast<std::size_t>(query.size) * query.dimension *
                      sizeof(float),
                  cudaMemcpyHostToDevice, stream),
              "copy query to device");
    if (!alias_inputs) {
      checkCuda(cudaMemcpyAsync(
                    device_database, database.data,
                    static_cast<std::size_t>(database.size) *
                        database.dimension * sizeof(float),
                    cudaMemcpyHostToDevice, stream),
                "copy database to device");
    }

    Workspace workspace;
    workspace.reserve(query.size, database.size, query.dimension,
                      max_neighbors, options.algorithm);
    buildEdgesAsync(
        {device_query, query.size, query.dimension},
        {device_database, database.size, database.dimension},
        {nullptr, 0, device_count}, radius, max_neighbors, options,
        workspace, stream);

    std::int64_t edge_count = 0;
    checkCuda(cudaMemcpyAsync(&edge_count, device_count, sizeof(edge_count),
                              cudaMemcpyDeviceToHost, stream),
              "copy edge count to host");
    checkCuda(cudaStreamSynchronize(stream), "synchronize edge count");
    std::vector<Edge> result(static_cast<std::size_t>(edge_count));
    if (edge_count > 0) {
      allocateDevice(&device_edges,
                     static_cast<std::size_t>(edge_count) * 2,
                     "allocate exact device edges");
      Workspace::Impl& memory = *workspace.impl_;
      writeEdges<<<blocksFor(query.size), kThreads, 0, stream>>>(
          memory.neighbor_indices, static_cast<int>(query.size),
          max_neighbors, options.undirected, memory.edge_offsets,
          device_edges);
      checkCuda(cudaPeekAtLastError(), "launch writeEdges");
      checkCuda(cudaMemcpyAsync(
                    result.data(), device_edges,
                    static_cast<std::size_t>(edge_count) * sizeof(Edge),
                    cudaMemcpyDeviceToHost, stream),
                "copy edges to host");
      checkCuda(cudaStreamSynchronize(stream), "synchronize edge output");
    }

    freeDevice(device_count);
    freeDevice(device_edges);
    if (!alias_inputs) {
      freeDevice(device_database);
    }
    freeDevice(device_query);
    checkCuda(cudaStreamDestroy(stream), "destroy CUDA stream");
    return result;
  } catch (...) {
    freeDevice(device_count);
    freeDevice(device_edges);
    if (device_database != device_query) {
      freeDevice(device_database);
    }
    freeDevice(device_query);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    throw;
  }
}

std::vector<Edge> buildEdges(PointView points, float radius,
                             int max_neighbors) {
  BuildOptions options;
  options.exclude_self = true;
  options.undirected = true;
  options.inputs_are_same = true;
  return buildEdges(points, points, radius, max_neighbors, options);
}

void buildEdges(std::vector<float>& query, std::vector<float>& database,
                std::vector<std::int64_t>& edge_list,
                std::int64_t num_spacepoints, int embedding_dim, float r_max,
                int k_max) {
  if (num_spacepoints < 0 || embedding_dim <= 0) {
    throw std::invalid_argument(
        "num_spacepoints and embedding_dim must be valid");
  }
  const std::size_t expected =
      static_cast<std::size_t>(num_spacepoints) * embedding_dim;
  if (query.size() != expected || database.size() != expected) {
    throw std::invalid_argument(
        "query and database vector sizes do not match the declared shape");
  }
  BuildOptions options;
  options.exclude_self = true;
  options.undirected = true;
  options.inputs_are_same = true;
  const std::vector<Edge> edges =
      buildEdges({query.data(), num_spacepoints, embedding_dim},
                 {database.data(), num_spacepoints, embedding_dim}, r_max,
                 k_max, options);
  edge_list.reserve(edge_list.size() + edges.size() * 2);
  for (const Edge& edge : edges) {
    edge_list.push_back(edge.source);
  }
  for (const Edge& edge : edges) {
    edge_list.push_back(edge.target);
  }
}

void buildEdges(std::vector<float>& points,
                std::vector<std::int64_t>& edge_list,
                std::int64_t num_spacepoints, int embedding_dim, float r_max,
                int k_max) {
  if (num_spacepoints < 0 || embedding_dim <= 0 ||
      points.size() !=
          static_cast<std::size_t>(num_spacepoints) * embedding_dim) {
    throw std::invalid_argument(
        "points vector size does not match the declared shape");
  }
  const std::vector<Edge> edges =
      buildEdges({points.data(), num_spacepoints, embedding_dim}, r_max, k_max);
  edge_list.reserve(edge_list.size() + edges.size() * 2);
  for (const Edge& edge : edges) {
    edge_list.push_back(edge.source);
  }
  for (const Edge& edge : edges) {
    edge_list.push_back(edge.target);
  }
}

}  // namespace frnn
