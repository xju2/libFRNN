#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace frnn {

struct PointView {
  const float* data = nullptr;
  std::int64_t size = 0;
  int dimension = 0;
};

struct DevicePointView {
  const float* data = nullptr;
  std::int64_t size = 0;
  int dimension = 0;
};

struct Edge {
  std::int64_t source;
  std::int64_t target;

  friend bool operator==(const Edge& lhs, const Edge& rhs) noexcept {
    return lhs.source == rhs.source && lhs.target == rhs.target;
  }
};

enum class SearchAlgorithm {
  automatic,
  grid,
  brute_force,
};

struct BuildOptions {
  bool exclude_self = false;
  bool undirected = false;
  bool inputs_are_same = false;
  SearchAlgorithm algorithm = SearchAlgorithm::automatic;
};

// Device output is a row-major [capacity, 2] array. edge_count is one
// device-resident int64 value populated by buildEdgesAsync. Passing
// edges=nullptr requests count-only execution and ignores capacity.
struct DeviceEdgeBuffer {
  std::int64_t* edges = nullptr;
  std::int64_t capacity = 0;
  std::int64_t* edge_count = nullptr;
};

class Workspace {
 public:
  Workspace();
  ~Workspace();
  Workspace(Workspace&&) noexcept;
  Workspace& operator=(Workspace&&) noexcept;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  // Preallocation is optional. Calling it before buildEdgesAsync avoids
  // allocation-related synchronization in the timed/asynchronous region.
  void reserve(std::int64_t max_query_points,
               std::int64_t max_database_points,
               int dimension,
               int max_neighbors,
               SearchAlgorithm algorithm = SearchAlgorithm::automatic);
  void clear() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend void buildEdgesAsync(DevicePointView, DevicePointView,
                              DeviceEdgeBuffer, float, int, BuildOptions,
                              Workspace&, cudaStream_t);
  friend void materializeEdgesAsync(std::int64_t*, std::int64_t, Workspace&,
                                    cudaStream_t);
  friend std::vector<Edge> buildEdges(PointView, PointView, float, int,
                                      BuildOptions);
};

// Returns the conservative device output capacity required for a call.
std::int64_t requiredEdgeCapacity(std::int64_t query_points,
                                  int max_neighbors);

// Fully device-resident asynchronous API. No query or database data is copied
// to the host. All kernels and CUDA operations are enqueued on stream.
void buildEdgesAsync(DevicePointView query,
                     DevicePointView database,
                     DeviceEdgeBuffer output,
                     float radius,
                     int max_neighbors,
                     BuildOptions options,
                     Workspace& workspace,
                     cudaStream_t stream);

// Writes the edges prepared by the most recent count-only buildEdgesAsync
// call using this workspace. capacity must be at least the device edge_count
// produced by that call. Enqueue this on the same stream as the count call.
void materializeEdgesAsync(std::int64_t* edges,
                           std::int64_t capacity,
                           Workspace& workspace,
                           cudaStream_t stream);

// Synchronous host convenience API. Inputs are copied to the device and the
// returned row-major Edge values are copied back to the host.
std::vector<Edge> buildEdges(PointView query,
                             PointView database,
                             float radius,
                             int max_neighbors,
                             BuildOptions options = {});

// Same-set convenience API: removes self-loops and emits each pair once as
// source > target.
std::vector<Edge> buildEdges(PointView points,
                             float radius,
                             int max_neighbors);

// Compatibility interfaces. edge_list uses the historical [2, num_edges]
// flattened layout: all source indices followed by all target indices.
void buildEdges(std::vector<float>& query,
                std::vector<float>& database,
                std::vector<std::int64_t>& edge_list,
                std::int64_t num_spacepoints,
                int embedding_dim,
                float r_max,
                int k_max);

void buildEdges(std::vector<float>& points,
                std::vector<std::int64_t>& edge_list,
                std::int64_t num_spacepoints,
                int embedding_dim,
                float r_max,
                int k_max);

}  // namespace frnn
