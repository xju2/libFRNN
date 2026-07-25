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

constexpr int kGridDimensions = 3;
constexpr int kMaximumGridResolution = 128;
constexpr int kMaximumGridCells =
    kMaximumGridResolution * kMaximumGridResolution *
    kMaximumGridResolution;
constexpr int kThreads = 256;

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

int blocksFor(std::int64_t count) {
  if (count <= 0) {
    return 1;
  }
  return static_cast<int>(
      std::min<std::int64_t>((count + kThreads - 1) / kThreads, 65535));
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
                             float radius, GridParameters* parameters) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float largest_extent = 0.0F;
  for (int axis = 0; axis < grid_dimensions; ++axis) {
    largest_extent =
        fmaxf(largest_extent,
              bounds[kGridDimensions + axis] - bounds[axis]);
  }
  float cell_size = fmaxf(radius * 0.5F,
                          largest_extent /
                              static_cast<float>(kMaximumGridResolution - 1));
  if (!(cell_size > 0.0F) || !isfinite(cell_size)) {
    cell_size = radius * 0.5F;
  }

  parameters->inverse_cell_size = 1.0F / cell_size;
  parameters->grid_dimensions = grid_dimensions;
  parameters->total_cells = 1;
  for (int axis = 0; axis < kGridDimensions; ++axis) {
    if (axis < grid_dimensions) {
      parameters->minimum[axis] = bounds[axis];
      int resolution = static_cast<int>(
                           floorf((bounds[kGridDimensions + axis] -
                                   bounds[axis]) /
                                  cell_size)) +
                       1;
      resolution = max(1, min(kMaximumGridResolution, resolution));
      parameters->resolution[axis] = resolution;
    } else {
      parameters->minimum[axis] = 0.0F;
      parameters->resolution[axis] = 1;
    }
    parameters->total_cells *= parameters->resolution[axis];
  }
}

__global__ void insertPoints(const float* points, int point_count,
                             int dimension,
                             const GridParameters* parameters,
                             int* grid_counts, int* point_cells,
                             int* point_cell_indices) {
  for (int point = blockIdx.x * blockDim.x + threadIdx.x;
       point < point_count; point += blockDim.x * gridDim.x) {
    int coordinate[kGridDimensions] = {0, 0, 0};
    for (int axis = 0; axis < parameters->grid_dimensions; ++axis) {
      coordinate[axis] = static_cast<int>(
          floorf((points[point * dimension + axis] -
                  parameters->minimum[axis]) *
                 parameters->inverse_cell_size));
      coordinate[axis] =
          max(0, min(parameters->resolution[axis] - 1, coordinate[axis]));
    }
    const int cell =
        (coordinate[0] * parameters->resolution[1] + coordinate[1]) *
            parameters->resolution[2] +
        coordinate[2];
    point_cells[point] = cell;
    point_cell_indices[point] = atomicAdd(&grid_counts[cell], 1);
  }
}

__global__ void countingSort(const float* points, int point_count,
                             int dimension, const int* point_cells,
                             const int* point_cell_indices,
                             const int* grid_offsets, float* sorted_points,
                             int* sorted_indices) {
  for (int point = blockIdx.x * blockDim.x + threadIdx.x;
       point < point_count; point += blockDim.x * gridDim.x) {
    const int destination =
        grid_offsets[point_cells[point]] + point_cell_indices[point];
    for (int axis = 0; axis < dimension; ++axis) {
      sorted_points[destination * dimension + axis] =
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

__global__ void findNeighbors(
    const float* query, int query_count, const float* sorted_database,
    const int* sorted_database_indices, int database_count, int dimension,
    const GridParameters* parameters, const int* grid_offsets, float radius,
    int max_neighbors, bool exclude_self, bool inputs_are_same,
    float* neighbor_distances, std::int64_t* neighbor_indices) {
  const float radius_squared = radius * radius;
  for (int query_index = blockIdx.x * blockDim.x + threadIdx.x;
       query_index < query_count;
       query_index += blockDim.x * gridDim.x) {
    float* distances =
        neighbor_distances + static_cast<std::int64_t>(query_index) *
                                 max_neighbors;
    std::int64_t* indices =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    int found = 0;
    for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
      distances[neighbor] = INFINITY;
      indices[neighbor] = -1;
    }

    int minimum_cell[kGridDimensions] = {0, 0, 0};
    int maximum_cell[kGridDimensions] = {0, 0, 0};
    for (int axis = 0; axis < parameters->grid_dimensions; ++axis) {
      const float coordinate = query[query_index * dimension + axis];
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

    for (int x = minimum_cell[0]; x <= maximum_cell[0]; ++x) {
      for (int y = minimum_cell[1]; y <= maximum_cell[1]; ++y) {
        for (int z = minimum_cell[2]; z <= maximum_cell[2]; ++z) {
          const int cell =
              (x * parameters->resolution[1] + y) *
                  parameters->resolution[2] +
              z;
          const int begin = grid_offsets[cell];
          const int end =
              cell + 1 < parameters->total_cells
                  ? grid_offsets[cell + 1]
                  : database_count;
          for (int sorted_index = begin; sorted_index < end;
               ++sorted_index) {
            const std::int64_t database_index =
                sorted_database_indices[sorted_index];
            if (exclude_self && inputs_are_same &&
                database_index == query_index) {
              continue;
            }

            float distance = 0.0F;
            for (int axis = 0; axis < dimension; ++axis) {
              const float difference =
                  sorted_database[sorted_index * dimension + axis] -
                  query[query_index * dimension + axis];
              distance += difference * difference;
            }
            if (distance > radius_squared) {
              continue;
            }

            int insertion = found;
            if (insertion > max_neighbors) {
              insertion = max_neighbors;
            }
            while (insertion > 0 &&
                   precedes(distance, database_index,
                            distances[insertion - 1], indices[insertion - 1])) {
              --insertion;
            }
            if (insertion >= max_neighbors) {
              continue;
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
        }
      }
    }
  }
}

__global__ void countEdges(const std::int64_t* neighbor_indices,
                           int query_count, int max_neighbors,
                           bool undirected, std::int64_t* edge_counts) {
  for (int query_index = blockIdx.x * blockDim.x + threadIdx.x;
       query_index < query_count;
       query_index += blockDim.x * gridDim.x) {
    std::int64_t count = 0;
    const std::int64_t* neighbors =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
      const std::int64_t target = neighbors[neighbor];
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

__global__ void writeEdges(const std::int64_t* neighbor_indices,
                           int query_count, int max_neighbors,
                           bool undirected,
                           const std::int64_t* edge_offsets,
                           std::int64_t* edges) {
  for (int query_index = blockIdx.x * blockDim.x + threadIdx.x;
       query_index < query_count;
       query_index += blockDim.x * gridDim.x) {
    std::int64_t output_index = edge_offsets[query_index];
    const std::int64_t* neighbors =
        neighbor_indices + static_cast<std::int64_t>(query_index) *
                               max_neighbors;
    for (int neighbor = 0; neighbor < max_neighbors; ++neighbor) {
      const std::int64_t target = neighbors[neighbor];
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

}  // namespace

struct Workspace::Impl {
  int device = -1;
  std::int64_t query_capacity = 0;
  std::int64_t database_capacity = 0;
  int dimension_capacity = 0;
  int neighbor_capacity = 0;

  float* bounds = nullptr;
  GridParameters* grid_parameters = nullptr;
  int* grid_counts = nullptr;
  int* grid_offsets = nullptr;
  int* point_cells = nullptr;
  int* point_cell_indices = nullptr;
  float* sorted_database = nullptr;
  int* sorted_database_indices = nullptr;
  float* neighbor_distances = nullptr;
  std::int64_t* neighbor_indices = nullptr;
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
                        int max_neighbors) {
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

  int current_device = -1;
  checkCuda(cudaGetDevice(&current_device), "cudaGetDevice");
  if (impl_->device == current_device &&
      impl_->query_capacity >= max_query_points &&
      impl_->database_capacity >= max_database_points &&
      impl_->dimension_capacity >= dimension &&
      impl_->neighbor_capacity >= max_neighbors) {
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

  try {
    allocateDevice(&impl_->bounds, 2 * kGridDimensions, "allocate bounds");
    allocateDevice(&impl_->grid_parameters, 1, "allocate grid parameters");
    allocateDevice(&impl_->grid_counts, kMaximumGridCells,
                   "allocate grid counts");
    allocateDevice(&impl_->grid_offsets, kMaximumGridCells,
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
    checkCuda(cub::DeviceScan::ExclusiveSum(
                  nullptr, grid_scan_bytes, impl_->grid_counts,
                  impl_->grid_offsets, kMaximumGridCells),
              "size grid prefix-sum workspace");
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
  if (output.capacity < required_capacity) {
    throw std::invalid_argument(
        "output.capacity is smaller than requiredEdgeCapacity");
  }
  if (required_capacity > 0 && output.edges == nullptr) {
    throw std::invalid_argument(
        "output.edges must not be null for non-empty output capacity");
  }
  if (options.undirected && !options.inputs_are_same) {
    throw std::invalid_argument(
        "undirected output requires inputs_are_same=true");
  }

  if (query.size == 0 || database.size == 0 || max_neighbors == 0) {
    checkCuda(cudaMemsetAsync(output.edge_count, 0, sizeof(std::int64_t),
                              stream),
              "clear edge count");
    return;
  }

  workspace.reserve(query.size, database.size, query.dimension,
                    max_neighbors);
  Workspace::Impl& memory = *workspace.impl_;
  const int query_count = static_cast<int>(query.size);
  const int database_count = static_cast<int>(database.size);
  const int grid_dimensions = std::min(query.dimension, kGridDimensions);

  initializeBounds<<<1, kGridDimensions, 0, stream>>>(memory.bounds);
  checkCuda(cudaPeekAtLastError(), "launch initializeBounds");
  computeBounds<<<blocksFor(database.size), kThreads, 0, stream>>>(
      database.data, database_count, database.dimension, grid_dimensions,
      memory.bounds);
  checkCuda(cudaPeekAtLastError(), "launch computeBounds");
  finalizeGrid<<<1, 1, 0, stream>>>(memory.bounds, grid_dimensions, radius,
                                    memory.grid_parameters);
  checkCuda(cudaPeekAtLastError(), "launch finalizeGrid");

  checkCuda(cudaMemsetAsync(memory.grid_counts, 0,
                            kMaximumGridCells * sizeof(int), stream),
            "clear grid counts");
  insertPoints<<<blocksFor(database.size), kThreads, 0, stream>>>(
      database.data, database_count, database.dimension,
      memory.grid_parameters, memory.grid_counts, memory.point_cells,
      memory.point_cell_indices);
  checkCuda(cudaPeekAtLastError(), "launch insertPoints");
  checkCuda(cub::DeviceScan::ExclusiveSum(
                memory.scan_temporary, memory.scan_temporary_bytes,
                memory.grid_counts, memory.grid_offsets, kMaximumGridCells,
                stream),
            "prefix sum grid counts");
  countingSort<<<blocksFor(database.size), kThreads, 0, stream>>>(
      database.data, database_count, database.dimension, memory.point_cells,
      memory.point_cell_indices, memory.grid_offsets, memory.sorted_database,
      memory.sorted_database_indices);
  checkCuda(cudaPeekAtLastError(), "launch countingSort");

  findNeighbors<<<blocksFor(query.size), kThreads, 0, stream>>>(
      query.data, query_count, memory.sorted_database,
      memory.sorted_database_indices, database_count, query.dimension,
      memory.grid_parameters, memory.grid_offsets, radius, max_neighbors,
      options.exclude_self, options.inputs_are_same,
      memory.neighbor_distances, memory.neighbor_indices);
  checkCuda(cudaPeekAtLastError(), "launch findNeighbors");
  checkCuda(cudaMemsetAsync(memory.edge_counts, 0,
                            (query.size + 1) * sizeof(std::int64_t), stream),
            "clear edge counts");
  countEdges<<<blocksFor(query.size), kThreads, 0, stream>>>(
      memory.neighbor_indices, query_count, max_neighbors, options.undirected,
      memory.edge_counts);
  checkCuda(cudaPeekAtLastError(), "launch countEdges");
  checkCuda(cub::DeviceScan::ExclusiveSum(
                memory.scan_temporary, memory.scan_temporary_bytes,
                memory.edge_counts, memory.edge_offsets, query_count + 1,
                stream),
            "prefix sum edge counts");
  writeEdges<<<blocksFor(query.size), kThreads, 0, stream>>>(
      memory.neighbor_indices, query_count, max_neighbors, options.undirected,
      memory.edge_offsets, output.edges);
  checkCuda(cudaPeekAtLastError(), "launch writeEdges");
  checkCuda(cudaMemcpyAsync(
                output.edge_count, memory.edge_offsets + query_count,
                sizeof(std::int64_t), cudaMemcpyDeviceToDevice, stream),
            "copy edge count");
}

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
    allocateDevice(&device_edges, static_cast<std::size_t>(capacity) * 2,
                   "allocate device edges");
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
                      max_neighbors);
    buildEdgesAsync(
        {device_query, query.size, query.dimension},
        {device_database, database.size, database.dimension},
        {device_edges, capacity, device_count}, radius, max_neighbors, options,
        workspace, stream);

    std::int64_t edge_count = 0;
    checkCuda(cudaMemcpyAsync(&edge_count, device_count, sizeof(edge_count),
                              cudaMemcpyDeviceToHost, stream),
              "copy edge count to host");
    checkCuda(cudaStreamSynchronize(stream), "synchronize edge count");
    std::vector<Edge> result(static_cast<std::size_t>(edge_count));
    if (edge_count > 0) {
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
