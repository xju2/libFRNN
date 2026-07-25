#include <frnn/frnn.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void cudaRequire(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

std::vector<frnn::Edge> bruteForce(const std::vector<float>& points,
                                   int dimension, float radius,
                                   int max_neighbors) {
  const std::int64_t count =
      static_cast<std::int64_t>(points.size() / dimension);
  std::vector<frnn::Edge> result;
  for (std::int64_t source = 0; source < count; ++source) {
    std::vector<std::pair<float, std::int64_t>> candidates;
    for (std::int64_t target = 0; target < count; ++target) {
      if (source == target) {
        continue;
      }
      float distance = 0.0F;
      for (int axis = 0; axis < dimension; ++axis) {
        const float difference =
            points[source * dimension + axis] -
            points[target * dimension + axis];
        distance += difference * difference;
      }
      if (distance <= radius * radius) {
        candidates.emplace_back(distance, target);
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first ||
                       (lhs.first == rhs.first && lhs.second < rhs.second);
              });
    if (candidates.size() > static_cast<std::size_t>(max_neighbors)) {
      candidates.resize(max_neighbors);
    }
    for (const auto& candidate : candidates) {
      if (source > candidate.second) {
        result.push_back({source, candidate.second});
      }
    }
  }
  return result;
}

void testReferenceAgreement() {
  constexpr int dimension = 3;
  std::vector<float> points = {
      0.0F, 0.0F, 0.0F, 0.2F, 0.0F, 0.0F, 0.0F, 0.3F, 0.0F,
      0.0F, 0.0F, 0.4F, 0.9F, 0.9F, 0.9F, 0.2F, 0.2F, 0.0F,
  };
  const auto actual =
      frnn::buildEdges({points.data(), 6, dimension}, 0.42F, 4);
  const auto expected = bruteForce(points, dimension, 0.42F, 4);
  require(actual == expected, "core result disagrees with brute-force result");
}

void testBoundaryTiesAndDuplicates() {
  std::vector<float> boundary = {
      0.0F, 0.0F, 0.0F,
      1.0F, 0.0F, 0.0F,
  };
  const auto boundary_edges =
      frnn::buildEdges({boundary.data(), 2, 3}, 1.0F, 1);
  require(boundary_edges == std::vector<frnn::Edge>{{1, 0}},
          "radius boundary must be inclusive");

  std::vector<float> duplicate = {
      0.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 0.0F,
      0.0F, 0.0F, 0.0F,
  };
  const auto duplicate_edges =
      frnn::buildEdges({duplicate.data(), 3, 3}, 0.1F, 1);
  require(duplicate_edges == std::vector<frnn::Edge>{{1, 0}, {2, 0}},
          "equal-distance ties must prefer the lower target index");
}

void testEmptyAndInvalidInput() {
  const auto empty = frnn::buildEdges({nullptr, 0, 3}, 1.0F, 4);
  require(empty.empty(), "empty input must return no edges");

  std::vector<float> point = {0.0F, 0.0F, 0.0F};
  require(frnn::buildEdges({point.data(), 1, 3}, 1.0F, 4).empty(),
          "a single point must return no edges");

  bool rejected = false;
  try {
    static_cast<void>(
        frnn::buildEdges({point.data(), 1, 3}, NAN, 4));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "non-finite radius must be rejected");
}

void testCompatibilityLayout() {
  std::vector<float> points = {
      0.0F, 0.0F, 0.0F,
      0.5F, 0.0F, 0.0F,
      2.0F, 0.0F, 0.0F,
  };
  std::vector<std::int64_t> flattened;
  frnn::buildEdges(points, flattened, 3, 3, 0.5F, 3);
  require(flattened == std::vector<std::int64_t>{1, 0},
          "compatibility output must use flattened [2, E] layout");
}

void testCallerStreamDeviceApi() {
  std::vector<float> points = {
      0.0F, 0.0F, 0.0F,
      0.25F, 0.0F, 0.0F,
      0.5F, 0.0F, 0.0F,
  };
  constexpr std::int64_t point_count = 3;
  constexpr int dimension = 3;
  constexpr int max_neighbors = 2;
  const std::int64_t capacity =
      frnn::requiredEdgeCapacity(point_count, max_neighbors);

  cudaStream_t stream = nullptr;
  float* device_points = nullptr;
  std::int64_t* device_edges = nullptr;
  std::int64_t* device_count = nullptr;
  cudaRequire(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create test stream");
  cudaRequire(cudaMalloc(reinterpret_cast<void**>(&device_points),
                         points.size() * sizeof(float)),
              "allocate test points");
  cudaRequire(cudaMalloc(reinterpret_cast<void**>(&device_edges),
                         capacity * 2 * sizeof(std::int64_t)),
              "allocate test edges");
  cudaRequire(cudaMalloc(reinterpret_cast<void**>(&device_count),
                         sizeof(std::int64_t)),
              "allocate test count");
  cudaRequire(cudaMemcpyAsync(device_points, points.data(),
                              points.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
              "copy test points");

  frnn::Workspace workspace;
  workspace.reserve(point_count, point_count, dimension, max_neighbors);
  frnn::BuildOptions options;
  options.exclude_self = true;
  options.undirected = true;
  options.inputs_are_same = true;
  frnn::buildEdgesAsync(
      {device_points, point_count, dimension},
      {device_points, point_count, dimension},
      {device_edges, capacity, device_count}, 0.5F, max_neighbors, options,
      workspace, stream);

  std::int64_t edge_count = 0;
  std::vector<frnn::Edge> actual(static_cast<std::size_t>(capacity));
  cudaRequire(cudaMemcpyAsync(&edge_count, device_count, sizeof(edge_count),
                              cudaMemcpyDeviceToHost, stream),
              "copy test count");
  cudaRequire(cudaMemcpyAsync(actual.data(), device_edges,
                              capacity * sizeof(frnn::Edge),
                              cudaMemcpyDeviceToHost, stream),
              "copy test edges");
  cudaRequire(cudaStreamSynchronize(stream), "synchronize test stream");
  actual.resize(static_cast<std::size_t>(edge_count));
  require(actual == bruteForce(points, dimension, 0.5F, max_neighbors),
          "device API result disagrees with reference");

  cudaFree(device_count);
  cudaFree(device_edges);
  cudaFree(device_points);
  cudaStreamDestroy(stream);
}

}  // namespace

int main() {
  try {
    testReferenceAgreement();
    testBoundaryTiesAndDuplicates();
    testEmptyAndInvalidInput();
    testCompatibilityLayout();
    testCallerStreamDeviceApi();
    std::cout << "All FRNN core tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FRNN core test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
