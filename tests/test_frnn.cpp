#include <frnn/frnn.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
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

std::vector<frnn::Edge> bruteForce(
    const std::vector<float>& query, std::int64_t query_count,
    const std::vector<float>& database, std::int64_t database_count,
    int dimension, float radius, int max_neighbors,
    frnn::BuildOptions options) {
  std::vector<frnn::Edge> result;
  const float radius_squared = radius * radius;
  for (std::int64_t source = 0; source < query_count; ++source) {
    std::vector<std::pair<float, std::int64_t>> candidates;
    for (std::int64_t target = 0; target < database_count; ++target) {
      if (options.exclude_self && options.inputs_are_same &&
          source == target) {
        continue;
      }
      float distance = 0.0F;
      for (int axis = 0; axis < dimension; ++axis) {
        const float difference =
            query[source * dimension + axis] -
            database[target * dimension + axis];
        distance += difference * difference;
      }
      if (distance <= radius_squared) {
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
      if (!options.undirected || source > candidate.second) {
        result.push_back({source, candidate.second});
      }
    }
  }
  return result;
}

std::vector<frnn::Edge> bruteForce(const std::vector<float>& points,
                                   int dimension, float radius,
                                   int max_neighbors) {
  frnn::BuildOptions options;
  options.exclude_self = true;
  options.undirected = true;
  options.inputs_are_same = true;
  const std::int64_t count = static_cast<std::int64_t>(
      points.size() / static_cast<std::size_t>(dimension));
  return bruteForce(points, count, points, count, dimension, radius,
                    max_neighbors, options);
}

void requireAgreement(const std::vector<float>& query,
                      std::int64_t query_count,
                      const std::vector<float>& database,
                      std::int64_t database_count, int dimension,
                      float radius, int max_neighbors,
                      frnn::BuildOptions options,
                      const std::string& context) {
  const auto expected =
      bruteForce(query, query_count, database, database_count, dimension,
                 radius, max_neighbors, options);
  const auto actual = frnn::buildEdges(
      {query.data(), query_count, dimension},
      {database.data(), database_count, dimension}, radius, max_neighbors,
      options);
  if (actual != expected) {
    std::ostringstream message;
    message << context << " disagrees with brute-force oracle: expected "
            << expected.size() << " edges, got " << actual.size();
    throw std::runtime_error(message.str());
  }
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

  const float inside = std::nextafter(1.0F, 0.0F);
  const float outside =
      std::nextafter(1.0F, std::numeric_limits<float>::infinity());
  std::vector<float> nextafter_points = {0.0F, 1.0F, inside, outside};
  frnn::BuildOptions options;
  options.inputs_are_same = true;
  for (frnn::SearchAlgorithm algorithm :
       {frnn::SearchAlgorithm::grid,
        frnn::SearchAlgorithm::brute_force,
        frnn::SearchAlgorithm::automatic}) {
    options.algorithm = algorithm;
    requireAgreement(nextafter_points, 4, nextafter_points, 4, 1, 1.0F, 4,
                     options, "nextafter radius boundary");
  }
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

  point[0] = std::numeric_limits<float>::infinity();
  rejected = false;
  try {
    static_cast<void>(frnn::buildEdges({point.data(), 1, 3}, 1.0F, 4));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "non-finite host coordinates must be rejected");
}

void testAllDispatchPaths() {
  std::vector<float> points = {
      0.0F, 0.0F, 0.0F, 0.25F, 0.0F, 0.0F,
      0.5F, 0.0F, 0.0F, 0.75F, 0.0F, 0.0F,
  };
  frnn::BuildOptions options;
  options.exclude_self = true;
  options.undirected = true;
  options.inputs_are_same = true;
  for (frnn::SearchAlgorithm algorithm :
       {frnn::SearchAlgorithm::grid,
        frnn::SearchAlgorithm::brute_force,
        frnn::SearchAlgorithm::automatic}) {
    options.algorithm = algorithm;
    requireAgreement(points, 4, points, 4, 3, 0.5F, 3, options,
                     "explicit dispatch path");
  }
}

void testRandomizedDifferential() {
  constexpr int dimensions[] = {1, 2, 3, 4, 8, 16, 32};
  constexpr int neighbor_counts[] = {0, 1, 4, 8, 16, 32, 64};
  constexpr float radii[] = {1.0e-4F, 0.03F, 0.25F, 1.0F, 8.0F};
  std::mt19937 generator(0x5eed1234U);
  std::uniform_real_distribution<float> coordinate(-2.0F, 2.0F);

  for (int case_index = 0; case_index < 2500; ++case_index) {
    const int dimension =
        dimensions[generator() % (sizeof(dimensions) / sizeof(*dimensions))];
    const bool identical = (generator() & 1U) != 0;
    const std::int64_t query_count = 1 + generator() % 18;
    const std::int64_t database_count =
        identical ? query_count : 1 + generator() % 18;
    const int max_neighbors =
        neighbor_counts[generator() %
                        (sizeof(neighbor_counts) /
                         sizeof(*neighbor_counts))];
    const float radius =
        radii[generator() % (sizeof(radii) / sizeof(*radii))];

    std::vector<float> database(
        static_cast<std::size_t>(database_count) * dimension);
    for (float& value : database) {
      value = coordinate(generator);
    }
    const int distribution = case_index % 6;
    if (distribution == 1) {
      for (float& value : database) {
        value *= 0.01F;
      }
    } else if (distribution == 2 && database_count > 1) {
      for (std::int64_t point_index = 1;
           point_index < database_count; point_index += 3) {
        std::copy_n(database.begin(), dimension,
                    database.begin() + point_index * dimension);
      }
    } else if (distribution == 3) {
      for (std::int64_t point_index = 0; point_index < database_count;
           ++point_index) {
        for (int axis = 1; axis < dimension; ++axis) {
          database[point_index * dimension + axis] = 0.0F;
        }
      }
    } else if (distribution == 4) {
      for (std::int64_t point_index = 0; point_index < database_count;
           ++point_index) {
        const float base = database[point_index * dimension];
        for (int axis = 1; axis < dimension; ++axis) {
          database[point_index * dimension + axis] =
              base + static_cast<float>(axis) * 0.001F;
        }
      }
    } else if (distribution == 5) {
      std::sort(database.begin(), database.end());
    }

    std::vector<float> query;
    if (identical) {
      query = database;
    } else {
      query.resize(static_cast<std::size_t>(query_count) * dimension);
      for (float& value : query) {
        value = coordinate(generator);
      }
    }

    frnn::BuildOptions options;
    options.exclude_self = identical && case_index % 3 != 0;
    options.undirected = identical && case_index % 4 == 0;
    options.inputs_are_same = identical;
    for (frnn::SearchAlgorithm algorithm :
         {frnn::SearchAlgorithm::grid,
          frnn::SearchAlgorithm::brute_force,
          frnn::SearchAlgorithm::automatic}) {
      options.algorithm = algorithm;
      std::ostringstream context;
      context << "random case " << case_index << ", algorithm "
              << static_cast<int>(algorithm) << ", dimension " << dimension
              << ", K " << max_neighbors << ", radius " << radius;
      requireAgreement(query, query_count, database, database_count,
                       dimension, radius, max_neighbors, options,
                       context.str());
    }
  }
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

  frnn::buildEdgesAsync(
      {device_points, point_count, dimension},
      {device_points, point_count, dimension},
      {nullptr, 0, device_count}, 0.5F, max_neighbors, options, workspace,
      stream);
  edge_count = -1;
  cudaRequire(cudaMemcpyAsync(&edge_count, device_count, sizeof(edge_count),
                              cudaMemcpyDeviceToHost, stream),
              "copy count-only result");
  cudaRequire(cudaStreamSynchronize(stream), "synchronize count-only result");
  require(edge_count ==
              static_cast<std::int64_t>(
                  bruteForce(points, dimension, 0.5F, max_neighbors).size()),
          "count-only device API returned the wrong edge count");

  for (int iteration = 0; iteration < 6; ++iteration) {
    const std::int64_t active_count = iteration % 2 == 0 ? 2 : point_count;
    frnn::buildEdgesAsync(
        {device_points, active_count, dimension},
        {device_points, active_count, dimension},
        {device_edges, capacity, device_count}, 0.5F, max_neighbors, options,
        workspace, stream);
    edge_count = 0;
    cudaRequire(cudaMemcpyAsync(&edge_count, device_count, sizeof(edge_count),
                                cudaMemcpyDeviceToHost, stream),
                "copy alternating-size edge count");
    cudaRequire(cudaMemcpyAsync(actual.data(), device_edges,
                                capacity * sizeof(frnn::Edge),
                                cudaMemcpyDeviceToHost, stream),
                "copy alternating-size edges");
    cudaRequire(cudaStreamSynchronize(stream),
                "synchronize alternating-size call");
    const std::vector<float> active_points(
        points.begin(), points.begin() + active_count * dimension);
    const auto expected =
        bruteForce(active_points, dimension, 0.5F, max_neighbors);
    actual.resize(static_cast<std::size_t>(edge_count));
    require(actual == expected,
            "reused workspace disagrees after alternating input sizes");
    actual.resize(static_cast<std::size_t>(capacity));
  }

  frnn::Workspace default_stream_workspace;
  default_stream_workspace.reserve(point_count, point_count, dimension,
                                   max_neighbors);
  frnn::buildEdgesAsync(
      {device_points, point_count, dimension},
      {device_points, point_count, dimension},
      {device_edges, capacity, device_count}, 0.5F, max_neighbors, options,
      default_stream_workspace, nullptr);
  edge_count = 0;
  cudaRequire(cudaMemcpy(&edge_count, device_count, sizeof(edge_count),
                         cudaMemcpyDeviceToHost),
              "copy default-stream edge count");
  actual.resize(static_cast<std::size_t>(capacity));
  cudaRequire(cudaMemcpy(actual.data(), device_edges,
                         capacity * sizeof(frnn::Edge),
                         cudaMemcpyDeviceToHost),
              "copy default-stream edges");
  actual.resize(static_cast<std::size_t>(edge_count));
  require(actual == bruteForce(points, dimension, 0.5F, max_neighbors),
          "default-stream result disagrees with reference");

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
    testAllDispatchPaths();
    testRandomizedDifferential();
    testCompatibilityLayout();
    testCallerStreamDeviceApi();
    std::cout << "All FRNN core tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FRNN core test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
