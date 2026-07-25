#include <frnn/frnn.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
  const char* name;
  std::int64_t query_count;
  std::int64_t database_count;
  int dimension;
  int max_neighbors;
  int target_neighbors;
  bool identical;
  frnn::SearchAlgorithm algorithm = frnn::SearchAlgorithm::automatic;
  const char* distribution = "uniform";
  bool non_default_stream = true;
};

struct Statistics {
  double minimum = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double mean = 0.0;
  double standard_deviation = 0.0;
};

struct Options {
  std::string output;
  std::string case_name;
  std::string embedding;
  frnn::SearchAlgorithm algorithm = frnn::SearchAlgorithm::automatic;
  int warmup = 5;
  int iterations = 30;
  bool full = false;
};

void checkCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--output" && index + 1 < argc) {
      options.output = argv[++index];
    } else if (argument == "--case" && index + 1 < argc) {
      options.case_name = argv[++index];
    } else if (argument == "--embedding" && index + 1 < argc) {
      options.embedding = argv[++index];
    } else if (argument == "--algorithm" && index + 1 < argc) {
      const std::string value = argv[++index];
      if (value == "auto") {
        options.algorithm = frnn::SearchAlgorithm::automatic;
      } else if (value == "grid") {
        options.algorithm = frnn::SearchAlgorithm::grid;
      } else if (value == "brute_force") {
        options.algorithm = frnn::SearchAlgorithm::brute_force;
      } else {
        throw std::invalid_argument(
            "--algorithm must be auto, grid, or brute_force");
      }
    } else if (argument == "--warmup" && index + 1 < argc) {
      options.warmup = std::stoi(argv[++index]);
    } else if (argument == "--iterations" && index + 1 < argc) {
      options.iterations = std::stoi(argv[++index]);
    } else if (argument == "--full") {
      options.full = true;
    } else if (argument == "--help") {
      std::cout
          << "Usage: frnn_benchmark [--output FILE] [--case NAME] "
             "[--algorithm auto|grid|brute_force] [--warmup N] "
             "[--iterations N] [--full] [--embedding FILE]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown or incomplete option: " + argument);
    }
  }
  if (options.warmup < 0 || options.iterations <= 0) {
    throw std::invalid_argument(
        "warmup must be non-negative and iterations must be positive");
  }
  return options;
}

std::vector<float> loadEmbedding(const std::string& path,
                                 std::int64_t expected_count,
                                 int dimension) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open embedding file: " + path);
  }
  std::vector<float> points;
  points.reserve(static_cast<std::size_t>(expected_count) * dimension);
  std::string line;
  while (std::getline(input, line)) {
    const char* cursor = line.c_str();
    char* end = nullptr;
    for (int axis = 0; axis < dimension; ++axis) {
      const float value = std::strtof(cursor, &end);
      if (end == cursor || !std::isfinite(value)) {
        throw std::runtime_error("invalid embedding coordinate");
      }
      points.push_back(value);
      cursor = end;
      if (axis + 1 < dimension) {
        if (*cursor != ',') {
          throw std::runtime_error("invalid embedding column count");
        }
        ++cursor;
      }
    }
    if (*cursor != '\0' && *cursor != '\r') {
      throw std::runtime_error("invalid trailing embedding data");
    }
  }
  if (points.size() !=
      static_cast<std::size_t>(expected_count) * dimension) {
    throw std::runtime_error("embedding row count does not match expectation");
  }
  return points;
}

double unitBallVolume(int dimension) {
  return std::pow(3.14159265358979323846, dimension * 0.5) /
         std::tgamma(dimension * 0.5 + 1.0);
}

float radiusForDensity(std::int64_t database_count, int dimension,
                       int target_neighbors) {
  if (target_neighbors <= 0) {
    return 1.0e-5F;
  }
  const double fraction =
      static_cast<double>(target_neighbors) /
      std::max<std::int64_t>(database_count, 1);
  return static_cast<float>(
      std::pow(fraction / unitBallVolume(dimension), 1.0 / dimension));
}

std::vector<float> makeUniform(std::int64_t count, int dimension,
                               std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> points(
      static_cast<std::size_t>(count) * dimension);
  for (float& value : points) {
    value = distribution(generator);
  }
  return points;
}

std::vector<float> makePoints(const Case& benchmark_case,
                              std::int64_t count, float radius,
                              std::uint32_t seed) {
  std::vector<float> points =
      makeUniform(count, benchmark_case.dimension, seed);
  const std::string distribution = benchmark_case.distribution;
  std::mt19937 generator(seed + 101U);
  if (distribution == "clustered" ||
      distribution == "separated_clusters") {
    std::normal_distribution<float> noise(
        0.0F, distribution == "clustered" ? 0.025F : 0.01F);
    constexpr float centers[4][3] = {
        {0.2F, 0.2F, 0.2F},
        {0.8F, 0.2F, 0.8F},
        {0.2F, 0.8F, 0.8F},
        {0.8F, 0.8F, 0.2F},
    };
    for (std::int64_t point = 0; point < count; ++point) {
      const int cluster =
          distribution == "clustered" ? 0 : static_cast<int>(point % 4);
      for (int axis = 0; axis < benchmark_case.dimension; ++axis) {
        const float center =
            centers[cluster][axis % 3] +
            (axis >= 3 ? 0.02F * static_cast<float>(axis % 5) : 0.0F);
        points[point * benchmark_case.dimension + axis] =
            std::clamp(center + noise(generator), 0.0F, 1.0F);
      }
    }
  } else if (distribution == "cell_boundary") {
    const float cell_size = std::max(radius * 0.5F, 1.0e-4F);
    for (std::int64_t point = 0; point < count; ++point) {
      for (int axis = 0; axis < benchmark_case.dimension; ++axis) {
        float& value = points[point * benchmark_case.dimension + axis];
        const float boundary = std::round(value / cell_size) * cell_size;
        value = std::nextafter(
            boundary, ((point + axis) & 1) == 0
                          ? std::numeric_limits<float>::infinity()
                          : -std::numeric_limits<float>::infinity());
      }
    }
  } else if (distribution == "duplicates") {
    for (std::int64_t point = 1; point < count; point += 5) {
      std::copy_n(points.begin(), benchmark_case.dimension,
                  points.begin() + point * benchmark_case.dimension);
    }
  } else if (distribution == "correlated") {
    for (std::int64_t point = 0; point < count; ++point) {
      const float base = points[point * benchmark_case.dimension];
      for (int axis = 1; axis < benchmark_case.dimension; ++axis) {
        points[point * benchmark_case.dimension + axis] =
            std::fmod(base + 0.003F * static_cast<float>(axis), 1.0F);
      }
    }
  } else if (distribution == "anisotropic") {
    for (std::int64_t point = 0; point < count; ++point) {
      for (int axis = 1; axis < benchmark_case.dimension; ++axis) {
        points[point * benchmark_case.dimension + axis] *=
            std::pow(0.35F, static_cast<float>(axis % 4));
      }
    }
  } else if (distribution == "degenerate") {
    for (std::int64_t point = 0; point < count; ++point) {
      for (int axis = 1; axis < benchmark_case.dimension; ++axis) {
        points[point * benchmark_case.dimension + axis] = 0.5F;
      }
    }
  } else if (distribution == "sorted") {
    const int dimension = benchmark_case.dimension;
    std::vector<std::int64_t> order(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::int64_t lhs, std::int64_t rhs) {
                for (int axis = 0; axis < dimension; ++axis) {
                  const float left = points[lhs * dimension + axis];
                  const float right = points[rhs * dimension + axis];
                  if (left != right) {
                    return left < right;
                  }
                }
                return lhs < rhs;
              });
    std::vector<float> sorted(points.size());
    for (std::int64_t destination = 0; destination < count; ++destination) {
      std::copy_n(points.begin() + order[destination] * dimension, dimension,
                  sorted.begin() + destination * dimension);
    }
    points = std::move(sorted);
  }
  return points;
}

Statistics summarize(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample");
  }
  std::sort(values.begin(), values.end());
  Statistics result;
  result.minimum = values.front();
  result.p50 = values[(values.size() - 1) / 2];
  result.p95 = values[static_cast<std::size_t>(
      std::ceil(0.95 * values.size()) - 1)];
  result.mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double sum_of_squares = 0.0;
  for (double value : values) {
    const double difference = value - result.mean;
    sum_of_squares += difference * difference;
  }
  result.standard_deviation =
      std::sqrt(sum_of_squares / static_cast<double>(values.size()));
  return result;
}

double elapsedMilliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

class DeviceFixture {
 public:
  DeviceFixture(const Case& benchmark_case, const std::vector<float>& query,
                const std::vector<float>& database)
      : benchmark_case_(benchmark_case),
        capacity_(frnn::requiredEdgeCapacity(benchmark_case.query_count,
                                             benchmark_case.max_neighbors)) {
    if (benchmark_case.non_default_stream) {
      checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "create benchmark stream");
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_query_),
                         query.size() * sizeof(float)),
              "allocate benchmark query");
    if (benchmark_case.identical) {
      device_database_ = device_query_;
    } else {
      checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_database_),
                           database.size() * sizeof(float)),
                "allocate benchmark database");
    }
    if (capacity_ > 0) {
      checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_edges_),
                           static_cast<std::size_t>(capacity_) * 2 *
                               sizeof(std::int64_t)),
                "allocate benchmark edges");
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_edge_count_),
                         sizeof(std::int64_t)),
              "allocate benchmark edge count");
    checkCuda(cudaMemcpyAsync(device_query_, query.data(),
                              query.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "copy benchmark query");
    if (!benchmark_case.identical) {
      checkCuda(cudaMemcpyAsync(device_database_, database.data(),
                                database.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream_),
                "copy benchmark database");
    }
    checkCuda(cudaStreamSynchronize(stream_), "finish benchmark input copy");
  }

  ~DeviceFixture() {
    if (device_edge_count_ != nullptr) {
      cudaFree(device_edge_count_);
    }
    if (device_edges_ != nullptr) {
      cudaFree(device_edges_);
    }
    if (device_database_ != device_query_ && device_database_ != nullptr) {
      cudaFree(device_database_);
    }
    if (device_query_ != nullptr) {
      cudaFree(device_query_);
    }
    if (benchmark_case_.non_default_stream) {
      cudaStreamDestroy(stream_);
    }
  }

  DeviceFixture(const DeviceFixture&) = delete;
  DeviceFixture& operator=(const DeviceFixture&) = delete;

  frnn::DevicePointView queryView() const {
    return {device_query_, benchmark_case_.query_count,
            benchmark_case_.dimension};
  }

  frnn::DevicePointView databaseView() const {
    return {device_database_, benchmark_case_.database_count,
            benchmark_case_.dimension};
  }

  frnn::DeviceEdgeBuffer outputView() const {
    return {device_edges_, capacity_, device_edge_count_};
  }

  cudaStream_t stream() const { return stream_; }

  std::int64_t readEdgeCount() const {
    std::int64_t count = 0;
    checkCuda(cudaMemcpyAsync(&count, device_edge_count_, sizeof(count),
                              cudaMemcpyDeviceToHost, stream_),
              "copy benchmark edge count");
    checkCuda(cudaStreamSynchronize(stream_), "read benchmark edge count");
    return count;
  }

  double copyOutputMilliseconds(std::int64_t edge_count) const {
    std::vector<frnn::Edge> host_edges(
        static_cast<std::size_t>(edge_count));
    const auto begin = Clock::now();
    if (edge_count > 0) {
      checkCuda(cudaMemcpyAsync(
                    host_edges.data(), device_edges_,
                    static_cast<std::size_t>(edge_count) * sizeof(frnn::Edge),
                    cudaMemcpyDeviceToHost, stream_),
                "copy benchmark edge output");
    }
    checkCuda(cudaStreamSynchronize(stream_), "finish edge output copy");
    return elapsedMilliseconds(begin, Clock::now());
  }

 private:
  const Case& benchmark_case_;
  std::int64_t capacity_ = 0;
  cudaStream_t stream_ = nullptr;
  float* device_query_ = nullptr;
  float* device_database_ = nullptr;
  std::int64_t* device_edges_ = nullptr;
  std::int64_t* device_edge_count_ = nullptr;
};

frnn::BuildOptions buildOptions(const Case& benchmark_case) {
  frnn::BuildOptions options;
  options.exclude_self = benchmark_case.identical;
  options.undirected = benchmark_case.identical;
  options.inputs_are_same = benchmark_case.identical;
  options.algorithm = benchmark_case.algorithm;
  return options;
}

const char* algorithmName(frnn::SearchAlgorithm algorithm) {
  switch (algorithm) {
    case frnn::SearchAlgorithm::automatic:
      return "auto";
    case frnn::SearchAlgorithm::grid:
      return "grid";
    case frnn::SearchAlgorithm::brute_force:
      return "brute_force";
  }
  return "unknown";
}

void invoke(const Case& benchmark_case, float radius,
            const DeviceFixture& fixture, frnn::Workspace& workspace) {
  frnn::buildEdgesAsync(
      fixture.queryView(), fixture.databaseView(), fixture.outputView(), radius,
      benchmark_case.max_neighbors, buildOptions(benchmark_case), workspace,
      fixture.stream());
}

std::vector<double> measureWarmGpu(const Case& benchmark_case, float radius,
                                   const DeviceFixture& fixture,
                                   frnn::Workspace& workspace, int warmup,
                                   int iterations) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    invoke(benchmark_case, radius, fixture, workspace);
  }
  checkCuda(cudaStreamSynchronize(fixture.stream()), "finish GPU warmup");

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  checkCuda(cudaEventCreate(&begin), "create start event");
  checkCuda(cudaEventCreate(&end), "create end event");
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    checkCuda(cudaEventRecord(begin, fixture.stream()), "record start event");
    invoke(benchmark_case, radius, fixture, workspace);
    checkCuda(cudaEventRecord(end, fixture.stream()), "record end event");
    checkCuda(cudaEventSynchronize(end), "synchronize end event");
    float milliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&milliseconds, begin, end),
              "calculate GPU elapsed time");
    samples.push_back(milliseconds);
  }
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return samples;
}

std::vector<double> measureWarmHost(const Case& benchmark_case, float radius,
                                    const DeviceFixture& fixture,
                                    frnn::Workspace& workspace,
                                    int iterations) {
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto begin = Clock::now();
    invoke(benchmark_case, radius, fixture, workspace);
    checkCuda(cudaStreamSynchronize(fixture.stream()),
              "synchronize warm device call");
    samples.push_back(elapsedMilliseconds(begin, Clock::now()));
  }
  return samples;
}

std::vector<double> measureColdHost(const Case& benchmark_case, float radius,
                                    const DeviceFixture& fixture,
                                    int iterations) {
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    frnn::Workspace workspace;
    const auto begin = Clock::now();
    invoke(benchmark_case, radius, fixture, workspace);
    checkCuda(cudaStreamSynchronize(fixture.stream()),
              "synchronize cold device call");
    samples.push_back(elapsedMilliseconds(begin, Clock::now()));
  }
  return samples;
}

std::vector<double> measureReserve(const Case& benchmark_case,
                                   int iterations) {
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    frnn::Workspace workspace;
    const auto begin = Clock::now();
    workspace.reserve(benchmark_case.query_count,
                      benchmark_case.database_count,
                      benchmark_case.dimension,
                      benchmark_case.max_neighbors,
                      benchmark_case.algorithm);
    samples.push_back(elapsedMilliseconds(begin, Clock::now()));
  }
  return samples;
}

std::vector<double> measureHostApi(const Case& benchmark_case, float radius,
                                   const std::vector<float>& query,
                                   const std::vector<float>& database,
                                   int iterations) {
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const auto begin = Clock::now();
    const auto result = frnn::buildEdges(
        {query.data(), benchmark_case.query_count, benchmark_case.dimension},
        {database.data(), benchmark_case.database_count,
         benchmark_case.dimension},
        radius, benchmark_case.max_neighbors, buildOptions(benchmark_case));
    static_cast<void>(result);
    samples.push_back(elapsedMilliseconds(begin, Clock::now()));
  }
  return samples;
}

void emitRow(std::ostream& output, const Case& benchmark_case, float radius,
             const char* phase, int iterations, const Statistics& statistics,
             const cudaDeviceProp& properties, int driver_version,
             int runtime_version) {
  output << FRNN_GIT_COMMIT << ',' << FRNN_BUILD_TYPE << ",\""
         << FRNN_CXX_COMPILER << "\",\"" << FRNN_CUDA_COMPILER << "\",\""
         << properties.name << "\"," << properties.major << '.'
         << properties.minor << ',' << driver_version << ',' << runtime_version
         << ',' << FRNN_CUDA_ARCHITECTURES << ',' << benchmark_case.name << ','
         << benchmark_case.distribution << ','
         << (benchmark_case.identical ? "identical" : "separate") << ','
         << algorithmName(benchmark_case.algorithm) << ','
         << (benchmark_case.non_default_stream ? "non_default" : "default")
         << ','
         << benchmark_case.query_count << ',' << benchmark_case.database_count
         << ',' << benchmark_case.dimension << ','
         << benchmark_case.max_neighbors << ','
         << benchmark_case.target_neighbors << ',' << std::setprecision(9)
         << radius << ',' << phase << ',' << iterations << ','
         << std::setprecision(6) << statistics.minimum << ','
         << statistics.p50 << ',' << statistics.p95 << ',' << statistics.mean
         << ',' << statistics.standard_deviation << '\n';
}

std::vector<Case> benchmarkCases(bool full) {
  std::vector<Case> cases = {
      {"small_sparse_d3_k4", 128, 128, 3, 4, 4, true},
      {"small_dense_d3_k32", 512, 512, 3, 32, 64, true},
      {"medium_sparse_d3_k16", 4096, 4096, 3, 16, 16, true},
      {"medium_sparse_d8_k16", 10000, 10000, 8, 16, 16, true},
      {"medium_separate_d4_k32", 10000, 10000, 4, 32, 64, false},
  };
  if (full) {
    cases.insert(cases.end(), {
        {"tiny_sparse_d16_k1", 128, 128, 16, 1, 1, true},
        {"small_sparse_d4_k8", 1000, 1000, 4, 8, 4, true},
        {"medium_dense_d3_k64", 10000, 10000, 3, 64, 256, true},
        {"large_sparse_d3_k32", 50000, 50000, 3, 32, 16, true},
        {"large_sparse_d3_k64", 50000, 50000, 3, 64, 16, true},
        {"large_sparse_d8_k32", 100000, 100000, 8, 32, 16, true},
        {"medium_realshape_d12_k64", 10000, 10000, 12, 64, 64, true},
        {"very_large_sparse_d3_k8", 500000, 500000, 3, 8, 4, true},
        {"medium_empty_d3_k4", 4096, 4096, 3, 4, 0, true},
        {"medium_clustered_d3_k16", 4096, 4096, 3, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "clustered"},
        {"medium_separated_d4_k16", 4096, 4096, 4, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "separated_clusters"},
        {"medium_boundaries_d3_k16", 4096, 4096, 3, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "cell_boundary"},
        {"medium_duplicates_d8_k16", 4096, 4096, 8, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "duplicates"},
        {"medium_correlated_d8_k16", 4096, 4096, 8, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "correlated"},
        {"medium_anisotropic_d8_k16", 4096, 4096, 8, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "anisotropic"},
        {"medium_degenerate_d4_k16", 4096, 4096, 4, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "degenerate"},
        {"medium_sorted_d3_k16", 4096, 4096, 3, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "sorted"},
        {"medium_default_stream_d3_k16", 4096, 4096, 3, 16, 16, true,
         frnn::SearchAlgorithm::automatic, "uniform", false},
    });
  }
  return cases;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    int device = 0;
    checkCuda(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device),
              "get CUDA device properties");
    int driver_version = 0;
    int runtime_version = 0;
    checkCuda(cudaDriverGetVersion(&driver_version), "get CUDA driver version");
    checkCuda(cudaRuntimeGetVersion(&runtime_version),
              "get CUDA runtime version");

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!options.output.empty()) {
      file.open(options.output);
      if (!file) {
        throw std::runtime_error("cannot open output file: " + options.output);
      }
      output = &file;
    }
    *output
        << "commit,build_type,cxx_compiler,cuda_compiler,gpu,compute_capability,"
           "driver_version,runtime_version,cuda_architectures,case,"
           "distribution,query_mode,algorithm,stream,"
           "query_count,database_count,dimension,max_neighbors,"
           "target_neighbors,radius,phase,iterations,min_ms,p50_ms,p95_ms,"
           "mean_ms,stddev_ms\n";

    std::vector<Case> cases =
        options.embedding.empty()
            ? benchmarkCases(options.full)
            : std::vector<Case>{{"real_embedding_d12_k1000",
                                 271663,
                                 271663,
                                 12,
                                 1000,
                                 0,
                                 true}};
    for (Case benchmark_case : cases) {
      if (!options.case_name.empty() &&
          options.case_name != benchmark_case.name) {
        continue;
      }
      benchmark_case.algorithm = options.algorithm;
      std::cerr << "Benchmarking " << benchmark_case.name << " ...\n";
      const float radius =
          options.embedding.empty()
              ? radiusForDensity(benchmark_case.database_count,
                                 benchmark_case.dimension,
                                 benchmark_case.target_neighbors)
              : 0.12F;
      std::vector<float> query =
          options.embedding.empty()
              ? makePoints(benchmark_case, benchmark_case.query_count, radius,
                           7)
              : loadEmbedding(options.embedding, benchmark_case.query_count,
                              benchmark_case.dimension);
      std::vector<float> database =
          benchmark_case.identical
              ? query
              : makePoints(benchmark_case, benchmark_case.database_count,
                           radius, 19);
      DeviceFixture fixture(benchmark_case, query, database);

      const int cold_iterations = std::min(options.iterations, 5);
      emitRow(*output, benchmark_case, radius, "workspace_reserve_host",
              cold_iterations,
              summarize(measureReserve(benchmark_case, cold_iterations)),
              properties, driver_version, runtime_version);
      emitRow(*output, benchmark_case, radius, "device_cold_host",
              cold_iterations,
              summarize(measureColdHost(benchmark_case, radius, fixture,
                                        cold_iterations)),
              properties, driver_version, runtime_version);

      frnn::Workspace workspace;
      workspace.reserve(benchmark_case.query_count,
                        benchmark_case.database_count,
                        benchmark_case.dimension,
                        benchmark_case.max_neighbors,
                        benchmark_case.algorithm);
      emitRow(*output, benchmark_case, radius, "device_warm_gpu",
              options.iterations,
              summarize(measureWarmGpu(
                  benchmark_case, radius, fixture, workspace, options.warmup,
                  options.iterations)),
              properties, driver_version, runtime_version);
      emitRow(*output, benchmark_case, radius, "device_warm_host",
              options.iterations,
              summarize(measureWarmHost(
                  benchmark_case, radius, fixture, workspace,
                  options.iterations)),
              properties, driver_version, runtime_version);

      invoke(benchmark_case, radius, fixture, workspace);
      const std::int64_t edge_count = fixture.readEdgeCount();
      std::vector<double> copy_samples;
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        copy_samples.push_back(fixture.copyOutputMilliseconds(edge_count));
      }
      emitRow(*output, benchmark_case, radius, "device_to_host",
              options.iterations, summarize(std::move(copy_samples)),
              properties, driver_version, runtime_version);

      if (options.embedding.empty()) {
        const int host_iterations = std::min(options.iterations, 5);
        emitRow(
            *output, benchmark_case, radius, "host_api_end_to_end",
            host_iterations,
            summarize(measureHostApi(benchmark_case, radius, query, database,
                                     host_iterations)),
            properties, driver_version, runtime_version);
      }
      output->flush();
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FRNN benchmark failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
