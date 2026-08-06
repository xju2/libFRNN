#include <Python.h>

#include <frnn/frnn.hpp>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {

void checkCuda(cudaError_t error, const char* operation) {
  TORCH_CHECK(error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

void validateTensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name,
              " must have dtype torch.float32");
  TORCH_CHECK(tensor.dim() == 2, name,
              " must have shape [num_points, dimension]");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.size(1) >= 1 && tensor.size(1) <= 32, name,
              " dimension must be in the range [1, 32]");
}

at::Tensor buildEdges(
    const at::Tensor& query,
    const std::optional<at::Tensor>& database_object,
    double radius,
    std::int64_t max_neighbors_value,
    bool exclude_self,
    const std::optional<bool>& directed_object) {
  validateTensor(query, "query");
  TORCH_CHECK(max_neighbors_value >= 0 &&
                  max_neighbors_value <= std::numeric_limits<int>::max(),
              "max_neighbors must be in the supported int range");
  const int max_neighbors = static_cast<int>(max_neighbors_value);
  const bool same_input = !database_object.has_value();
  const at::Tensor& database = same_input ? query : *database_object;
  validateTensor(database, "database");
  TORCH_CHECK(query.device() == database.device(),
              "query and database must be on the same CUDA device");
  TORCH_CHECK(query.size(1) == database.size(1),
              "query and database must have the same dimension");

  const bool directed = directed_object.value_or(!same_input);
  TORCH_CHECK(same_input || directed,
              "directed=False is only supported when database is omitted");

  c10::cuda::CUDAGuard device_guard(query.device());
  const c10::cuda::CUDAStream stream =
      at::cuda::getCurrentCUDAStream(query.get_device());
  const std::int64_t capacity =
      frnn::requiredEdgeCapacity(query.size(0), max_neighbors);
  const auto output_options = query.options().dtype(at::kLong);
  at::Tensor edge_count = at::empty({}, output_options);

  frnn::BuildOptions options;
  options.exclude_self = exclude_self;
  options.undirected = !directed;
  options.inputs_are_same = same_input;
  frnn::Workspace workspace;
  frnn::buildEdgesAsync(
      {query.data_ptr<float>(), query.size(0), static_cast<int>(query.size(1))},
      {database.data_ptr<float>(), database.size(0),
       static_cast<int>(database.size(1))},
      {nullptr, 0, edge_count.data_ptr<std::int64_t>()},
      static_cast<float>(radius), max_neighbors, options, workspace,
      stream.stream());

  std::int64_t count = 0;
  checkCuda(cudaMemcpyAsync(&count, edge_count.data_ptr<std::int64_t>(),
                            sizeof(count), cudaMemcpyDeviceToHost,
                            stream.stream()),
            "copy edge count");
  checkCuda(cudaStreamSynchronize(stream.stream()), "synchronize edge count");
  TORCH_CHECK(count >= 0 && count <= capacity,
              "libFRNN returned an invalid edge count");
  at::Tensor edges = at::empty({count, 2}, output_options);
  if (count > 0) {
    frnn::materializeEdgesAsync(edges.data_ptr<std::int64_t>(), count,
                                workspace, stream.stream());
    checkCuda(cudaStreamSynchronize(stream.stream()),
              "synchronize edge output");
  }
  return edges.transpose(0, 1);
}

}  // namespace

TORCH_LIBRARY(frnn_cuda, module) {
  module.def(
      "build_edges(Tensor query, Tensor? database, float radius, "
      "int max_neighbors, bool exclude_self, bool? directed) -> Tensor");
}

TORCH_LIBRARY_IMPL(frnn_cuda, CUDA, module) {
  module.impl("build_edges", &buildEdges);
}

PyMODINIT_FUNC PyInit__frnn_torch() {
  static PyModuleDef definition = {
      PyModuleDef_HEAD_INIT,
      "_frnn_torch",
      "PyTorch registrations for libFRNN",
      -1,
      nullptr,
  };
  return PyModule_Create(&definition);
}
