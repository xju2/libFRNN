#include <frnn/frnn.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct ArrayView {
  py::array array;
  frnn::PointView points;
};

ArrayView validateArray(const py::handle& object, const char* name) {
  if (!py::isinstance<py::array>(object)) {
    throw py::type_error(std::string(name) +
                         " must be a NumPy ndarray");
  }
  py::array array = py::reinterpret_borrow<py::array>(object);
  if (!array.dtype().is(py::dtype::of<float>())) {
    throw py::type_error(std::string(name) + " must have dtype numpy.float32");
  }
  if (array.ndim() != 2) {
    throw py::value_error(std::string(name) +
                          " must have shape [num_points, dimension]");
  }
  if ((array.flags() & py::array::c_style) == 0) {
    throw py::value_error(std::string(name) + " must be C-contiguous");
  }
  if (array.shape(0) > std::numeric_limits<int>::max()) {
    throw py::value_error(std::string(name) + " has too many points");
  }
  if (array.shape(1) <= 0 || array.shape(1) > 32) {
    throw py::value_error(std::string(name) +
                          " dimension must be in the range [1, 32]");
  }
  return {
      array,
      {static_cast<const float*>(array.data()),
       static_cast<std::int64_t>(array.shape(0)),
       static_cast<int>(array.shape(1))},
  };
}

py::array_t<std::int64_t> buildEdges(
    const py::object& query_object, const py::object& database_object,
    float radius, int max_neighbors, bool exclude_self,
    const py::object& directed_object) {
  ArrayView query = validateArray(query_object, "query");
  const bool same_input = database_object.is_none();
  std::optional<ArrayView> database;
  if (!same_input) {
    database.emplace(validateArray(database_object, "database"));
    if (database->points.dimension != query.points.dimension) {
      throw py::value_error(
          "query and database must have the same dimension");
    }
  }

  bool directed = !same_input;
  if (!directed_object.is_none()) {
    directed = py::cast<bool>(directed_object);
  }
  if (!same_input && !directed) {
    throw py::value_error(
        "directed=False is only supported when database is omitted");
  }

  frnn::BuildOptions options;
  options.exclude_self = exclude_self;
  options.undirected = !directed;
  options.inputs_are_same = same_input;
  std::vector<frnn::Edge> edges;
  {
    py::gil_scoped_release release;
    edges = frnn::buildEdges(
        query.points, same_input ? query.points : database->points, radius,
        max_neighbors, options);
  }

  py::array_t<std::int64_t> result(
      {static_cast<py::ssize_t>(2),
       static_cast<py::ssize_t>(edges.size())});
  auto output = result.mutable_unchecked<2>();
  for (py::ssize_t index = 0;
       index < static_cast<py::ssize_t>(edges.size()); ++index) {
    output(0, index) = edges[static_cast<std::size_t>(index)].source;
    output(1, index) = edges[static_cast<std::size_t>(index)].target;
  }
  return result;
}

}  // namespace

PYBIND11_MODULE(_frnn, module) {
  module.doc() =
      "NumPy bindings for the standalone libFRNN C++/CUDA library";
  module.attr("__version__") = "1.0.0";
  module.def(
      "build_edges", &buildEdges, py::arg("query"),
      py::arg("database") = py::none(), py::kw_only(), py::arg("radius"),
      py::arg("max_neighbors"), py::arg("exclude_self") = true,
      py::arg("directed") = py::none(),
      R"doc(
Build fixed-radius nearest-neighbor edges.

query and optional database must be C-contiguous NumPy float32 arrays with
shape [num_points, dimension]. When database is omitted, the default is an
undirected unique edge list with self-loops removed. With database supplied,
the default is directed query-to-database edges. Results have int64 dtype and
shape [2, num_edges].

Neighbors at exactly radius are included. At most max_neighbors candidates are
kept per query, ordered by squared distance and then target index. The call is
synchronous: NumPy host inputs are copied to CUDA memory and results are copied
back to host memory. No internal CUDA buffers are exposed.
)doc");
}
