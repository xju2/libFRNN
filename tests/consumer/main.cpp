#include <frnn/frnn.hpp>

#include <iostream>
#include <vector>

int main() {
  std::vector<float> points = {
      0.0F, 0.0F, 0.0F,
      0.25F, 0.0F, 0.0F,
  };
  const auto edges =
      frnn::buildEdges({points.data(), 2, 3}, 0.5F, 4);
  if (edges.size() != 1 || edges[0].source != 1 ||
      edges[0].target != 0) {
    return 1;
  }
  std::cout << edges[0].source << ' ' << edges[0].target << '\n';
  return 0;
}
