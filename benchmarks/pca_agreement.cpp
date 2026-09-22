// Quantify how much the PCA-rotation front-end changes the *result* on the
// real D=12 embedding. Runs buildEdges twice in-process (rotation controlled
// by FRNN_PCA_ROTATE via setenv before each call is not possible since the env
// is read at reserve() time; instead run this binary twice and diff files).
//
// Usage: pca_agreement <embedding.csv> <N> <D> <radius> <K> <out.txt>
#include <frnn/frnn.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: %s <csv> <N> <D> <radius> <K> <out>\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const std::int64_t n = std::stoll(argv[2]);
  const int d = std::stoi(argv[3]);
  const float radius = std::stof(argv[4]);
  const int k = std::stoi(argv[5]);
  const std::string out = argv[6];

  std::vector<float> points;
  points.reserve(static_cast<std::size_t>(n) * d);
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
  std::string line;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) points.push_back(std::stof(tok));
  }
  if (points.size() != static_cast<std::size_t>(n) * d) {
    std::fprintf(stderr, "row count mismatch: got %zu want %lld\n",
                 points.size(), (long long)(n * d));
    return 2;
  }

  frnn::BuildOptions options;
  options.exclude_self = true;
  options.inputs_are_same = true;
  auto edges = frnn::buildEdges({points.data(), n, d},
                                {points.data(), n, d}, radius, k, options);
  // Canonicalize: sort by (source,target) so two runs are comparable.
  std::sort(edges.begin(), edges.end(), [](const frnn::Edge& a,
                                           const frnn::Edge& b) {
    return a.source < b.source || (a.source == b.source && a.target < b.target);
  });
  std::ofstream os(out);
  for (const auto& e : edges) os << e.source << ' ' << e.target << '\n';
  std::fprintf(stderr, "wrote %zu edges to %s\n", edges.size(), out.c_str());
  return 0;
}
