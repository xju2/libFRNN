#pragma once

#include <cstdint>
#include <functional>

namespace frnn {

// Fast exact edge construction for the identical-set case.
//
// Builds a kd-ordering of the points on the GPU (32-point leaves), then each
// warp processes one leaf of 32 queries.  Candidates are pruned exactly by
// leaf bounding-box and point-to-query-box lower bounds (computed with the
// same monotone float operations as the final distance), and surviving pairs
// evaluate the squared distance with exactly the arithmetic of the grid and
// brute-force kernels.  Because max_neighbors truncation would require
// ordered selection, the function returns false (and produces nothing) when
// any query has more than max_neighbors in-radius candidates; callers must
// then use buildEdges().  It also returns false for unsupported inputs.
//
// On success, *edges receives a buffer allocated with std::aligned_alloc
// (release it with std::free) holding 2 * (*edge_count) int64 values laid out
// as [sources..., targets...].  Edge order is unspecified.
bool buildEdgesFast(const float* points, std::int64_t size, int dimension,
                    float radius, int max_neighbors, bool exclude_self,
                    bool undirected, std::int64_t** edges,
                    std::int64_t* edge_count);

}  // namespace frnn
