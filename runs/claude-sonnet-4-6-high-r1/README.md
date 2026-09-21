# Claude Sonnet 4.6 high r1 results

This branch preserves run `claude-sonnet-4-6-high-r1` from its exact input
commit, `5ce9f21e2a2f12edfc5734fdd31973098a2f8d6d`. The branch source is the
final candidate produced by the run.

## Outcome

The candidate is exact on the supplied real workload:

| Item | Result |
|---|---:|
| Embedding | 271,663 points x 12 dimensions |
| Radius / K | 0.12 / 1000 |
| Reference edges | 9,279,672 |
| Candidate edges | 9,279,672 |
| Missing / extra / duplicates | 0 / 0 / 0 |
| Precision / recall | 1.0 / 1.0 |

The original run measured 207.299 ms p50 and 208.730 ms p95 over 30 warm
device iterations. Two independent reconstructions measured 207.161 ms and
207.423 ms p50; the archived rerun in `benchmark.csv` measured 208.677 ms
p95. The result is therefore reproducible on the recorded RTX 2070 SUPER.

This is a workload-specific result, not a generally exact replacement for
`main`. For `K >= 256`, the candidate removes heap maintenance and assumes
that every query has fewer than K in-radius neighbors. That assumption holds
for the supplied workload, but a denser input could retain the first K
candidates rather than the nearest K.

## Optimization history

| Variant | Warm device p50 | Outcome |
|---|---:|---|
| Native SM75 baseline | 307.861 ms | Baseline |
| Warp-cooperative search | 537.598 ms | Rejected |
| Append then post-sort | 292.526 ms | Retained |
| Grouped D12 distance loads | 252.996 ms | Retained |
| No distance early exit | 268.761 ms | Rejected |
| Heap-free K>=256 dispatch | 250.198 ms | Retained for target workload |
| Deferred original-index lookup | 207.299 ms | Final candidate |

The largest final gain came from loading `sorted_database_indices` only after
a candidate passed the radius test. Roughly 96% of candidates were rejected,
so the change avoided most indirect index loads.

## Archived files

- `goal.txt`: exact prompt.
- `manifest.json`: environment, inputs, hashes, and original failed status.
- `events.jsonl`: run lifecycle.
- `transcript.log`: complete Claude Code stream.
- `result.patch`: original captured diff.
- `validation.txt`: post-run CTest and full reference comparison.
- `benchmark.csv`: post-run 10-warmup, 30-iteration benchmark.

The original `result.patch` ended one unchanged context line early and is not
directly applicable. It is retained byte-for-byte because its SHA-256 is
recorded in `manifest.json`. The checked-out branch source is the reconstructed
and validated candidate.

## Reproduction

Place the canonical `embedding_data.csv` and `edge_list.csv` in `data/`,
then run:

```bash
cmake -S . -B build-r1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DFRNN_BUILD_TESTS=ON \
  -DFRNN_BUILD_BENCHMARKS=ON
cmake --build build-r1 --parallel
ctest --test-dir build-r1 --output-on-failure

CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75" \
  python3 -m pip install --no-build-isolation .
python3 tests/compare_reference_edges.py --minimum-agreement 1.0

./build-r1/frnn_benchmark \
  --embedding data/embedding_data.csv \
  --warmup 10 --iterations 30
```
