# Initial optimization summary

This document summarizes the first active turn of Codex session
`019f9a08-691f-7282-8a24-3b53b084721f`, recorded on 2026-07-25 from
16:18:07 to 16:53:22 UTC. It describes the historical change from baseline
commit `a35c713` through `5ce9f21`, not necessarily the repository's current
implementation.

## Session configuration

| Item | Recorded value |
|---|---|
| Codex CLI | 0.145.0 (`codex-tui`, CLI source) |
| Model provider / model | OpenAI / `gpt-5.6-sol` |
| Reasoning / summary | `high` / `auto` |
| Context window | 258,400 tokens |
| Composition hash | `3000` |
| Personality / collaboration | `pragmatic` / `default` |
| Multi-agent / realtime | v2 / disabled |
| Goal | Optimize standalone libFRNN for low latency with exact brute-force-equivalent results, following `docs/optimize_compute.md` |
| Initial repository state | `improve` at `a35c713227500614d7b102da7966454b210a8fbb` |
| Approval / sandbox | `never` / `danger-full-access` |
| Permission profile | Disabled; unrestricted filesystem and network |
| Tools invoked | `apply_patch`, `exec`, `update_plan`, `write_stdin` |
| Turn start / stop | 2026-07-25 16:18:07.162 / 16:53:22.057 UTC |
| Active wall time | 35 minutes 14.896 seconds |

The complete available-tool schemas and authoritative HTTP request count were
not retained. The log contains 219 token-accounting snapshots, whose final
cumulative value for this turn was 28,739,060 input tokens (28,270,592
cached), 88,764 output tokens (31,645 reasoning), and 28,827,824 total tokens.
Reasoning tokens are included in output tokens, and cached tokens are included
in input tokens.

All libFRNN sessions in this period used the same 17,730-character base system
prompt, beginning with "You are Codex, an agent based on GPT-5." Its recorded
SHA-256 fingerprint was
`35d8b5d513fff3b55344d5f9f3169305cc276aee053f5be140a77708e0926e7c`.
Developer instructions supplied memory, repository, agent, skill, permission,
and collaboration behavior on top of that base prompt.

## Build and machine configuration

| Item | Recorded value |
|---|---|
| libFRNN project version | 1.0.0 |
| Build | Release |
| GPU | NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5, 8 GiB |
| NVIDIA driver | 580.173.02 |
| CUDA Toolkit | 12.0.140 |
| CMake CUDA architecture | 52, the environment's detected/default value |
| CMake | 3.28.3 |
| C++ compiler | GNU 13.3.0 |
| CPU | AMD Ryzen 5 3600 6-Core Processor |
| Benchmark stream | Non-blocking caller stream unless named otherwise |
| Synthetic sampling | 10 warmup and 50 measured iterations |
| Real-data sampling | 10 warmup and 30 measured iterations |

GPU-event timings were recorded on the caller's stream; host timings used
`std::chrono::steady_clock` and synchronized before stopping. Total GPU time
for the Codex turn was not recorded and cannot be reconstructed from the
benchmark timings. Nsight Systems 2025.5.2 could not generate reports because
its QDSTRM importer was missing, while Nsight Compute hardware counters were
blocked by `ERR_NVGPUCTRPERM`.

## Commits

| Commit | Change |
|---|---|
| `237bb2f` | Optimize exact FRNN search and add benchmarks |
| `2a5da29` | Expand asynchronous workspace validation |
| `ceb4969` | Document final exact FRNN performance |
| `5ce9f21` | Add benchmark-only pipeline stage timing |

Together these commits changed 23 files with 2,887 insertions and 183
deletions. Most additions were benchmark data and documentation; the core
implementation change in `src/frnn.cu` was 607 insertions and 162 deletions.

## Implementation changes

- Added an exact CUDA brute-force path for small workloads and an overflow-safe
  automatic dispatch rule between brute-force and grid search.
- Added early termination of squared-distance accumulation once the partial
  sum exceeded the radius, plus specialized kernels for dimensions 1, 2, 3,
  4, 8, 12, and 16.
- Processed identical query/reference sets in spatial order for better cache
  locality while preserving original output indices and deterministic
  `(distance, original index)` ordering.
- Replaced 64-bit internal neighbor indices with 32-bit indices and replaced
  initialization of every neighbor slot with a single lazy `-1` sentinel.
- Introduced dimension-sensitive dense-grid limits: 131,072 cells for
  dimensions 1-4 and 2,097,152 cells for dimensions 5-32.
- Added count-only device execution. The synchronous host API first obtained
  the exact edge count, allocated only the required output, and then wrote
  edges without repeating neighbor search.
- Added a deterministic hybrid top-K implementation: sorted insertion for
  short rows, switching to a max heap after 24 accepted candidates when
  `K >= 64`.
- Added benchmark-only CUDA event instrumentation for grid construction,
  neighbor search, edge counting, scans, edge writing, and count copying.

All retained search paths preserved inclusive-radius matching, complete
dimensional distance evaluation, deterministic tie-breaking, and exact edge
post-processing.

## Benchmark and validation additions

The turn added `benchmarks/frnn_benchmark.cu` and machine-readable CSV results
covering cold and warm calls, host and device timing, workspace allocation,
output copies, multiple dimensions and K values, identical and separate query
sets, different point distributions, and default and non-default streams.

The correctness suite was expanded to 2,500 randomized cases evaluated using
forced grid search, forced brute force, and automatic dispatch: 7,500 path
comparisons per run. It also covered exact and adjacent `nextafter` radius
boundaries, ties, duplicates, empty and dense neighborhoods, reused
workspaces, alternating sizes, count-only calls, and caller-provided CUDA
streams.

The supplied real workload of 271,663 twelve-dimensional points at radius
0.12 and K=1000 produced exactly 9,279,672 directed edges: zero missing and
zero extra relative to `data/edge_list.csv`.

## Measured results

Representative warm-GPU results reported during the turn were:

| Workload | Baseline p50 | Optimized p50 | Speedup |
|---|---:|---:|---:|
| 128 points, D=3, K=4 | 0.168 ms | 0.058 ms | 2.88x |
| 512 points, D=3, K=32 | 1.031 ms | 0.776 ms | 1.33x |
| 4,096 points, D=3, K=16 | 0.454 ms | 0.235 ms | 1.93x |
| 10,000 points, D=8, K=16 | 8.240 ms | 3.228 ms | 2.55x |
| 10,000 separate queries, D=4, K=32 | 2.733 ms | 2.751 ms | 0.99x |

The geometric-mean p50 speedup across these cases was 1.80x. Count-first host
allocation reduced temporary device edge storage for the real workload from
about 4.35 GB of conservative capacity to 148 MB of exact output.

## Reproducing the historical review

```bash
git log --oneline --reverse a35c713..5ce9f21
git diff --stat a35c713..5ce9f21
git diff a35c713..5ce9f21
git show 5ce9f21:docs/optimization_log.md
git show 5ce9f21:docs/performance_report.md
```

The final command in the turn also generated
`benchmarks/results/final_stages_5ce9f21.csv`; that artifact was committed
later in `ccd2c23`.
