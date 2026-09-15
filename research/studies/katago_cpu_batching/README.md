Increasing analysis concurrency did not produce larger neural-network batches
in this CPU pilot. All six runs processed 72 NN rows in 38 batches, averaging
1.895 rows per batch, despite configured maxima of 8, 16 and 32.

| Analysis concurrency | Geometric mean queries/s | Relative to 8 |
| --- | ---: | ---: |
| 8 | 1.3534 | 1.000 |
| 16 | 1.3104 | 0.968 |
| 32 | 1.4112 | 1.043 |

The [completed result](run-002/result.json) covers 64 fixed real-19×19 training
prefixes per run, with mirrored case order 8,16,32,32,16,8. Each fresh engine
received eight warmup queries, then a cache clear before timed one-visit
queries. The actual NN counters confirm one evaluation per query, with no
extra evaluations or cache reuse. All legal masks match the stored fixture;
raw policy/value outputs agree with its teacher labels and across cases
within the registered absolute tolerance of 3e-5. The pilot took 342.66 seconds
on two auxiliary CPU cores and changed no production process.

The inspected KataGo source explains the observed behavior: the Eigen backend
selects `BatchPolicy::CpuLocal`, whose setup fixes NN batch size at two and
ignores the configured `nnMaxBatchSize`. `numEigenThreadsPerModel` supplies the
number of independent NN server threads. These are separate from analysis
concurrency. The source describes CPU parallelism as concurrent evaluations
on separate cores. The measured binary reports the Eigen backend and two NN
server threads; its build omits its Git revision, so the pinned source reading
is not an exact binary rebuild-provenance claim.

These two measurements per setting show small timing differences in a
fixed-query workload. They do not establish a benefit for live 16-visit games
on a complete worker allocation. Burst response latency includes time queued
behind earlier queries. Retain the current producer settings. A future test of
actual larger NN batches would need an explicitly supported backend or direct
benchmark path and its own numerical/throughput validation.

`plan-001.json` stopped at preflight before launching an engine: corpus identity
serialization omits a final newline, unlike snapshot canonical JSON. The
original script and failure are retained. `benchmark_v2.py` and `plan-002.json`
correct only that identity check and the output directory; all pilot settings
are unchanged. No failed or capped games were added to the corpus.
