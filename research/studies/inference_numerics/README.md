# Fixed-input inference transport and numerics

The [registered probe](spec.json) and its immutable analysis (external or omitted experiment artifact)
evaluate the same trained Gumbel CNN on 8,192 real replay positions across the
four TPU hosts. Each host uses batch 128 repeated over 16 branches, or one batch
2048. Three prescribed timing orders include input/output transfers and host
assembly, with 50 iterations per mode after warmup.

| Mode | Median speed relative to serial | Prediction agreement |
|---|---:|---|
| Serial small batches | 1.00× | Reference |
| Queued small batches | 1.26× | Every tested output exact |
| One wide batch | 6.26× | 35/8,192 legal top-action changes |

Each ratio uses the slowest host in the corresponding repetition. Individual
wide ratios were 6.56×, 6.26× and 6.23×. These short, fixed-workload timings are
diagnostics rather than confidence intervals across training workloads.

Wide execution had mean legal policy KL 0.00003125 and mean total variation
0.002428; maxima were 0.001249 and 0.023972. Mean absolute value error was
0.0003936, maximum 0.009133. Raw observations, all three output sets, sampled
indices, source/checkpoint hashes and compiled HLO are retained per host.
The analyzer independently reproduced the observations from the original
checkpoints and recalculated the reported numerical metrics.

This probe omits native search, learning and padding. The earlier root-prefetch
execution qualification had a much smaller full-segment improvement for wide
execution and changed the learned trajectory. The next decision requires a
prospective learning comparison and useful-work accounting; exact output
equality is not required of every faster numerical implementation.

The first attempt failed during preparation because the original controller,
invoked from a frozen capsule, placed operational records inside that capsule.
All four source checks rejected the additional files before execution. The
records were preserved outside the snapshots, which verify again. The controller
now accepts an explicit workspace root and rejects writes inside either frozen
source. Seven pod tests passed, including four direct regressions. The
amendment (external or omitted experiment artifact) records the unchanged scientific
inputs and one additional attempt before any TPU measurement.

The successful attempt used 0.09753 chip-hours; the retained preparation failure
used 0.02853, totaling **0.12607 attempt chip-hours**. The broader reservation
also includes engineering and idle time. No rollout speed, MFU, strength or
production-promotion claim follows from this microbenchmark.
