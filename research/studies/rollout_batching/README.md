# Dispatch and batching study

Queued inference improved measured fixed-network rollout throughput by a median
1.1812×, corresponding to about 15.3% less wall time. It missed the registered
1.20× screen threshold. This is a modest systems result, not a measured training,
sample-efficiency, or MFU improvement.

all configured hosts ran 1,024 native games with 32 Rust workers and the same fixed
307,396-parameter model. Each inference executable evaluated 128 positions per
host across four local TPU chips. Queuing eight independent calls before host
materialization preserved the executable, data placement and all outputs.
The 32-turn warmup preceded 128 measured turns. No learner ran.

| Repetition | Serial segment, seconds | Queued segment, seconds | Throughput ratio |
| --- | ---: | ---: | ---: |
| 1 | 25.382 | 21.488 | 1.1812× |
| 2 | 25.296 | 21.107 | 1.1984× |
| 3 | 25.266 | 21.888 | 1.1543× |

Values use the slowest host per attempt. The order was serial/queued,
queued/serial, serial/queued. Median inference speedup was 1.2165×. All useful
work counters matched across all six attempts. No verification hashing,
duplicate reference inference, evaluation panel or native build ran during the
timing study. Its six complete attempt windows total 1.2021 chip-hours; this is
only a subset of the reserved allocation window.

Before timing, exact qualification checked all feature requests, training
targets, completed game records, complete actor states and unfinished root
values against the serial run on all configured hosts. It also compared 11,141,120
position evaluations (including warmup) with serial reference inference, with
identical outputs. Each measured segment advanced 524,288 real moves globally.
This workload spans early and middle game states; it does not establish
steady-state throughput for arbitrary models or larger searches.

The initial direct batch-1,024 comparison failed exact trace parity. A follow-up
compiled device loop retaining microbatch128 also changed outputs and traces.
The retained identical-input probes demonstrate numerical differences, including
at the empty board. These are not evidence that wider or looped inference plays
worse; they require validation as approximate execution variants, using a
numerical reference corpus and independent learning/evaluation runs. Their
diagnostic runtimes must not be presented as equivalent-work speedups. The
current study does not implement speculative moves or the proposed paired
causal decoder.

`qualification_spec.json` and `dispatch_qualification_spec.json` register the
qualifications. `qualification_failed.json`, `scan_qualification_failed.json`,
`async_qualification.json` and `numerics_result.json` retain both successful and
unsuccessful findings. `timing_spec.json` was frozen before the timing runs;
`timing_result.json` comes from the frozen `measure.py` in source
`4888871d83bf5564d4d3eb9682a2cabd677bb1ee125878be3c6b80d9b51464f4`.
All source, config, model, native and attempt identities are in those reports.
