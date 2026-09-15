Removing cross-host reductions from search inference improved the fixed 9×9 workload by **1.40× end to end** and **2.03× in inference time** (median paired ratios). All saved learning arrays, actor states, counters describing useful work, and game artifacts matched exactly. This establishes a systems improvement on this workload; it does not establish sample-efficiency gains, MFU, or strong Go play.

[The specification](spec.json) was frozen before the pilot and measured sequence. The full result (external or omitted experiment artifact) records all attempts, hashes, per-host equivalence checks and timings. The source-qualified controller ran three fresh interleaved baseline/candidate pairs with no concurrent evaluation workloads. The same seed is intentional for exact systems controls, not independent learning replication.

| Pair | Baseline training seconds | Local inference training seconds | Speedup |
|---|---:|---:|---:|
| 1 | 31.092 | 21.763 | 1.429× |
| 2 | 30.249 | 21.643 | 1.398× |
| 3 | 30.416 | 22.251 | 1.367× |

The primary time is the slowest host's complete training segment, including checkpointing and game-record I/O. Staging, startup, compilation and collection are separately retained in each pod receipt. Paired end-to-end speedups ranged from 1.367× to 1.429×. These are three measurements on the supplied pod; no broad hardware or workload confidence claim is made.

Both variants advanced 262,144 real moves, performed 4,350,678 useful neural evaluations, completed 1,398 games, retained 209,315 training positions and made 418 global learner updates. The network, seeds, native code, CPU sets, search budgets, replay sampling and global optimizer batches were held constant. The three comparisons checked 900 saved arrays in total and all 8,604 game JSON/SGF artifacts. Final weights have SHA-256 `cdf6a6ece4f6cae578a1de5a8e5a404c07239597406682a97e5894f9cf0fa643` in every run and host.

The baseline ran one global inference graph with a reduction of active slots, including an extra all-padding dispatch to discover completion. The candidate reuses local device views of replicated weights, runs a graph over each host's four TPU devices, and resolves search completion locally. Global learning and checkpoint collectives retain their original order and semantics. Root searches still use frozen network versions. No speculative moves, smaller search budgets or skipped learning updates explain the speedup.

Padding slots fell from 4,718,592 to 4,456,448 while useful evaluations remained fixed. A CPU↔TPU round trip still occurs for each native search batch. Native buffer collation, Python dispatch, checkpoint I/O and remaining small-model inefficiencies are still measurable costs. Larger independent-game batches, isolated CPU sets, overlap, and the user's multi-step graphs should be compared against this stronger control.

Baseline source: `7a7745037997ddad53162a92d2fbead438bc9a6235b240e9e973142772618b66`. Candidate source: `d2d32bc65e032c9b77e5d50e60c9529505747241f0c7032aae63ca85f8215e26`. The successful pilot is `pod-20260911T053158Z-b9e66752`; its exact equivalence report is retained separately. The six-run controller ledger is `runs/studies/local-inference-5087e6ad/result.json`. The frozen analysis is `6bcba1474a83b732e32bb19fd03baa07b826721ed7775d1676489e24b176cd06`.

The six measured attempt windows total 1.3409 chip-hours. This excludes the pilot, previous qualifications, and reservation time outside the attempts. The reservation ledger (external or omitted experiment artifact) includes engineering and idle time within the observed allocation window. The local-inference recipe has not yet received its own fresh-process recovery/fault-injection qualification or production promotion.
