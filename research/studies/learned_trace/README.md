The trained paired causal decoder passed exact sequential action-prefix qualification in both greedy and stochastic play. This establishes execution semantics for a fixed policy; it does not establish MCTS equivalence, TPU speed, learning efficiency or stronger Go play.

The same 890,240-parameter student plays both colors. Each speculative condition runs 128 packets for four games, with horizon 4. Forecast modes use four independent samples per player view; known-policy joint continuation uses one sample. The sequential control runs until it covers every speculative event stream. All committed actions, terminal scores and episode resets match its prefixes exactly.

| Forecast condition | Greedy moves/game/dispatch | Stochastic moves/game/dispatch |
|---|---:|---:|
| Initialized behavior head, trained own policy | 1.0566 | 1.0762 |
| Trained behavior head | 1.7051 | 1.3066 |
| Known policy, independent forecast draws | 3.2344 | 1.3867 |
| Known policies, actual self-play draws | 3.2344 | 3.0039 |
| Sequential control | 1.0000 | 1.0000 |

Known-policy joint continuation is available when both self-play policies and their samplers are controlled by the engine. Predicting an external opponent does not provide its future randomness or unknown policy. The stochastic independent-policy condition shows why matching policy probabilities alone does not imply agreement on future sampled actions. These are conditional observations from one fixed student and seed, without a population-level effect estimate. Greedy games from identical roots can repeat exactly.

Each multi-step packet still prefills 329 token slots per game. Depending on mode, 86.7–91.4% of those slots are padding in these runs. The initialized and trained behavior controls have the same own-policy parameters and executable shapes; only the private behavior parameters differ. The known-policy joint condition uses one quarter of the branch append slots of the four-sample modes. Fixed packet counts yield different real work, and the baseline sometimes advances shorter game streams beyond the comparison endpoint. Raw elapsed times therefore are diagnostics, not a registered speed comparison. Own-policy legality corrections are recorded; they are not the illegal rate of all forecasts.

The two corrected executions took about 50 seconds each on CPU. They required no new learner update. All teacher/student training costs remain in the causal-distillation lineage. TPU MFU and CPU scaling remain unmeasured for this decoder, whose native batch resolver is still serial.

The original two attempts stopped before model inference because the harness incorrectly demanded identical NPZ serialization after transfer from TPU to CPU. All 44 arrays matched in dtype, shape and canonical element bytes; two matrices changed Fortran/C layout. `spec.json`, `attempts.json`, their frozen sources and full logs retain those failures. `corrected_spec.json` registered the corrected element-wise identity check before either new execution; no action-equivalence threshold changed.

- Corrected protocol: `14427f90a0e2c8d46dd184d5deeeea3252c856086ca84f7ccd8ab9705686511b`.
- Greedy source: `c2d0154613f5f31aef0a88f4052d1ddf6ec565ce8def54cc35127f9890ba9c2d`.
- Stochastic source: `a603423512bb3f249f2e30f70d7931a77fc2838c1a77399635965f80d0a82eb9`.
- Both native receipts, HLO, parameter exports, packets and event streams are retained in the corresponding `runs/learned-trace-*` directories.
- Audit source: `c8b18d4a27e126101e941f1aa78feeb832e80f11d364b4ca8cee8b58cedce691`.
- Audited result: result.json (external or omitted experiment artifact), SHA256 `22ae36cfb221247d440e68293068e229ee07894f0f419123571a7e4d09a9456a`.

Reproduce the audit with that snapshot's `research/recipes/learned_trace/analyze.py --artifacts-root /workspace/go --output <fresh-path>`. Study specifications are external, immutable, hash-pinned artifacts; recipe capsules intentionally exclude `research/studies`. The first analysis source mistakenly looked for them inside its capsule; the failed analysis is recorded in `analysis_failures.json`. Experiments were not rerun for that analysis fix.

Next studies should compare matched real work on TPU, improve forecast coverage at fixed branch budget, and test explicit board-state conditioning. The separate [causal MCTS qualification](../causal_mcts/README.md) confirms that this student can use the real benchmark harness, while also showing weak play.
