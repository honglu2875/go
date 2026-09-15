# Complete-state checkpoint forks

The owned `continued_selfplay` recipe can start a new immutable experiment from a complete online checkpoint without resetting parameters, SGD momentum, replay, either learner RNG, unfinished games, actor RNGs or work counters. Ordinary resume continues to require the exact child source/configuration. The parent native binary retains its original identity and is an explicit verified dependency for both training and GTP evaluation.

The initial contract permits only a stop turn, checkpoint/log cadence and an explicitly declared constant learning-rate override. It rejects model-code/architecture, rules, search, replay shape, batch, seed, worker-placement and topology changes. Parent files are never rewritten. Each child checkpoint records initialization lineage and phase counters separately from cumulative work. A later same-source resume can survive removal of old parent replay arrays while retaining the small source, descriptor and native dependencies.

| Qualification | Exact evidence |
| --- | --- |
| CPU fork versus original continuation | 82 arrays, complete actors/scientific state, 20 subsequent games |
| CPU child stop/restart | 82 arrays, complete actors/scientific state, 12 subsequent games |
| multi-host 9×9 fork versus original continuation | 648 arrays, complete actors/scientific state, 328 subsequent games |
| multi-host child stop/restart | 648 arrays, complete actors/scientific state, 135 subsequent games |
| Recorded real-game histories through both adapters | 566 search results, 1,144 board comparisons, 16 terminal-score comparisons |
| Fresh real KataGo games with one search thread | 8 completed games; identical paired moves/search/SGFs and 946 checked boards |

The four short TPU attempts used 0.530024 recorded attempt chip-hours. This excludes the rest of the allocation window. No failure was injected; host loss, external durability, altered topology, strength and utilization improvements are outside these checks' scope.

Two failed checks remain visible. The first data-only unit fixture omitted its required `train.py` and was corrected before exercising the contract. The initial real-game exactness test used the existing four-search-thread KataGo configuration. Its eight games completed, but identical seeds and parameters did not produce identical opponent moves, so that criterion failed. Replaying all eight observed histories through both adapters established exact candidate behavior. A separately registered single-search-thread configuration then passed the original exact-game criterion on eight new games. The initial failure was not overwritten.

The implementation adds `gozero.checkpoint_forks`, an owned plain-JAX online trainer, explicit native dependencies in model-artifact validation, and corresponding learned GTP/match support. Six fork/evaluation-contract tests and four existing model-artifact tests passed. The native build/source checks remain in force for ordinary historical candidates.

Protocols and raw references are in `cpu_spec.json`, `tpu_spec.json`, `gtp_spec.json`, `recorded_gtp_spec.json` and `single_thread_gtp_spec.json`. `result.json` and `artifacts.json` seal the closed qualification, including failures. The bounded learning-rate experiment is tracked separately under [online annealing](../online_annealing/README.md).

Use `checkpoint_forks.describe` to verify every parent rank before creating a descriptor, then freeze the new recipe and resolved configuration. Descriptors bind explicit source paths; regenerate the descriptor/config references when creating a differently named clone. The training/model code is owned by the clone and does not import another mutable recipe.
