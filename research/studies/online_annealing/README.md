# Online learning-rate continuation

This registered seed-27 pilot compares inherited SGD learning rate 0.02 with a fourfold reduction to 0.005 after the same complete 16,777,216-move checkpoint. Parameters, unscaled momentum, replay, RNGs, unfinished games, model code, search, rules and host/worker placement are retained. Each arm adds 4,194,304 global moves and 8,192 updates, reaching 20,971,520 cumulative moves. The parent checkpoint is also evaluated on the same fresh openings.

Both training arms completed and passed the independent complete-state audit. Each performed 8,388,608 global learner exposures. The inherited-rate arm produced 44,537 completed games and 328 move-limit truncations; the lower-rate arm produced 41,678 completed games and 127 truncations. Their 0.7311% and 0.3038% truncation fractions both exceed the registered 0.1% training threshold. The two attempts used 3.706148 recorded chip-hours, separate from qualification and the full allocation window.

The evaluation registration fixes 32 fresh two-stone opening pairs, both colors, 16 candidate simulations and 16 single-thread KataGo visits per model. The primary opponent is the pinned official historical b6c96 checkpoint. Four additional games per model use the existing stronger b18c384 checkpoint as a diagnostic; it is not the current 2026 best engine. No external engine targets or weights enter learning.

The primary screen is the paired-opening 95% bootstrap outer interval for annealed minus inherited score, with 20,000 draws and seed 91310317. Unresolved move-limit results remain [0,1] and propagate into uncertainty. Both primary arms must complete at least 98% of games. A positive result would require independent training-seed replication. Full-study promotion also requires successful execution and the registered training stability gate.

All 204 scheduled games are closed. The independent audit verified 20,536 observed positions and all 198 completed scores. One move-limit game and five diagnostic timeouts remain unassigned.

| Checkpoint | Historical primary: wins / losses / caps | Stronger diagnostic: wins / losses / timeouts |
| --- | --- | --- |
| Parent, 16.8M moves | 52 / 12 / 0 | 0 / 2 / 2 |
| Inherited rate, 21.0M moves | 43 / 21 / 0 | 0 / 2 / 2 |
| Annealed rate, 21.0M moves | 46 / 17 / 1 | 1 / 2 / 1 |

The annealed-minus-inherited scheduled score difference is bounded by +4.6875 to +6.25 percentage points, but its paired 95% bootstrap outer interval is −9.375 to +21.875 points. The primary completion condition passed; the positive lower endpoint did not. Both training stability conditions also failed. The one stronger-checkpoint win is a small diagnostic observation, with incomplete execution, and does not support a strength claim. Neither continuation is promoted; retain the parent as the reference.

Each stronger-checkpoint panel encountered the registered 180-second deadline. The original suites remain failed. The audit separately verifies jointly observed board prefixes, preserves the timeout errors and distinguishes evidence integrity from complete execution. Its `status: passed` means the retained evidence was audited; `complete_study_execution: false` and `registered_combined_criterion_met: false` explicitly preserve the study failure. No run or opening was retried, and neither the primary criterion nor diagnostic time budget changed after seeing results.

`pilot_spec.json` contains the hypothesis, fixed budgets, complete evaluation-input closure, opening book and decision rule. `training_result.json` is the independent state/work audit; `result.json` contains the final external analysis and `artifacts.json` seals the raw records. Three paired-score tests and raw replay checks also passed. The native dependency, trainer and initialization contract were qualified in [the fork study](../continued_selfplay/README.md). [Next work](NEXT.md) separates online stability from measured actor/worker utilization and speculative MCTS integration.
