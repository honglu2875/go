# Paired architecture learning curves

The immutable [registration](spec.json), SHA-256
`d7dbd7461cdc76e2222609971b8895f17d6baa7cfcec82d2515449130a61b6d3`,
compares the Gumbel CNN and attention models at four saved training budgets,
using independent training seeds 27 and 28. It was frozen before these matches;
the seed-28 attention training attempt was still running.

Each of the 16 checkpoints plays 64 games on the same 32 new paired openings
against the officially released early KataGo level 3 retained by the separate
calibration confirmation (external or omitted experiment artifact). Candidate
search uses 16 Gumbel simulations and KataGo uses one visit. The study permits
1,024 games, 7,200 panel wall seconds and no new training or automatic retries.
The per-game cap is 648 moves and the deadline is 60 seconds.

The primary comparison averages attention-minus-CNN score equally over the
four real-move budgets and two fixed seeds. A capped game contributes an
unassigned interval [0, 1], never a draw. The conservative difference uses
attention's lower score bound minus CNN's upper bound. Bootstrap sampling keeps
both colors, all four budgets and both seeds together within each opening unit.
The registered 10,000-draw PCG64 bootstrap is conditional on these checkpoints;
it does not estimate population variation over training seeds.

The criterion requires all scheduled records, no integrity or process failure,
at least 90% completed games at every checkpoint, a positive bootstrap 95% lower
limit for the conservative difference, and a positive conservative mean
difference separately in both seeds. All curves and unresolved bounds are
reported regardless of the decision. Strong-anchor matches and the independent
direct-match replication remain separate evidence.

`eval/run_curve_batch.py` serializes the registered panels and checks remaining
budget before launching each checkpoint. The initial 12 panels run from source
`4fb6ee819ecd671b821e9ad5ba1854beacc985fc4594dbd1a24b92afc3030803`.
Remaining checkpoint descriptors are frozen after their committed exports
become available. The registered evaluator file hashes and panel specifications
must remain identical across those source snapshots.

`eval/analyze_architecture_curves.py` validates all four checkpoint ranks and
their exact exported parameters, source/config/native identities, game records,
budgets and KataGo artifacts. It reports real moves, terminal replay rows,
actual neural evaluations, updates and global learner exposures. Historical
intermediate wall times were not recorded; only whole-attempt costs are reported.
Five statistical edge cases passed. The complete immutable analysis (external or omitted experiment artifact)
audited all 16 checkpoints and all scheduled game records: 102,181 board checks,
1,003 completed scores and 21 unassigned caps, with no failed games or integrity
failures. Panel execution used 1,477.42 seconds of the 7,200-second budget.

| Seed | Real moves (M) | CNN score bounds | Attention score bounds |
|---|---:|---:|---:|
| 27 | 4.19 | 3.1–4.7% | 29.7% |
| 27 | 8.39 | 48.4–56.3% | 81.3% |
| 27 | 12.58 | 57.8–68.8% | 85.9–90.6% |
| 27 | 16.78 | 75.0–76.6% | 85.9–89.1% |
| 28 | 4.19 | 3.1–4.7% | 32.8% |
| 28 | 8.39 | 46.9–48.4% | 78.1% |
| 28 | 12.58 | 78.1% | 93.8% |
| 28 | 16.78 | 84.4% | 87.5% |

These bounds cover every scheduled game; ranges reflect unresolved caps and are
not confidence intervals. The figure (external or omitted experiment artifact) connects the
lower bounds and shows the unresolved ranges. No intermediate training budgets
were evaluated or interpolated into the primary statistic.

The mean attention-minus-CNN difference lies in **[19.14, 23.24] percentage
points**. The paired bootstrap 95% interval for the conservative lower bound
is [16.02, 22.46] points; the outer interval covering both bounds is
[16.02, 26.95]. The conservative mean is positive separately in both seeds.
This is conditional on the two trained seeds and the registered opening panel.

**The full registered criterion was not met.** The seed-27 CNN checkpoint at
12.58M moves completed 57/64 games (89.06%), below the required 90%. Its seven
caps remain unresolved even though the conservative effect estimate is
positive. The study supports a follow-up hypothesis, with no production
promotion or claim of superiority over current KataGo. The separate direct
replication and strong anchor remain required. Both training attempts were
complete when analyzed; the copied registration limitation about an ongoing
seed-28 run describes its status when the protocol was frozen.
