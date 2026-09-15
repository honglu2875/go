# Official KataGo ladder calibration

The [registered calibration](calibration_spec.json) completed with the frozen
analysis (external or omitted experiment artifact). Four early official KataGo checkpoints
were downloaded from the [official network index](https://katagotraining.org/networks/)
and pinned by content hash. Every level used one KataGo visit and 16 candidate
search simulations on 9x9, with two openings and both colors.

| Official checkpoint suffix | PUCT wins / losses / caps | Gumbel wins / losses / caps |
|---|---:|---:|
| s938496-d1208807 | 4 / 0 / 0 | 4 / 0 / 0 |
| s10014464-d2201128 | 0 / 0 / 4 | 4 / 0 / 0 |
| s16525312-d2925067 | 1 / 3 / 0 | 3 / 1 / 0 |
| s24455424-d3879081 | 0 / 4 / 0 | 3 / 1 / 0 |

All 32 scheduled games were recorded. The four capped games have no assigned
outcome. Every played board and completed score matched the real KataGo engine.
The two harder levels both meet the declared intermediate-score calibration
rule for Gumbel, so both should proceed to a fresh larger panel. Four games per
level do not support an algorithmic strength or sample-efficiency claim. The
selection rule does not choose whichever level shows the largest method gap.

These models are historical small networks, not current KataGo. Their published
site ratings use other conditions and must not be interpreted as 9x9 or human
ratings. The strong pinned 9x9 checkpoint remains in the benchmark suite.
No KataGo weights, policy labels, or game records enter our training.

The [larger confirmation](confirmation_spec.json) also completed. Its
frozen analysis (external or omitted experiment artifact) validates 128 games on 16 fresh
opening pairs per arm and level. PUCT recorded 1 win, 30 losses and one cap
against level 2, and 32 losses against level 3. Gumbel recorded 27 wins and
5 losses against level 2, and 24 wins and 8 losses against level 3. All 14,204
played boards and all 127 completed scores agreed with KataGo. The two panels
used 157.31 seconds of their 2,100-second allowance.

The registered rule retains **level 3 only** for future learning curves: its
candidate score is 0.75, within the declared [0.2, 0.8] calibration range.
Level 2 scored 0.84375 and exceeds the ceiling. Both arms reused the discovery
checkpoints, so this confirms opening coverage rather than training-seed
robustness. Future curves require fresh openings and retain the strong anchor.
The protocol's anticipated overlap mentions the seed-28 run; actual overlap
was with the subsequent attention pilot on separate physical cores.
