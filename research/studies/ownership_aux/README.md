The first ownership component pilot does not justify promotion. Across its
16-game real KataGo panel, all 11 completed games were losses and five games
reached the 324-move limit. Incomplete games have no assigned result. This is a
one-seed exploratory screen at a zero-win floor, not evidence that ownership is
ineffective or that the two methods are equivalent.

[pilot_spec.json](pilot_spec.json) was registered before both training runs.
pilot_result.json (external or omitted experiment artifact) records the frozen analysis, every input
hash, the native/recovery evidence and the external match summaries. The complete
analysis entrypoint is recipe-owned `analyze_pilot.py`; it verifies the training,
checkpoint and evaluation identities before writing its report. The original
failed panel is retained under `runs/eval/ownership-pilot-6168b13c`.

Both arms used a 307,396-parameter plain-JAX CNN, native 16-simulation PUCT,
512 concurrent games across four hosts, and the qualified pass-alive scoring
profile. They differed only in ownership-loss coefficient: 0 versus 1.5. Each
started from scratch with seed 27, advanced 4,194,304 real moves and made 8,097
global learner updates. The shared native binary hash is
`01b32ef6b09209eee8e02cdc10626dcde526c9648b10c4d1b606313fb7c48ce9`.

| Measurement | Coefficient 0 | Coefficient 1.5 |
|---|---:|---:|
| Completed self-play games | 26,673 | 18,275 |
| Move-limit self-play truncations | 3,511 | 3,598 |
| Eligible terminal training rows | 2,995,489 | 2,971,185 |
| Useful neural evaluations | 69,420,131 | 69,932,061 |
| Slowest-host training segment | 318.51 s | 321.97 s |
| Mean CPU cores active per host | about 3.0 | about 3.0 |

Different game lengths and truncations make completed-game counts alone a poor
sample-efficiency metric. Both arms had the same real-move and update budgets;
this is not a repeated systems timing study. Raw compiler cost estimates are
retained, but no MFU claim is made.

| Opponent budget | Control outcomes | Ownership outcomes |
|---|---|---|
| Official KataGo, 1 visit | 0 wins, 3 losses, 1 incomplete | 0 wins, 2 losses, 2 incomplete |
| Official KataGo, 16 visits | 0 wins, 4 losses | 0 wins, 2 losses, 2 incomplete |

The official 9x9 b18 checkpoint and executable were pinned. Each opponent level
used both colors from an empty board and the C3/G7 opening. Eight child match
jobs used disjoint physical CPU cores. All 3,359 played boards matched, and all
completed numeric scores matched under the selected profile. There were no
transport failures or clock timeouts; the five incomplete games reached the
move cap. CPU evaluation overlapped only the separate recovery qualification,
not either training comparison segment.

Fresh CPU recovery matched all 56 arrays, actor states and 25 subsequent game
records. Fresh multi-host TPU continuation from turn 4,096 to 8,192 matched all
320 arrays, complete actor states and 9,747 subsequent game records. Timing
counters were excluded; no faults were injected. Ownership loss gradients,
D4 target transforms, zero-coefficient isolation, and full CNN/momentum updates
on one versus four CPU devices passed. Native tests also verify label
perspective, terminal score consistency and truncation exclusion.

The last replay windows averaged 6.86 and 6.19 visited root actions at a
16-simulation budget; only 6.5% and 5.8% of their roots visited 16 distinct
actions. These observations do not establish that uniformly broad search
explains the poor play. A network-value FPU control is a targeted follow-up;
its native setting and 3x3 CPU path are qualified in `low_visit_puct`, but its
9x9 configurations have not yet run. Search calibration, an informative opponent
ladder, stronger controls and larger actor batches precede any scale-up decision.
Independent learning seeds and larger held-out panels remain necessary.
