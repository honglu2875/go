Most of the completed larger9 transformer/CNN validation gap occurs in the
first 64 moves: 74.1% in paired seed 1 and 90.7% in seed 2. Those ranges contain
about 70.0% of the validation positions. The
[audited diagnostic](architecture-gap-001.json) decomposes the existing endpoint
KL aggregates; [the plot](architecture-gap-002.png) shows both seeds separately.

| Move range | Mean CNN KL | Mean transformer KL | Validation positions |
| --- | ---: | ---: | ---: |
| 1–16 | 0.11529 | 0.12528 | 247,040 |
| 17–64 | 0.19103 | 0.19562 | 723,954 |
| 65–128 | 0.03103 | 0.03353 | 402,146 |
| 129–256 | 0.00512 | 0.00791 | 14,495 |

The transformer has higher KL in every populated phase and all eight opponent
strata on both seeds. The first-16-move conditional gap is similar across
seeds; the next 48 moves account for much of the larger seed-2 gap. Rare
positions after move 128 contribute less than 1% of the gap on average across
the paired seeds. This supports focusing the next investigation on current
board representation and policy readout/optimization.

The transformer also has lower predicted entropy and lower target-top-1
agreement than the CNN on both seeds. Entropy averages alone cannot establish
miscalibration or choose a temperature; many soft-target rankings can be close.
These observations do not identify the cause of the KL difference.

This is post-hoc description, with no new test-set access or selection rule.
The validation set has a concentrated opening-family mixture, so phase totals
do not establish a general opening-strength weakness. Opponent and phase
partitions describe the same positions and must not be added together.
The running encoder initialization probe retains its original endpoint,
tail and overfit criteria; review its complete result before scheduling another
change. No readout or temperature intervention is selected by this diagnostic.
