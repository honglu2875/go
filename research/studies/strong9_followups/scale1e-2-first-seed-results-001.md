The encoder residual-scale initialization changed from 1e-6 to 0.01.
All widths, parameter shapes, encoder/decoder arithmetic, LR settings and
training targets remained fixed. The first seed completed all 4,096 accepted
updates and 49,753,641 position exposures; every rank, draw sequence and
validation population passed the full audit.

| Measure | Original transformer | Scale 0.01 | Relative improvement |
| --- | ---: | ---: | ---: |
| Position KL | 0.133493900 | 0.131976008 | 1.137% |
| Equal-family KL | 0.093453884 | 0.092829466 | 0.668% |

The last-three means improve by 1.413% and
0.184%, respectively, with no sustained overfit
flag. This passes the existing first-seed screen. The conditional second seed
was launched by the original continuation; the intervention is **not yet
accepted**. No subsequent intervention has been selected.

The paired CNN endpoint remains lower: 0.129946470 position KL and
0.089861631 family KL. The candidate is still
1.562% and
3.303% higher at equal updates.
Its learning-update time was 5937.93 seconds.
The complete final checkpoint is retained on the owner and a verified peer.

The readable final plot is `scale1e-2/seed1-final-001.png`, with its CSV beside
it. Live plots and all raw audit/contrast evidence remain retained.
See [the first-seed contrast](scale1e-2/seed1-contrast-001.json).
