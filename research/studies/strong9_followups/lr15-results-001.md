The 1.5× learning-rate intervention did not pass its first-seed screen. Retain
the original peak 1e-3 / final 3e-4 schedule. The completed trial used peak
1.5e-3 / final 4.5e-4, with the same 4,096 updates and 64-update warmup.

| Validation measure | Original LR | LR ×1.5 | Candidate excess |
| --- | ---: | ---: | ---: |
| Final position-weighted KL | 0.133494 | 0.134750 | 0.94% |
| Final equal-opening-family KL | 0.093454 | 0.094697 | 1.33% |
| Last-three mean position KL | 0.136352 | 0.138128 | 1.30% |
| Last-three mean family KL | 0.095167 | 0.097087 | 2.02% |

The higher rate helped some early validation points, but that advantage did
not persist through the full registered horizon. The fixed training probe
and validation both kept improving; neither co-primary metric triggered the
sustained-overfit rule. There is no evidence here to reduce the dataset or
shorten the training horizon to address overfitting.

The [all-rank audit](lr15/seed1-audit-001.json) and
[paired contrast](lr15/seed1-contrast-001.json) verify identical initialization,
game and D4 draws, 49,753,641 training position exposures, and evaluation
populations. All 4,096 updates were accepted. Measured learning time was
5,928.88 seconds, versus 5,922.69 for the original transformer seed.
The [final curves](lr15/seed1-final-002.png) retain all 17 validation and
training-probe evaluations for this trial, its transformer parent and the
paired CNN control. The corresponding CSV is retained alongside the image.

The frozen rule required at least a 0.5% improvement in both final KL measures,
nonregressing last-three means, and no sustained overfit before a second seed.
The [continuation outcome](lr15-continuation/outcome-001.json) closed without
launching seed 2. This rejects this rate change under this screen; it does not
establish a globally optimal LR or a playing-strength result.

The subsequent [review](scale1e-2-review-001.json) selects an isolated encoder
residual-scale initialization change from 1e-6 to 0.01, using the original LR.
Its [registration](scale1e-2-registration-001.json) preserves architecture,
parameter count, complete decoding FLOPs, data and horizon. Review that full
learning result before selecting another intervention.
