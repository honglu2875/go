# Matched CNN, dense-transformer and MoE comparison

At 64 updates:

| Architecture | Parameters | Policy KL | Value MSE | Top-1 | Learning hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn | 233,220,870 | 1.343490 | 0.359578 | 0.3560 | 0.948 |
| dense | 232,011,540 | 1.133431 | 0.169567 | 0.4195 | 1.209 |
| temporal | 317,001,492 | 1.102060 | 0.186074 | 0.3668 | 1.214 |
| all_experts | 420,991,764 | 1.086170 | 0.195272 | 0.4009 | 2.715 |

At 128 updates:

| Architecture | Parameters | Policy KL | Value MSE | Top-1 | Learning hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn | 233,220,870 | 1.034096 | 0.225195 | 0.4479 | 1.835 |
| dense | 232,011,540 | 0.842229 | 0.129620 | 0.4808 | 2.362 |
| temporal | 317,001,492 | 0.844658 | 0.127651 | 0.4644 | 2.370 |

One seed on fixed teacher data. Common 64/128-update endpoints of an unchanged 512-update schedule; architecture-specific helper losses preserved. Historical timing is not simultaneous wall time. No extrapolated all-expert 128 result, test access, playing-strength or MFU claim.

All-rank state, exact logical-rank game/D4 draws, evaluation populations and checkpoint replicas were verified. JSON includes train/validation separation, last-three means, router diagnostics, overfit flags and the secondary attention-pooling reference.
