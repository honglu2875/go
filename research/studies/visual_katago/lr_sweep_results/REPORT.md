The lowest final validation KL in this CNN-encoder transformer sweep is **0.540367 at peak LR 3e-04**. All four arms completed 1,024 accepted updates and passed independent checkpoint and data-draw audits.

| Peak LR | Final LR | Validation KL ↓ | Top-move agreement ↑ | Training minutes | Clipped updates |
|---|---:|---:|---:|---:|---:|
| 3e-05 | 9e-06 | 0.996168 | 49.14% | 43.39 | 1024/1024 |
| 1e-04 | 3e-05 | 0.640906 | 58.84% | 43.40 | 964/1024 |
| 3e-04 | 9e-05 | 0.540367 | 61.93% | 43.40 | 239/1024 |
| 1e-03 | 3e-04 | 0.678066 | 57.49% | 43.39 | 288/1024 |

The selected endpoint differs from the existing 1e-4 reference by -0.100539 KL. This selects a candidate within the registered range and schedule; it does not establish a globally optimal LR or convergence.

All non-LR settings, initial parameter bytes, episode and D4 draws, validation games and evaluation cadence matched exactly. Each arm saw 11,469,333 position exposures from the same 836,486 training positions. Validation contains 102,339 positions from 1,170 games; the test split remains closed.

The architecture is unchanged: two 3×3 convolutions with 64 channels, per-point RMSNorm and SiLU, 36 overlapping visual tokens, and a width-768, 34-layer causal transformer with one policy head. Encoder and complete neural decode arithmetic are unchanged across rates. Warmup remains 64 updates; each cosine schedule ends at 30% of its peak.

Learning dynamics at the registered validation points:

| Peak LR | KL at 128 | KL at 512 | KL at 896 | KL at 1024 |
|---|---:|---:|---:|---:|
| 3e-05 | 2.252012 | 1.242977 | 1.027022 | 0.996168 |
| 1e-04 | 1.977043 | 0.818848 | 0.658202 | 0.640906 |
| 3e-04 | 1.638951 | 0.722762 | 0.563500 | 0.540367 |
| 1e-03 | 1.668407 | 0.882006 | 0.699335 | 0.678066 |

This is a one-seed, fixed weak-teacher policy-learning screen. LR selection uses validation data and carries selection uncertainty. CNN and linear-patch transformer learning rates were not swept here, so their earlier common-schedule comparison and this tuned result answer different questions. These losses do not measure Go strength or RL improvement.

Learning curves (external or omitted experiment artifact) · Phase errors (external or omitted experiment artifact) · Audited analysis (external or omitted experiment artifact)
