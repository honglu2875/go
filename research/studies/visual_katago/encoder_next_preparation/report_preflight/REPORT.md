Encoder interventions use the same **11,469,333 training-position exposures**, exact game/D4 draws and full validation population. Decoder width remains 768. Registered decoder depth and FFN compensation maintain the full parameter/decoding budget when encoder capacity changes.

| Experiment | Peak LR | Final validation KL | Top-move agreement | Learning minutes | Decode ms |
|---|---:|---:|---:|---:|---:|
| c64_lr_1e-04 | 1e-04 | 0.640906 | 58.84% | 43.40 | 82.09 |
| c64_lr_3e-04 | 3e-04 | 0.540367 | 61.93% | 43.40 | 81.99 |
| c64_lr_6e-04 | 6e-04 | 0.644910 | 58.70% | 43.40 | 82.09 |
| c128_lr_1e-04 | 1e-04 | 0.628599 | 59.32% | 43.61 | 81.92 |
| c128_lr_3e-04 | 3e-04 | 0.521731 | 62.68% | 43.62 | 81.96 |
| Selected CNN AdamW | 1e-3 | 0.420361 | 66.66% | 31.98 | — |

Registered wave complete.

Single seed, fixed weak-teacher targets, matched complete logical decoding budget at the registered reference context. Final validation KL compares learnability, not Go strength or RL efficiency. Choices between waves are adaptive.

Curves (external or omitted experiment artifact) · Audited comparison (external or omitted experiment artifact)
