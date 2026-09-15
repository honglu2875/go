The encoder comparison uses the same **11,469,333 training-position exposures**, exact episode/D4 draws and full validation set. Transformer width is 768 with 34 causal layers; encoder channels and the registered FFN compensation vary.

| Encoder channels | Peak LR | Final validation KL | Top-move agreement | Learning minutes | Decode ms |
|---|---:|---:|---:|---:|---:|
| 64 | 1e-04 | 0.640906 | 58.84% | 43.40 | 82.09 |
| 64 | 3e-04 | 0.540367 | 61.93% | 43.40 | 81.99 |
| 64 | 6e-04 | 0.644910 | 58.70% | 43.40 | 82.09 |
| 128 | 1e-04 | 0.628599 | 59.32% | 43.61 | 81.92 |

Grid incomplete; pending cells: [(128, 0.0003), (128, 0.0006)]

One seed, fixed weak-teacher dataset, equal exposure and complete logical decoding-FLOP budget. Width and compensated FFN shapes differ. Results do not establish Go strength or RL efficiency.

Selected CNN AdamW reference: validation KL 0.420361 and 66.66% top-move agreement. Historical CNN SGD: KL 0.424920 and 66.31%.

Curves (external or omitted experiment artifact) · Audited comparison (external or omitted experiment artifact)
