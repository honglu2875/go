The historical SGD control and CNN AdamW controls used the same **11,469,333 position/symmetry exposures**, shuffled and regrouped into256-position SGD batches. All share the same model parameters at initialization and full validation population.

| Recipe | Updates | Final validation KL | Top-move agreement | Learning minutes | Attempt chip-hours |
|---|---:|---:|---:|---:|---:|
| SGD: historical256-position recipe | 44,803 | 0.424920 | 66.31% | 45.79 | 13.54 |
| AdamW: 1e-3 (selected) | 1,024 | 0.420361 | 66.66% | 31.98 | 10.31 |
| AdamW: 1e-4 (original) | 1,024 | 0.513569 | 62.83% | 31.98 | 10.31 |

SGD uses Nesterov momentum0.9, per-sample LR2e-5 through the first5M exposures then6e-5, and coupled L2 coefficient3e-5 in the historical TensorFlow half-squared-norm convention. The final batch has21 positions. There is no gradient clipping, Lookahead, cosine decay or compressed imitation of the original19-day run’s final LR drop.

The current9x9 fixed weak-teacher corpus, modern norm-free main CNN, same-target training helper and BF16 arithmetic differ from the2019 self-play run. This checks transfer of its optimizer recipe. Batch grouping/order, optimizer, schedule and regularization differ from our AdamW controls; update count and training FLOPs are not equal. AdamW1e-3 is a selected endpoint from a four-rate grid; SGD is one pre-registered recipe. The test split remains closed. No Go-strength or RL-efficiency conclusion follows.

Exposure and wall-time curves (external or omitted experiment artifact) · Audited comparison (external or omitted experiment artifact)
