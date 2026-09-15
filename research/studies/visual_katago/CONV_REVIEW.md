The CNN visual encoder improves the causal transformer's fixed-data endpoint, while the KataGo CNN retains the lead under the shared 1e-4 learning-rate schedule. All three models completed 1,024 updates and passed independent checkpoint, sampling and validation audits.

| Model, all width 768 | Parameters | Validation KL ↓ | Top-move agreement ↑ | Training minutes |
|---|---:|---:|---:|---:|
| KataGo nested CNN | 232,431,872 | 0.513569 | 62.83% | 31.98 |
| Transformer, linear visual patches | 234,409,728 | 0.826776 | 54.10% | 43.38 |
| Transformer, CNN visual encoder | 234,588,416 | 0.640906 | 58.84% | 43.40 |

The encoder reduces KL by 0.185869, closing 59.34% of the preceding transformer's gap to the CNN. It improves top-move agreement by 4.74 percentage points at essentially unchanged measured training time. The complete neural decoding calculation remains within 1% of the CNN's parameter and operation budgets at 9×9, batch 128 and 128 prior moves, including the encoder, policy head and pending action. Other history lengths and full-sequence training are not equated. The original decoding implementation measures 82.09 ms for this transformer versus 16.38 ms for the CNN; equal arithmetic does not imply equal latency.

The encoder consists of two 3×3 convolutions with 64 channels, each followed by per-point RMSNorm and SiLU. Their output becomes the same 36 overlapping 2×2 visual tokens. The 34-layer causal transformer, width 768, SwiGLU width 2,304, policy head, complete game history and learning schedule are unchanged. All unchanged parameters initialize to the same bytes. The encoder and transformer use no batch normalization.

All runs use identical episode and D4 draws: 11,469,333 position exposures from 836,486 training positions. Every evaluation uses the same 102,339 positions from 1,170 held-out games. The original fixed expert MCTS targets are preserved; these come from earlier weak self-play teachers, rather than KataGo checkpoints. The test split remains closed.

A separate evaluation loaded the saved parameters, executed each exact original model implementation, and reproduced the original aggregate and phase metrics. The CNN encoder reduced per-game KL relative to linear patches on 1,165/1,170 games. The position-weighted KL difference was −0.185869, with a whole-game bootstrap 95% interval [−0.190983, −0.180806]. The remaining gap to the KataGo CNN was +0.127337 [0.122977, 0.131605]. These intervals describe validation-game sampling uncertainty, not training-seed variability.

Moves 16–63 show the largest improvement and remaining gap: KL changes from 0.981321 with linear patches to 0.712045 with the CNN encoder, versus 0.520984 for the CNN. This region contains 53,317 validation positions. The 256+ region has only 97 positions and does not support a strong phase-specific conclusion. The CNN-encoder transformer still improves from 0.658202 to 0.640906 over the final 128 updates, and clips gradients on 964/1,024 updates. Neither fact establishes convergence or identifies an optimal learning rate.

The next intervention is the user-confirmed learning-rate sweep on the CNN-encoder transformer, before another encoder or token-count change. Reuse this 1e-4 run and train registered peaks 3e-4, 1e-3 and 3e-5 sequentially for the complete 1,024 updates. Keep initialization, exact data draws, 64-update warmup, cosine schedule ending at 30% of peak, and all other AdamW settings fixed. Select by final validation KL after reporting every arm. The runner independently audits each arm before starting the next and stops on failure for diagnosis.

This remains a one-seed policy learnability comparison, not a Go playing-strength or RL-efficiency result. The CNN retains KataGo's same-target training helper under the common AdamW adaptation; this is not a reproduction of the entire production KataGo training recipe.

Learning curves (external or omitted experiment artifact) · Phase errors (external or omitted experiment artifact) · Audited review (external or omitted experiment artifact) · Paired game audit (external or omitted experiment artifact) · [LR registration](lr_sweep_registration.json)
