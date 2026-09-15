The first equal-width causal transformer finished all 1,024 updates and passed the independent checkpoint and sampling audit. The CNN has the better fixed-data endpoint under the shared schedule.

| Model | Parameters | Validation KL ↓ | Top-move agreement ↑ | Measured training minutes |
|---|---:|---:|---:|---:|
| KataGo nested CNN, width 768 | 232,431,872 | 0.513569 | 62.83% | 31.98 |
| Causal transformer, width 768, linear visual patches | 234,409,728 | 0.826776 | 54.10% | 43.38 |

Both runs saw exactly the same episode and D4 draws: 11,469,333 position exposures sampled from 836,486 training positions. Every evaluation used the same 102,339 positions from 1,170 held-out games. The test split remains closed. All updates and checkpoint arrays were finite. The transformer clipped gradients on 972/1,024 updates; the CNN clipped on 883.

The largest phase KL difference is on moves 16–63: 0.981321 for the transformer versus 0.520984 for the CNN, covering 53,317 validation positions. The transformer still improved from 0.866303 to 0.826776 over the final 128 updates. This endpoint does not establish convergence or long-run scaling. The common optimizer is a control, not a claim that either model's hyperparameters are optimal.

The reference transformer has 34 layers, width 768, SwiGLU hidden width 2,304, 12 query heads, four KV heads, RoPE, and per-token RMSNorm. Each board's 22 spatial feature planes become 36 visual tokens through overlapping 2×2 linear projections; 19 global features are projected and broadcast. The sequence repeats `[board tokens, policy readout, played action]`, using full causal attention over the game prefix. The readout cannot see the action it predicts. One tied policy head serves the fixed expert target. There are no behavior/value tasks or batch statistics in this transformer.

Decision: test two 3×3 convolution layers with 64 channels, per-point RMSNorm and SiLU before the unchanged overlapping 2×2 visual patches. Keep the trunk, head, history, data draws and training schedule fixed, including bitwise-identical initialization of unchanged parameters. This tests local board encoding as one possible source of the early learning deficit; the phase result alone does not establish that cause.

The complete neural decoding budget is matched within 1% at 9×9, global batch 128 and 128 prior moves, including the encoder, head, pending action and actual attention extent. Other histories, 19×19 and full-sequence training are not equated. Warm original decoding measured 16.38 ms for the CNN and 81.98 ms for the transformer; CPU feature generation and transfers are excluded.

This is a one-seed, policy-only screen on fixed weak-teacher data. The CNN preserves KataGo's same-target training helper and uses the common AdamW recipe. It is not the entire production KataGo training setup, and these losses are not playing-strength or RL-efficiency measurements.

Learning curves (external or omitted experiment artifact) · Phase errors (external or omitted experiment artifact) · Paired audit (external or omitted experiment artifact) · Decision record (external or omitted experiment artifact)
