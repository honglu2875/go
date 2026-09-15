The qualified KataGo policy CNN reached validation KL 0.513569 after its fixed 1,024 updates, with 62.83% top-move agreement. All updates were accepted. The independent audit reconstructed every episode and D4 draw across four hosts and verified the final model, Adam moments, RNG state and checkpoint hashes.

Training presented 11,469,333 positions sampled with replacement from 836,486 training positions in 9,466 games. Validation always used the same 102,339 positions in 1,170 held-out games. The original whole-game test split remains closed.

| Update | Validation KL | Top-move agreement |
|---:|---:|---:|
| 0 | 2.712115 | 4.35% |
| 128 | 1.030759 | 46.66% |
| 256 | 0.767716 | 54.46% |
| 384 | 0.679340 | 57.19% |
| 512 | 0.601285 | 59.69% |
| 640 | 0.602253 | 59.77% |
| 768 | 0.547210 | 61.40% |
| 896 | 0.522319 | 62.31% |
| 1024 | 0.513569 | 62.83% |

The final 128 updates reduced KL from 0.522319 to 0.513569. The endpoint is still improving; it is not evidence that the CNN has converged. Its largest remaining phase error is on moves 64–127 (KL 0.6662), compared with 0.3308 on moves 0–15. There are only 97 validation positions beyond move 255.

Measured learning time was 31.98 minutes, excluding compilation, sampling, evaluation and saving. The closed pod attempt took 38.65 minutes and 10.305 TPU chip-hours. Warm batch-128 neural inference had median latency 16.38 ms; this excludes CPU feature generation and transfers.

Gradient clipping applied to 883/1024 updates. All updates and saved arrays were finite. These diagnostics do not warrant changing the common optimizer for one arm.

Decision: run the complete causal transformer at width 768 with identical data draws, augmentation, AdamW schedule, update count and validation schedule. Compare both by update and measured training time, and retain phase metrics. Select one subsequent encoder intervention using that result. No encoder sweep has been launched.

This is a single-seed policy-only architecture comparison. The CNN preserves KataGo’s same-target training helper and uses the common AdamW recipe; it is not KataGo’s full production training setup. Offline KL does not establish playing strength or self-play sample efficiency.

Audited curve (external or omitted experiment artifact) · Machine-readable review (external or omitted experiment artifact) · Independent audit (external or omitted experiment artifact)
