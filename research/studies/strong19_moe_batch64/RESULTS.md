# Tuned-default dense and temporal MoE comparison

All arms: LR 1e-3, batch 64 games, 256 updates, 7,001,181 matched position exposures.

| Arm | Parameters | Policy KL | Value MSE | Learning hours |
| --- | ---: | ---: | ---: | ---: |
| dense | 232,011,540 | 0.728001 | 0.121956 | 2.437 |
| temporal | 317,001,492 | 0.743047 | 0.124549 | 2.444 |
| balance_low | 317,001,492 | 0.733949 | 0.120592 | 2.444 |

Provisional winner: dense. Retained MoE checkpoint: balance_low.

One-seed fixed-data 256-update prefix of a 1024-update exposure-based schedule. Dense timing is historical; learning time excludes compile/evaluation/loading. No playing-strength, MFU or convergence claim.
