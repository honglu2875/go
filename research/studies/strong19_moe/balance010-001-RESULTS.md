# Sparse/dense learning screen

Completed 64 updates and 3,596,975 matched position exposures per model.

| Model | Parameters | Validation policy KL | Validation value MSE | Learning hours |
|---|---:|---:|---:|---:|
| dense | 232,011,540 | 1.133431 | 0.169567 | 1.209 |
| moe | 420,991,764 | 1.086170 | 0.195272 | 2.715 |

Policy KL change: -4.17%. Value MSE change: +15.16% (lower is better).
Learning time ratio: 2.245x. Logical active matrix FLOP ratio: 1.000637x.

All updates, logical-rank data/augmentation replay, full model/AdamW state, and disk peer replicas passed their audits. Router load, train/validation curves and historical time comparison are in the accompanying JSON.

One paired seed; warmup-length short screen, not a convergence or generality result.
Equal logical active matrix FLOPs excludes physical padding, routing, sorting, optimizer and communication work.
No KataGo match evaluation or test-target access in this architecture screen.
