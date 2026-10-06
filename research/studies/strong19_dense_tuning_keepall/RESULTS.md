# Dense LR and batch tuning

Settled study peak LR: 0.001. Production defaults unchanged.

| Global games | Updates | Policy KL | Family KL | Value MSE | Learning hours |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 256 | 0.728001 | 0.763186 | 0.121956 | 2.437 |
| 128 | 128 | 0.842229 | 0.879755 | 0.129620 | 2.362 |
| 256 | 64 | 1.024584 | 1.068218 | 0.169125 | 2.431 |

Quality candidate: batch64. Throughput candidate: control.

Single-seed bounded fixed-data tuning. Later LR check at 256 updates; batch comparisons at equal 7,001,181 position exposures. No strength, convergence or MFU claim.

See each stage audit for every validation/probe checkpoint and overfit observation. Existing checkpoints are preserved. New trial states have RAM replicas; selected new endpoints have two verified disk copies.
