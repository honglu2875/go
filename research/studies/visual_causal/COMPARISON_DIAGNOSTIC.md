# Understanding the CNN–transformer comparison

The immediate research window is closed. Training, inference qualification, KataGo transcript/board/score audits and the final deferred-repair probe are complete. The collection-holdout diagnostic and second fixed-block refill timing pair are deferred. Capped evaluation games retain their unresolved outcomes. This document analyzes existing artifacts and launches no TPU work.

## What the saved learning curves show

| Update | CNN expert KL | Transformer expert KL | CNN behavior CE | Transformer behavior CE | CNN value MSE | Transformer value MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.7065 | 2.8765 | 3.5005 | 3.4956 | 1.0012 | 1.0072 |
| 64 | 1.5769 | 2.4271 | 2.3733 | 3.2533 | 0.7795 | 0.9997 |
| 128 | 1.2207 | 1.9924 | 2.0476 | 2.9590 | 0.6783 | 0.9795 |

The CNN already leads all three validation metrics at update 64, including when compared with the transformer at update 128. The transformer improves between 64 and 128, so these measurements do not establish a plateau. Its value MSE changes little from initialization, while the CNN learns much more of the terminal-outcome target. Both the representation and optimization deserve investigation before attributing this to architecture alone.

Both arms accepted all 128 updates. CNN gradients exceeded the global clip threshold on 128/128 updates; transformer gradients did so on 127/128. These are finite, accepted updates, and Adam prevents a direct interpretation of the clipping factor as an effective learning-rate reduction. Per-head gradients and update-to-parameter norms were not saved.

The normalization decay-mask discrepancy is real. Under this short schedule, decay alone would shrink a decayed parameter by 0.00785%. This arithmetic gives its direct scale, not a bound on its indirect effect through learning. Fix the semantic mask in a newly frozen control.

## Next comparison, one question at a time

1. Walk through the exact inputs, token ordering, current-board visibility, policy readouts and gradient paths. Keep the expert, behavior and value roles explicit. Test small-fixture overfitting and one-device/distributed gradient agreement if the code review exposes an unresolved concern.
2. Measure a longer common schedule with predefined update and learning-time checkpoints, matched data draws and additional seeds. Record per-head gradients, logit/value distributions and update-to-parameter norms. Use validation for diagnosis; retain fresh games for the next external comparison.
3. Isolate representation choices: first match the eight-board history; separately test a spatial policy readout; separately test a small CNN observation encoder feeding the causal transformer. Combining these immediately would obscure which intervention helped.

Before another substantial training run, reserve persistent checkpoint space and use /dev/shm for disposable staging or restored working copies. The current persistent filesystem has less than 1 GB free. Preserved checkpoints and archives must remain available; RAM scratch is not the durable copy.

[Audited eight-hour report](REPORT.md), raw diagnostic (external or omitted experiment artifact), [recipe and limitations](../../recipes/visual_baseline/README.md).
