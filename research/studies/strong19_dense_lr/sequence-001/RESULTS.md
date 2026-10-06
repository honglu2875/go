# Dense learning-rate screen

All arms share 128 updates and 7,001,181 position exposures.

| Peak LR | Policy KL | Family KL | Value MSE | Top-1 | Learning hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6.0e-04 | 0.875002 | 0.913791 | 0.153700 | 0.4440 | 2.363 |
| 1.0e-03 | 0.842229 | 0.879755 | 0.129620 | 0.4808 | 2.362 |
| 1.5e-03 | 0.841816 | 0.879565 | 0.128236 | 0.4760 | 2.362 |

Candidate for possible longer confirmation: control. Production settings are unchanged.

LR scale only, one matched seed and a 128-update prefix. Historical control timing. Selection is for possible longer confirmation, not convergence, production promotion or tuned MoE comparison.

See comparison.json for every selection gate, last-three means, clipping diagnostics and overfit observations. All initial weights, logical-rank draws, LR schedule ratios, saved states and disk peer copies were audited.
