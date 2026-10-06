# Longer paired 19×19 results

| Arm | Endpoint policy KL | Last-three policy KL | Family KL | Value MSE | Learning minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| cnn | 0.558877 | 0.578643 | 0.577843 | 0.090402 | 432.61 |
| transformer | 0.546668 | 0.551016 | 0.565472 | 0.082739 | 638.97 |

One paired supervised seed; fixed endpoint and tails, complete validation/probe curves. No test targets, playing-strength or MFU claim.

Positive relative transformer gains below mean lower loss; endpoint and tail must be interpreted together.

| Metric | Endpoint transformer gain | Last-three transformer gain |
| --- | ---: | ---: |
| expert_kl | 2.18% | 4.77% |
| family_kl | 2.14% | 4.66% |
| value_mse | 8.48% | 17.12% |
| value_family_mse | 5.01% | 13.83% |

Registered sustained-overfit observations: CNN 1; transformer 8.

The fixed endpoint is reported without selecting a best validation checkpoint. Full curves and opponent-stratified endpoints are in RESULTS_001.json.

![Full learning curves](curves-001.png)

Training and checkpoint audits completed. Two earlier disk-full monitoring errors prevented the automatic sequence summary; the completed artifacts were subsequently reconciled in sequence-recovery-001/result.json. Original failure records are retained.
