# Fixed-student value errors are concentrated before the late endgame

The complete registered multi-host diagnostic passed. Each fixed model produced predictions for all 102,339 expert and 78,262 observed-behavior validation positions. Independent reconstruction reproduced every original validation aggregate within 0.00000165, below the registered absolute tolerance 0.0002. The attempt took 25.96 seconds and used 0.115367 recorded attempt chip-hours. It performed no training and generated no new games.

The exact-board student improves policy KL in every phase and greatly reduces raw illegal probability. Its remaining value problem is concentrated early and in the middle game: prediction zero has outcome MSE 1 on these binary outcomes, whereas the student scores 1.088 before ply 20 and 1.023 on plies 20–59. Fixed-bin calibration error is also appreciable in those ranges. After ply 120, value prediction is very accurate. High MSE alone would not establish overconfidence; the calibration bins and saturated-error counts are reported separately.

| Ply range | Expert positions | Exact policy KL | Exact value MSE | Exact value ECE | Empty value MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–19 | 23,400 | 0.8156 | 1.087522 | 0.1098 | 1.0656 |
| 20–59 | 44,932 | 1.2354 | 1.023091 | 0.1792 | 1.1919 |
| 60–119 | 24,257 | 1.0867 | 0.485000 | 0.0812 | 0.8358 |
| 120–323 | 9,750 | 0.7531 | 0.000029 | 0.0002 | 0.0497 |

In plies 20–59, 8,230 exact-student predictions have absolute value at least 0.95; 15.71% of those saturated predictions have the wrong outcome sign. In plies 120–323, all 9,739 saturated predictions have the correct sign. These are descriptive counts on correlated positions, not independent-game confidence bounds.

The 3,820 held-out expert positions after opponent passes differ from the previously inspected losing external-game tail: mean predicted pass probability is 30.88% versus a 30.45% teacher policy target, policy KL is 0.713 and value ECE is 0.0350. Across all expert validation positions, mean predicted/target pass probabilities are 4.884%/4.848%. Matching aggregate pass probability does not imply correct individual decisions or playing strength. Raw illegal probability after a pass remains 7.16%.

The resulting isolated learning intervention compared logit-based outcome cross entropy with MSE using the same scalar head, parameters, examples, policy/behavior targets and optimizer. Its multiplier matched MSE’s gradient at a neutral prediction. The completed pilot increased validation outcome MSE by 2.08% and failed its fresh 144-game external criterion. This does not establish MSE as the cause of the observed overconfidence or external losses. Keep the separate expert-representation hypothesis open; see [value_logit_distillation](../value_logit_distillation/README.md).

CPU qualification checked 136 positions per model and passed four independent metric checks, including calibrated uncertainty with MSE 1, probability-bin endpoints, raw versus legal pass normalization and observed-behavior targets. Source `c05cad6a…`, CPU audit SHA256 `0ff37e5b1f152013d6b40ec0308666342142d6df2faf4213539cf2d2e77fc9dc`.

The full registration is [tpu_spec.json](tpu_spec.json), SHA256 `e253c3aa968a4df9cf7a654a76d489b2191d9278158727120a526bd6a86c8d82`. Execution source is `6647495f05da8f1d16ec6bae55a52a5d5a4e90314994bde0349f47400f197dfb`; attempt `pod-20260911T204136Z-c80a3f76`. The independent audit (external or omitted experiment artifact) has SHA256 `241247a436445cbd52e78f9a38353dd77b6fb3fbfd83eec8e26c88f4d11c08cf`, analysis source `6164ccbda…`. Raw logits/values, exact position IDs, model/dataset identities, HLO and runtime receipts remain pinned in the audit. Standalone bfloat16 compilation was checked by aggregate tolerance, not bit-exact trajectory equivalence.

Preflight found the exact model and dataset on all hosts, while the empty control’s logical rank-0 artifact path was missing on three. The new frozen `ops/build_causal_bundle.py` creates deterministic archives of eight original referenced files; the existing materializer validates the whole model/checkpoint lineage after publishing. all configured hosts passed this check before launch. Bundle SHA256 is `57d8bea2522103b1bd54534c703e05d9c450d94da6367f2a7543cdf23fca0b43`; manifest SHA256 `34ac4821ebe05075eb62c828f378c381b51d7735748d0f9069ec9b00693c51c8`. These copies are within the pod, without external durability.

This is a prospectively stratified reanalysis of an already inspected validation split. It is not untouched-test evidence, RL sample efficiency, MFU improvement, stronger Go play or production promotion. Teacher shards and positions are correlated. No KataGo data enters these targets.
