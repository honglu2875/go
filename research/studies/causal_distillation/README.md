The first trained causal expert/behavior recipe completed its registered feasibility pilot. It is a supervised preparation stage for learned multi-step execution, with no Go-strength or RL sample-efficiency claim.

| Metric | Initialization validation | Final validation | Final held-out test |
| --- | ---: | ---: | ---: |
| Expert policy KL | 3.5783 | 2.5095 | 2.4967 |
| Observed-move NLL | 4.4091 | 3.5990 | 3.5879 |
| Expert top-action agreement | 1.44% | 21.27% | 21.55% |
| Behavior top-action accuracy | 1.28% | 14.91% | 14.99% |
| Expert probability on illegal actions | 48.27% | 37.31% | 37.28% |

The fixed final checkpoint meets the registered20% expert-KL and10% behavior-NLL reduction thresholds. The remaining37% illegal probability and value MSE near0.91 make this a weak Go policy. No checkpoint was selected from the intermediate validation curve. The test split was inspected during short implementation qualifications, so future confirmatory claims require fresh games.

Training consumed23,140,715 expert-token and25,592,798 behavior-token exposures in4,096 global updates. These counters are already global; summing them over the four replicas would quadruple the work. The attempt took64.045 seconds / 0.284643 chip-hours. The reused teachers cost67,108,864 real moves and25.761545 recorded attempt chip-hours. Engineering/idle time appears separately in the reservation ledger.

The frozen dataset manifest is `9a4ef7a1034897f0991e197b5c92f5303d777a5cfb8a23bfa6dcbbd3b7c8f0b6`, under `.gozero/datasets/causal-5c5f2c0d`. Its operator is `89dd2d46670d4224c97f969d782f49469aeb1b0f2f2ff886d37d301f7c742060`. It contains1,047,681 MCTS rows from11,871 complete games, plus798,969 observed moves from8,192 separately sampled games. Twenty-five capped behavior games have no outcome labels. Action and model-version alignment are verified against archived game bytes; only an overwritten first replay suffix may be omitted.

CPU recovery matched60 arrays, sampler state and final metrics. multi-host TPU recovery matched528 arrays and the same scientific state; both attempts cost0.216409 chip-hours. Causality, population masks and stopped-gradient tests passed. The qualification source is `23628ac3255959d6a37003ca86ec272b4377831a36f79f64e850396f95f99ab0`.

Pilot source: `2be69574f7f8e2db9f5e6eb4bf3f9716870fb0c934e3d84f2f93a2fc5b78a528`. Attempt: `pod-20260911T142624Z-5d9b5429`. Model: `e42ace9667ffb58aeb5b3d57cea97a0c05838869267a3c49abcf7ee7e5c7fb70`. `pilot_spec.json`, `pilot_result.json`, and the CPU/TPU qualification files are read-only. Analysis source: `74f7fbb5cec39d8b56ca8fa6c051ac0e88c11df179e127f1eef3918474ab2ce7`.

Follow-up work: evaluate the learned behavior head against play-head and history-free prediction controls; load the actual student into paired causal decoding; audit sequential equivalence and numerical changes; add explicit state information or state auxiliary targets; then integrate causal evaluations with MCTS and benchmark real games against KataGo. Automatic opponent-level calibration and production suitability remain unestablished.

The subsequent registered head comparison did not reach its0.1-nat benefit threshold. On779 held-out behavior episodes /76,740 moves, behavior NLL was3.5880 versus3.6216 for the expert head. The paired episode-bootstrap difference was−0.03356 nats,95% interval[−0.04003,−0.02717]. Zeroing the explicit action-count input increased behavior NLL by only0.00304 nats; the full causal history remained in the trunk. A training-only unconditional action-frequency model had NLL4.3701. These are post-pilot diagnostics on existing episodes, not fresh-opponent confirmation or an automatic level-calibration result. Raw per-episode sums are retained under `runs/analysis/causal-head-controls-4427e020`; the read-only protocol/result are `head_controls_spec.json` and `head_controls_result.json`.
