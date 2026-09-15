# Expert spatial state and fixed opponent history

This study compares a new expert-only history control with a spatial attention expert. Both use the same expert-policy cross entropy, terminal-outcome MSE, verified internal-teacher examples, AdamW schedule and clipping convention. The observed-opponent predictor remains the separately pinned original exact-board checkpoint, outside the trainable parameter tree. Its full lineage is validated before and after training and when loading an expert for evaluation.

The control has 894,208 expert parameters; the spatial model has 909,174. The spatial model uses exact native stone/legality/pass/turn/komi features, global attention with D4-tied relative biases, and shared point, pass and value heads. This compares complete representations; it does not isolate every architectural ingredient. Removing behavior gradients also removes their contribution to gradient clipping. The old jointly trained MSE model is therefore not the matched control.

The implementation is owned by `research/recipes/state_expert_distillation`. Training is plain JAX, independent of Flax. The native engine retains full history for positional superko; the neural feature planes are not a claim of complete Markov state. The spatial parameter tree supports declared board-size buckets, with 3x3/9x9/19x19 arithmetic checks. All learning and competitive evaluation in this study are 9x9.

## Qualifications

- The registered CPU run passed ten model/native feature tests. These include expert-control prediction/gradient parity, an independent NumPy attention reference, D4 equivariance, padding and role isolation, requested-leaf parity and actual native capture/ko/pass replay. Eight-update continuation from checkpoint four matched all 87 history and 60 spatial saved arrays, samplers and final metrics. See `cpu_spec.json` and `cpu_result.json`.
- The first GTP qualification failed because it treated KataGo's `play` acknowledgment as strict legality. The native engine rejected the ko recapture; KataGo deliberately accepts tolerant moves through this interface. The failed attempt remains at `runs/qualification/state-expert-gtp-b64444d7`. The amended qualification compares the strict `kata-raw-nn 0` legality mask and checks rejection through the native adapter. Both experts passed, with 296 board comparisons, 656 strict mask entries, eight searched positions, two complete scoring-fixture replays and four clean process exits. Three parser tests and independent transcript/identity audit passed. See `gtp_spec.json`, `gtp_retry_1_spec.json` and `gtp_result.json`. This qualification played no new competitive games.
- The four registered TPU attempts passed on all configured hosts and 16 chips. Continuation from update eight to sixteen matched all 708 history and 912 spatial parameter/Adam arrays, exact sampler states and final metrics. Both consumed 90,288 global expert-token exposures. The attempts used 0.665570 chip-hours. This is fresh-process continuation on the same disks/topology, without a host-loss or external-durability claim. See `tpu_qualification_spec.json` and `tpu_qualification_result.json`.
- Full-model CPU compiler estimates favor spatial leaf evaluation, but the short TPU qualification measured about 4.89 times more learner time for spatial training. The spatial trainer evaluates every board independently; the history trainer shares sequence computation. These are distinct workloads. Neither compiler estimates nor the qualification timings establish MFU or a replicated systems speedup.

KataGo's behavior is defined in the pinned [search implementation](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/cpp/search/search.cpp), [board history implementation](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/cpp/game/boardhistory.cpp) and [raw neural output interface](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/docs/GTP_Extensions.md). Existing competitive audits independently replay every move through the strict native engine; they do not rely on a `play` acknowledgment alone.

## Registered pilot

`pilot_spec.json` fixes one 4,096-update training attempt per arm, seed 41, the final checkpoint and a 600-second training limit per arm. The multi-host qualification establishes the expected initial TPU parameter identities. The scientific/model code remains identical to the CPU-qualified implementation. The three paired external-criterion tests pass, including caps in either arm, missing colors and assigned-cap rejection.

The prediction screen requires spatial/control validation outcome MSE at most 0.90, policy KL ratio at most 1.05, and illegal policy-mass increase at most 0.002. Validation/test splits have already been inspected in previous studies; these are diagnostics on reused data.

Both fixed checkpoints receive 64 games against pinned historical b6c96 KataGo at one visit and eight against pinned b18c384 at one/16 visits. Thirty-four new exact opening pairs and their reversals exclude earlier match specifications; D4 equivalents and board distributions may overlap. Every move, board, completed score, process identity and search budget is independently audited. Caps remain unassigned score bounds. The paired-opening outer interval must have a positive state-minus-history lower endpoint, and both arms must complete at least 95% of their historical-anchor games. The external screen proceeds even if the prediction screen fails.

## Completed result

Both scientific screens failed. The integrity audits passed; the spatial model is not promoted or selected for a larger training run. All process handles are closed.

| Measure | History control | Spatial expert |
|---|---:|---:|
| Expert parameters | 894,208 | 909,174 |
| Updates / global expert exposures | 4,096 / 23,164,695 | 4,096 / 23,164,695 |
| Validation outcome MSE | 0.799460 | 0.636780 |
| Validation expert-policy KL | 1.057274 | 1.315905 |
| Validation illegal policy mass | 0.0142520 | 0.0001995 |
| Maximum measured learner time | 41.81 s | 229.07 s |
| Complete pod-attempt time | 86.04 s | 269.38 s |
| Historical KataGo: wins / losses / caps | 4 / 55 / 5 | 2 / 46 / 16 |
| Stronger KataGo: wins / losses / caps | 0 / 8 / 0 | 0 / 4 / 4 |

Spatial validation value MSE improved by 20.35%, but expert-policy KL increased by 24.46%, exceeding the allowed 5% regression. Test-split MSE also improved (0.807265 to 0.621531), while test policy KL worsened (1.044424 to 1.298251). Neither result overrides the registered screen. Every per-update exposure count and final sampler state matched between architectures, and all four replicas agreed on their committed parameter arrays. The separately pinned observer remained unchanged. See `training_result.json` and `registration_result.json`.

Historical completion was 59/64 (92.19%) for history and 48/64 (75%) for spatial, both below 95%. With all caps unassigned, scheduled score bounds are [0.0625, 0.140625] and [0.03125, 0.28125]. The paired state-minus-history outer interval is [-0.203125, 0.343750], so it does not establish an improvement. These bounds are conditional on one trained pair, without independent-seed or Elo uncertainty.

Independent audit checked all **20,484 played boards and 119 completed scores** across 144 games. There were 25 unresolved caps and zero process failures. Actual loaded model/helper/native identities, fixed observer lineage, search budgets, external weights/configurations and every retained GTP transcript passed. The two suites took 464.88 and 495.58 seconds on the same CPU allocations; their different trajectories do not constitute a matched-work inference benchmark. See `result.json`.

The endgame descriptions reinforce the need for a targeted explanation. In historical-anchor games, the spatial expert passed after an opponent pass only 4 times in 1,953 opportunities, versus 22 in 842 for history. These are unequal visited-state populations, not a causal comparison or a new success criterion. Lower aggregate value MSE and lower illegal policy mass did not yield a completed-game strength improvement.

The four TPU qualification attempts plus two pilot-training attempts used **2.245239 recorded chip-hours**, including 1.579669 for pilot training. Teacher and observer pretraining costs remain part of lineage; the reservation ledger separately includes engineering, evaluation and idle allocation time. This was offline distillation from internal teachers, not new RL self-play, matched-compute training or evidence of superiority over current KataGo. The immutable JSON receipts, raw data, descriptors and source manifests are indexed by `artifacts.json`. See `NEXT.md` for the resulting decision and the distinct paired-decoding design.
