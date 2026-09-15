# Logit cross entropy failed the prediction and external criteria

The fixed-budget training comparison and all 144 prescheduled real-KataGo games completed. Independent training and game-replay audits passed. BCE2 increased held-out outcome MSE by 2.08%, missing the registered requirement of at least a 10% reduction, and failed the external criterion. The MSE control reproduced the previous exact-board student's exported weights byte for byte. Retain the MSE control; this pilot does not justify replication or promotion under its registered rule.

Both complete plain-JAX trainers own their model, optimizer, sampler and recovery code in [the cloneable recipe](../../recipes/value_logit_distillation/README.md). The only scientific configuration difference is `model.value_objective`. For raw logit r, value v = tanh(r), and terminal outcome z, MSE uses (v−z)² and BCE2 uses 2[softplus(2r)−(z+1)r]. Their gradients agree at r=0. BCE2 retains gradient correction for confidently wrong saturated values. Inference and the 939,968-parameter tree are unchanged. Validation `value_loss` is outcome MSE in both arms; total training losses are not comparable.

The motivation was the [fixed-student calibration diagnostic](../student_calibration/README.md), which found early/middle-game overconfidence alongside accurate late-game values. This tested an optimization hypothesis; it did not establish that MSE caused the external losses. KataGo already uses value cross entropy and binary logit losses: see its [pinned loss implementation](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/python/katago/train/metrics_pytorch.py). This scalar-head comparison is neither a novel cross-entropy method nor the full KataGo loss stack.

| Fixed final validation metric | MSE | BCE2 | Registered condition |
| --- | ---: | ---: | --- |
| Outcome MSE | 0.812813 | 0.829740 | Candidate/control ≤ 0.90: failed |
| Expert policy KL | 1.058237 | 1.079930 | Ratio ≤ 1.05: passed |
| Raw illegal probability | 0.0146954 | 0.0147305 | Increase ≤ 0.002: passed |
| Observed-behavior NLL | 2.039736 | 2.051890 | Increase ≤ 0.05: passed |

Each arm used seed 41, 4,096 updates, identical initial parameters and exactly the same sampled examples: 23,140,715 expert and 25,592,798 behavior token exposures. These are replicated global counters, not values to sum over four ranks. Both use the same 16-shard internal-teacher dataset, original split, bfloat16 model, optimizer and schedule. The final validation includes 102,339 expert and 78,262 behavior positions. The analysis checked all four final checkpoints, group manifests, exported parameters, sampler states, initialization and work counters, and enforced registration time, execution order and one-attempt limits.

The two full pod attempts took 104.008 and 104.053 seconds, using 0.924714 recorded attempt chip-hours combined. Similar durations are not evidence of a systems speedup or MFU. Teacher generation, qualification, engineering and idle allocation time remain separately accounted for. Direct behavior gradients are blocked from the expert; the shared clipping norm and optimizer convention still couple update scaling, so complete optimizer isolation is not claimed.

The [pilot registration](pilot_spec.json) has SHA256 `94ae20910be7ca0202a467665e39d3040bafff89d92c78d72d834a21c4bb9f2c`. It fixes both final checkpoints before training, requires all prediction conditions, and schedules 144 fresh KataGo games regardless of the prediction result. The historical primary comparison uses 32 paired openings and both colors per arm. It requires at least 95% completion in **each** arm and a strictly positive paired-bootstrap outer lower endpoint for BCE2 minus MSE. Caps remain unassigned [0,1]. Each arm also has eight games against the pinned stronger 9x9 anchor. These are historical/specialty checkpoints, not current best KataGo.

| External anchor | MSE wins / losses / caps | BCE2 wins / losses / caps |
| --- | ---: | ---: |
| Historical b6c96, one visit | 4 / 55 / 5 | 5 / 57 / 2 |
| Stronger b18c384, one visit | 0 / 4 / 0 | 0 / 4 / 0 |
| Stronger b18c384, 16 visits | 0 / 3 / 1 | 0 / 3 / 1 |

For the historical primary panel, scheduled score bounds are [6.25%, 14.06%] for MSE and [7.81%, 10.94%] for BCE2. The difference bounds are [−6.25, +4.69] percentage points. The prespecified 20,000-draw paired-opening bootstrap gives an outer 95% interval of [−14.10, +10.94] points. Its lower endpoint is not positive, and the MSE completion fraction is 59/64 = 92.19%, below the 95% requirement. BCE2 completes 62/64 = 96.88%. Both external conditions therefore fail. These intervals are descriptive and conditional on the single trained pair; they are not an Elo estimate or independent-seed uncertainty.

The independent audit replayed all 14,757 boards through the qualified Rust rules and checked all 135 completed scores against the actual KataGo transcripts. It verified loaded model/native/KataGo identities, per-game seeds/configuration, search work, paired colors and opening coverage. There were zero process failures and nine unresolved caps. Four independent CPU groups ran within each arm; arms ran sequentially and took 531.58 and 517.53 seconds. Fresh ordered openings were excluded from existing evaluation configurations, without claiming they or symmetry-equivalent positions were absent from training.

The external audit (external or omitted experiment artifact) has SHA256 `9d5145bdb1bae36dd49e1c131baf08168d1819493e356f6717b5e1a8bf9ca8cd`. Execution sources are `5a3f8ea470b44a69c897a2893feb8a7b3096a7355b94341905ec5f14cd1d54ca` (MSE) and `7a4b282afc94a7a28d95e4f9c527da4cf27593601a5452c32b1b7d8ff6f3266a` (BCE2 and analysis). Raw suites remain under `runs/eval/value-logit-mse-5a3f8ea4` and `runs/eval/value-logit-bce2-7a4b282a`. The artifact index (external or omitted experiment artifact) covers the closed runs, retained qualification failures, protocols, audits and frozen identities.

Training sources are `b211cc7d07baa29bde30c9e67ee30ba7c92daa5f37b91a6ebc8692218dbb204e` (MSE) and `d7820d61a93dc2c3e8a994b037447fbd8762fec4459123b7339828ddb2ea0baa` (BCE2). Attempts are `pod-20260911T210435Z-5f48d3ca` and `pod-20260911T210807Z-14850206`. The independent training audit (external or omitted experiment artifact) has SHA256 `39273fe16bb3485a0ef83b39577b8cb2e8fa93b3232b1e4ed7c0d450060568df`, analysis source `7a4b282afc94a7a28d95e4f9c527da4cf27593601a5452c32b1b7d8ff6f3266a`.

Qualification passed ten CPU model/loss tests, exact CPU continuation of 111 arrays and sampler/final metrics, and multi-host continuation of all 804 saved arrays with identical sampler states and final metrics. The first CPU attempt failed because a regression test referred to a sibling recipe excluded by snapshotting; it launched no training. The fix copied the unchanged, SHA-pinned reference model into the clone. An earlier incorrect caller-supplied protocol hash was rejected before any child invocation and is also retained. Both are engineering failures, not discarded scientific runs. CPU qualification (external or omitted experiment artifact) and TPU qualification (external or omitted experiment artifact) precede the full pilot; TPU recovery used another 0.337346 attempt chip-hours.

Three independent paired-score tests passed, including caps in either arm, assigned-cap rejection and missing-color coverage. They do not replace replaying the actual external games.

This is single-seed offline distillation using a previously examined validation split. The teacher mixture remains in the lineage and contains no external KataGo labels. No RL sample-efficiency, stronger 19x19 play, current-KataGo superiority or production promotion follows. The [next architecture design](NEXT.md) separates state-expert representation from behavior-model ownership and gives each change its own comparison.
