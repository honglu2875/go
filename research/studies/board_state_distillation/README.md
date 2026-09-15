# Explicit board inputs for the causal student

The registered input ablation improved prediction substantially but failed its external-playing criterion. It does not justify another independent training seed or production promotion under this protocol. The exact-state model remains useful for studying execution semantics; that is a separate qualification, not stronger-play evidence.

Both arms started with identical parameter elements and the same 939,968-parameter architecture. They used the same seed 41, 4,096 AdamW updates, sampled games, target masks and token exposure: 23,140,715 expert tokens and 25,592,798 observed-behavior tokens. These are replicated global counters, not per-host quantities to sum. Only `model.board_mode` changed. The empty arm supplies zero board codes; the exact arm supplies authoritative pre-action stones. Both retain the same full action history.

| Fixed final validation metric | Empty boards | Exact boards |
|---|---:|---:|
| MCTS expert-policy KL | 2.493137 | 1.058237 |
| Raw expert probability on illegal actions | 0.373641 | 0.014695 |
| Observed-behavior NLL | 3.573029 | 2.039736 |
| Value MSE | 0.969810 | 0.812813 |

All three registered prediction conditions passed: expert KL ratio 0.42446 ≤ 0.8, illegal-probability ratio 0.03933 ≤ 0.5, and behavior NLL change −1.53329 ≤ 0.02. The checkpoint was fixed in advance; validation and test splits are inherited from previously inspected data.

| Real KataGo panel | Empty boards | Exact boards |
|---|---:|---:|
| Historical b6c96, one visit, 64 games | 0 wins / 64 losses | 1 win / 56 losses / 7 caps |
| Official 9×9 b18c384, one visit, 4 games | 0 wins / 4 losses | 0 wins / 4 losses |
| Official 9×9 b18c384, 16 visits, 4 games | 0 wins / 4 losses | 0 wins / 4 losses |

The historical panel uses 32 fresh two-stone openings with both colors. Caps remain unassigned [0,1]. Exact-minus-empty scheduled score is bounded by [0.015625, 0.125]; the registered whole-opening paired bootstrap gives a descriptive 95% outer interval [0, 0.234375]. Its lower endpoint is not positive, and exact-board completion is 57/64 = 89.06%, below 95%. Both external conditions therefore failed. This interval is conditional on one trained model pair and these openings; it does not represent independent-training-seed uncertainty or Elo.

All 144 scheduled games are retained. Every one of 15,374 board comparisons and 137 completed score comparisons agreed with pinned real KataGo; seven games capped and no process/integrity failures occurred. Each candidate used native Gumbel search with 16 simulations excluding the root, its expert policy/value heads, no search noise, and the frozen full-history inference contract. Behavior predictions never selected MCTS moves. The native pass-alive area scorer and positional-superko rules match the evaluator's declared contract.

Post hoc endgame inspection found 873 exact-model plays following an opponent pass; 744 had root value ≤ −0.95. In five of seven capped games, the opponent passed on all 16 turns in the final 32 plies, the candidate never passed, and its root values were close to −1. The other two caps had no opponent passes in that window. These observations motivate checking pass decisions, terminal backups and outcome calibration. They do not isolate a causal explanation, authorize relabeling caps, or replace the failed strength criterion.

## Implementation and qualification

The complete trainer is owned by `research/recipes/board_state_distillation`. Its canonical history transformer depends only on prior actions. A small per-position convolutional board encoder supplies expert policy/value corrections. The separate observed-behavior head receives stopped gradients from both encoders; its loss cannot update the expert trunk, board encoder or expert projections. Five model tests cover future/padding/game isolation, gradient separation, explicit-state sensitivity and full-batch versus single-leaf arithmetic.

`ops/prepare_board_sequence_data.py` uses one bulk GIL-free Rust replay API per source shard. It verifies all 1,047,681 expert legal masks and terminal value labels, and all 8,192 observed-game terminal/cap flags, including 25 capped games without value targets. The overlay adds 798,969 behavior pre-action states while preserving the existing parent arrays and episode splits. It is staged and hash-verified on all four TPU hosts. The loader qualification compared all population indices and 144 original fields across 16 sampled batches; the RNG states and all non-board inputs matched exactly.

CPU recovery matched all 111 optimizer/parameter arrays and sampler/metric state across a checkpoint-4 restart. multi-host TPU recovery matched all 804 arrays and scientific state across a checkpoint-8 restart. The actual GTP adapter passed nine 16-simulation moves on three verified KataGo fixture prefixes before pilot registration. Native replay was also compared against the independent per-move binding on a full fixture; 18 native binding tests passed.

Exact states are supplied at every evaluated position. A future speculative decoder must validate the board inputs associated with every consumed own-policy prediction, in addition to matching action prefixes and model/context tickets. Reusing the root board through a capture would change this model's policy. No speculative board generation, multi-step MCTS equivalence, MFU improvement or end-to-end speedup is claimed here.

## Immutable evidence

- [Pilot registration](pilot_spec.json), SHA-256 `ee1e35a16ea0830034101a5203df2c338b4429b890716405401276c8c98f9cbd`. Registered before either training arm; includes recursive evaluator inputs, evaluator code, qualifications, budgets and ordering. Only the two future model descriptors were deferred.
- Training audit (external or omitted experiment artifact), SHA-256 `aeedc9cc6dd68d5e4a579d07d659bb72c30dac968fd34cd5a339bab6dc24e8fe`.
- Final match audit (external or omitted experiment artifact), SHA-256 `7dcbf47e914e73c63a2c49a38dc696219b07aa5d28f40a213bb390bab92cbece`. Includes loaded engine identities, every raw file hash, per-opening statistics, search counts and endgame diagnostics. `status: passed` means the audit passed; `registered_combined_criterion_met: false` records the scientific outcome.
- CPU recovery (external or omitted experiment artifact), TPU recovery (external or omitted experiment artifact), batch qualification (external or omitted experiment artifact), and [TPU qualification registration](tpu_qualification_spec.json).
- Dataset `.gozero/datasets/board-causal-13bcd5cd`, manifest `6c935aa44ec03893878f43f2456e01be96240b478489547235175441cb58169e`; parent `.gozero/datasets/causal-5c5f2c0d`, manifest `9a4ef7a1034897f0991e197b5c92f5303d777a5cfb8a23bfa6dcbbd3b7c8f0b6`.
- Empty training source `dc09adfcaaf086f266bf21b3611c2eb7f498f8684e85ec2f6590c5e2e42b5180`, attempt `pod-20260911T162016Z-3f43c06f`, model `80429ac0d7967b90c0f7c934cd7673e9ccdb0c370ee7e0362bc2d00ab43a96be`.
- Exact training source `e2318f977ff2b60ef4c78c841d25ad153f6fe0fcbf78a0934f57afae872fd02a`, attempt `pod-20260911T162449Z-016a59f3`, model `c865322774cf333073f05413a41b645116adc788a916c6c40b6855a6b28b7e13`.
- Raw suites `runs/eval/board-state-{empty,exact}-ee1e35a1`; execution sources `5023538fcc8aace803792bf5b6d085597be7c32e5872800afa1b01ecb5ed87b7` and `5a0de21c8bca407dcc11ab56877f3d3f668f01b6ef0cf1b9158457bbf70cdfb5`. Analysis source `2809492aeaf15562b33c7e0a5e17791b0aff666744369dfcfdc447ba25bbc996`.

Reproduce the audit from its frozen operator (choose a fresh output path):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 100-107 .venv/bin/python -B \
  .gozero/snapshots/2809492aeaf15562b33c7e0a5e17791b0aff666744369dfcfdc447ba25bbc996/research/recipes/board_state_distillation/analyze_external.py \
  --workspace-root /workspace/go \
  --empty-suite runs/eval/board-state-empty-ee1e35a1 \
  --exact-suite runs/eval/board-state-exact-ee1e35a1 \
  --output /tmp/board-state-audit.json
```

The training attempts took 101.390 and 103.267 seconds, totaling 0.909584 recorded attempt chip-hours. TPU recovery qualification adds 0.329908 chip-hours, for 1.239493 new attempt chip-hours. Reused teacher lineage retains its 67,108,864 generated moves and 25.761545 recorded chip-hours. CPU evaluation took 425.10 and 509.24 wall seconds on four disjoint worker groups; these are not a systems speed comparison. The reservation ledger separately includes engineering and idle allocation time. No current-KataGo superiority, RL sample-efficiency gain, 19×19 transfer or production readiness follows from this pilot.
