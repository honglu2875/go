# Eight-hour Go research report — 12 September 2026

The large CNN baseline is now implemented, trained and compared with the new causal visual transformer. In the short matched-update screen, the CNN fits the held-out weak-teacher corpus better. The real KataGo panel below is a separate measurement. No strong 19x19 engine, equal-compute architecture winner, hardware MFU improvement or faster-than-KataGo RL result has been established.

## Large architecture baseline

Both pure-JAX models trained from scratch for 128 updates on identical complete-episode and D4 draws, with 128 sequences across the multi-chip/multi-host pod. Expert policy, observed behavior and value objectives were shared; auxiliary draft losses were disabled. The CNN has 232,389,632 parameters and 49 width 512 residual blocks. The transformer has 233,137,152 parameters and 18 width 1024 causal decoder layers. The CNN sees eight recent observations and preceding-action planes; the transformer sees the full causal visual/action history. This is a practical baseline with an explicit history difference.

The immutable 9x9 corpus contains 11,871 expert and 8,192 behavior episodes from four earlier native-search self-play teacher runs. The expert head learns stored MCTS distributions and the value head learns terminal outcomes; the behavior head predicts actual archived moves, including capped games without outcome supervision. These are observed engine behaviors, not scripted opponents. No KataGo data entered training. The 80/10/10 splits hold out entire games, although board positions can repeat across games and no unseen-opponent claim is made.

Each update samples 64 expert and 64 behavior sequences, with identical whole-game rotations/reflections in both arms. Both runs saw exactly 769,036 expert-position and 790,401 behavior-position exposures. AdamW uses betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01 and gradient clipping at 1.0; the learning rate warms up for 32 steps to 1e-4, then decays by cosine to 3e-5. Expert cross-entropy, observed-action cross-entropy and terminal value MSE each have weight 1. Score, ownership and draft losses are disabled.

The transformer is trained with exact observations and past actions supplied throughout the complete sequence, using causal masking and parallel losses at each move. It does not generate its training contexts or learn a future-board reconstruction loss. The CNN evaluates the corresponding eight-observation window at every position. Both behavior losses update their shared trunks.

| Held-out test/cost | CNN | Transformer |
| --- | ---: | ---: |
| Expert KL | 1.20966 | 1.97943 |
| Behavior cross-entropy | 2.04707 | 2.92923 |
| Value MSE | 0.66170 | 0.97724 |
| Critical-rank learning seconds | 294.390 | 133.594 |
| Full attempt seconds | 580.899 | 544.288 |
| Full attempt chip-hours | 2.582 | 2.419 |

All 1,024 rank-update draws and both complete parameter/Adam checkpoints passed independent audit. The CNN took 2.20x the learning time. Initialization, inductive biases and the inherited shape-based decay mask also differ; one seed and 128 updates cannot settle long-run architecture performance. Training audit (external or omitted experiment artifact), [recipe and limitations](../../recipes/visual_baseline/README.md).

Both inference owners passed full-model native search/GTP qualification before external matches: 17,408 leaf comparisons per arm and 32 real GTP probe boards per arm. Floating-point tolerance is explicit; this is not a bitwise equivalence claim. Service audit (external or omitted experiment artifact).

## Real KataGo panel

Each arm played 128 games: 32 paired-color openings against each of two fixed historical checkpoints. Candidate search used 16 simulations excluding its root; KataGo used one visit and one thread. These are explicit search budgets, not equal compute. Counts are wins/losses/unresolved move caps.

| KataGo checkpoint | CNN | Transformer |
| --- | ---: | ---: |
| `kata1-b6c96-s10014464-d2201128` | 54 / 3 / 7 | 2 / 34 / 28 |
| `kata1-b6c96-s938496-d1208807` | 64 / 0 / 0 | 25 / 17 / 22 |

Raw GTP commands/replies, SGFs, all 37,480 recorded boards and 199 completed scores passed independent audit. Caps are retained without assigned outcomes; the complete-panel strength qualification therefore fails whenever any cap remains. The observed score bounds still favor the CNN at both anchors even if all unresolved outcomes are assigned in the transformer's favor. These are sample bounds, not confidence intervals. The models faced common opponents; no direct CNN-versus-transformer games were played. No Elo or general architecture superiority is inferred. Full panel and missing-outcome bounds (external or omitted experiment artifact).

## Rollout and learning findings

- Fixed one-position scoring blocks plus continuously refilled native actor slots produced exactly the same 1,024 game traces, seven target arrays and game records as the blocked control. The generation segment fell from 716.919 to 474.205 seconds: 1.5118x. This is one paired timing result, not a hardware MFU or strength claim. Audit (external or omitted experiment artifact).
- Native suffix observation encoding passed its separate gate at 1.6501x scorer speed, with 48,323 leaf predictions and 289,938 head outputs bitwise equal. Timing scopes differ; do not multiply speedups. Audit (external or omitted experiment artifact).
- Dynamic-block refill was fast but changed 230 of 1,024 action tapes and failed the exactness gate. Complete eager-repair speculative loops were slower than sequential policy decoding: 0.726x at H2 and 0.724x at H4. D4 training improved draft agreement but did not make the independent-root speculative screen faster. Refill failure (external or omitted experiment artifact), loop (external or omitted experiment artifact), [D4 speculation](speculation_d4_result.json).
- The final deferred-repair loop also failed its timing screen: 0.596x at H2, 0.539x at H4. Its audit replayed 24,224 committed moves and 639 terminal paths; maximum reported full-policy TV was 0.005295. Pending work and final deep cache drain are timed; initial prefill and compilation are excluded. This is bounded policy sampling without MCTS, and individual acceptance probabilities/uniforms were not retained for independent replay. Audit (external or omitted experiment artifact).
- The first full-Adam online continuation did not improve its fresh KataGo panel. Reconstructed expert-data reuse was 18.66 exposures per available position for new self-play versus 1.86 for old data; this motivates a controlled replay-ratio experiment but does not explain the decline causally. Online games (external or omitted experiment artifact), reuse audit (external or omitted experiment artifact).

## Infrastructure and next work

Clone-owned model/training recipes live in research/recipes; shared Rust rules, native bindings, datasets, artifacts and checkpoints live outside those recipes. Cargo/uv locks, content-addressed source/config snapshots, rank checkpoint groups and retained failures support reproduction. Both large baseline checkpoints include full Adam moments.

Host 1 ran out of disk space while staging the transformer checkpoint. The unchanged service passed after verified lossless archival of 20 older arrays restored about 5.35 GB free. The restore/reader round trip used the user-provided /dev/shm scratch; committed checkpoints and archives remain on persistent disk. The failed staging attempt and repair receipts are preserved. Repair receipt (external or omitted experiment artifact).

The nominal eight-hour allocation (01:39:55–09:39:55 UTC) corresponds to 128 chip-hours on 16 chips. The 74 recorded, nonoverlapping pod attempts intersecting that window account for 70.427 chip-hours inside it, including startup, compilation, evaluation and checkpointing. The final registered TPU probe ended at 09:40:36 UTC, just after the nominal window; full attempt costs and subsequent allocation time remain in the ledger. It has 159 closed attempts and no observed open attempts at publication, and also covers work before this window. Full teacher/data-generation costs matter when comparing learning systems. Ledger (external or omitted experiment artifact).

The final user-requested CNN comparison took priority over optional pending work. The separately registered collection-holdout model diagnostic and second fixed-block refill timing pair remain unrun. They are not negative results.

Keep the CNN as a serious control. Next test spatial policy readouts and small CNN observation encoders inside the causal transformer, a matched-history control, longer common schedules, and equal-compute learning curves. Retain the distinct expert/behavior objectives; their log-density ratio is not automatically a value advantage. Production training still needs external checkpoint durability, host-loss/mid-update recovery and a sustained generation/learner coordinator. [Detailed design](NEXT.md).
