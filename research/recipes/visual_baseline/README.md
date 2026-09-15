# CNN versus causal visual transformer

This recipe adds the missing large CNN baseline. The first screen uses 128
updates from scratch on the same immutable 9x9 weak-teacher corpus. It is a
short offline comparison, not an RL scaling result or a KataGo reproduction.
The exact sources and controls are recorded in
`research/studies/visual_causal/cnn_baseline_registration.json`.

| Contract | CNN | Transformer |
|---|---|---|
| Parameters | 232,389,632 | 233,137,152 |
| Trunk | 49 residual blocks, two 3x3 convolutions each, width 512 | 18 causal decoder layers, width 1,024, GQA and SwiGLU |
| Observation input | Eight recent six-plane boards, presence flags, preceding move/pass planes | Nine learned continuous visual patches per 9x9 board |
| History | Explicit window of eight observations and their preceding actions | Complete observation/action history, up to 512 positions |
| Readout | Spatial policy logits, global pooling, separate behavior adapter | Causal readout, separate behavior adapter |
| Objectives | MCTS policy CE, observed-move CE, terminal value MSE; each weight 1 | Same |
| Auxiliary draft loss | Disabled | Disabled |
| Batch | 128 complete sequences across 16 TPU chips | Same episodes and D4 draws |

The fixed corpus comes from four earlier native-search self-play teacher runs,
with 11,871 expert episodes and 8,192 behavior episodes across all partitions.
Expert targets are stored search distributions and side-to-move terminal
outcomes from complete replay games. Behavior targets are actual moves sampled
from archived episodes, including capped games, with no outcome labels for that
role. They are not scripted opponents or KataGo training data. Splits are
80/10/10 by whole game, shared across roles; repeated positions can occur in
different games, and held-out opponents are not established.

Both arms start from random weights and use teacher-forced exact observations
and past actions. Losses are averaged over eligible nonpadding positions in
each role, then added with equal weights. Each update samples 64 expert and 64
behavior sequences globally. Both runs saw exactly 769,036 expert-position and
790,401 behavior-position exposures. The transformer computes the sequence
loss in parallel with causal masking; it does not autoregressively sample its
own training contexts or learn a future-board reconstruction objective. The
CNN evaluates the eight-state window at every corresponding position.

The shared AdamW schedule warms up for 32 updates to 1e-4, then decays by cosine
to 3e-5 at update 128, with betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01
and gradient clipping at 1.0. Whole-game D4 augmentation transforms observations,
actions and policy targets together. Score, ownership and early-exit losses are
disabled for both arms. The behavior loss updates the shared trunk in both.

CNN frames are microbatched in groups of 32 within each shard and rematerialized
across both the frame and residual-layer loops. This bounds activation memory
while retaining the complete-episode sampler and exact position weighting.
Parameters, residuals and Adam states use FP32; matrix/conv operands use BF16
with highest JAX precision. There is no Flax dependency.

Parameter count is within 0.33%; FLOPs and history are not matched. Convolutions
reuse each kernel across 81 board points, whereas the transformer reuses its
projections across visual/action tokens and attends over time. This comparison
cannot isolate architecture from history representation or establish optimal
hyperparameters for either family. The common inherited AdamW implementation
excludes rank-one arrays from decay; stacked CNN normalization arrays therefore
receive decay while unstacked transformer normalization vectors do not. A
semantic parameter mask is a declared follow-up, not a silent change to either
registered run.

The CPU tests cover target/future-action leakage, history truncation, frame
microbatch consistency, padding, finite gradients, and the final-frame inference
path. The frozen CPU service qualification also compares every native search
leaf with the full training path and exercises real GTP clients. A small
multi-host TPU learner qualification precedes the full run.

The KataGo panel design is frozen separately in `cnn_katago_design.json`: use
both final 128-update endpoints regardless of losses, 32 paired openings per
anchor, two early checkpoints, 16 candidate simulations versus one KataGo visit.
Every recorded board and terminal score must be checked independently; caps
remain unresolved. Inference and match execution require their own receipts.

`analyze_learning.py` reconstructs every episode and D4 draw, checks both arms'
full Adam checkpoints and population-weighted metrics, and reports learning
seconds separately from full reservation cost. Compiler FLOP estimates are
recorded but are not hardware MFU and can treat loop bodies differently.

The source is cloned and owned here. Future experiments should clone this
recipe and freeze a new source/config identity; do not change an existing
checkpoint's interpretation or reuse held-out outcomes to choose an endpoint.

Both 128-update runs and the learning audit passed. Test expert KL / behavior CE /
value MSE were 1.20966 / 2.04707 / 0.66170 for the CNN and 1.97943 / 2.92923 /
0.97724 for the transformer. Critical-rank learning times were 294.39 and 133.59
seconds. The matched KataGo panel also favored the CNN; read
`research/studies/visual_causal/cnn_katago_result.json` for all outcomes, caps,
missing-outcome bounds and the raw-transcript audit.
