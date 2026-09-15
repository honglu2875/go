The first scientific 19×19 comparison will use the fixed cohort selected from
inventory 005: 2,634 training games (1,091,214 positions) and 146 validation games
(60,284 positions). All eight opponent strata are represented. The ten
validation games against the teacher itself give only a coarse estimate for
that stratum. Collection continues separately; no later games enter this view.
The test targets remain closed.

The packed view has passed native board/legal replay, source-byte checks,
lossless feature/target serialization and the existing corpus loader. No
complete canonical trajectory crosses training and validation. Its manifest is
`2d39f7647a463e2e718844b19d45c35052d88f9ab9d8b2b222f5cd6c1f791def`.
The two runtime buckets are 512 and 768 positions, covering every complete
history, whose maximum length is 698. The 1,536-position execution fixture
tests a larger shape; it is unnecessary padding for this fixed cohort.

Use three arms to separate architecture from optimizer transfer:

| Arm | Architecture | Optimizer |
| --- | --- | --- |
| Source CNN | Joint KataGo b40c768nbt/FSON | Qualified source Muon/AuxAdam, adaptive decay and Lookahead |
| CNN control | Identical joint CNN | AdamW |
| Transformer | One board token, shared encoder and contextual spatial readout; confirmed scale 0.01 | Same AdamW settings as CNN control |

Both architectures retain width 768 and the previously counted joint
policy/value heads: 233,220,870 CNN and 232,011,540 transformer training
parameters. Complete deployed decoding, including encoder/readout/value work,
remains within one percent at the declared batch and past-history reference.
Training work is measured separately: equal inference FLOPs do not imply equal
training FLOPs. The CNN's intermediate helper and transformer's first-pass
auxiliary supervision retain their existing coefficients, 0.8 and 0.25.

All arms use the same seed, whole-game/D4 draws and targets, 128 global games
per update, full validation and a fixed 128-game training probe. Bucket
probabilities match the training game counts (2,321 short and 313 long).
After the two declared shape-warmup draws, each game has equal marginal draw
probability. Policy targets are the raw teacher distributions; values are
signed player-to-move expectations with MSE weight 0.7. This objective adapts
KataGo's richer original training labels: ownership, score and its other
auxiliary targets are absent.

The first horizon is 108 accepted updates, with evaluation every nine
updates and at the endpoint. The predeclared seed replay contains 5,787,025
position exposures, or 5.3033 passes over this training population. This pilot is intended to
establish joint learnability and guide the next horizon, not train a strong
19×19 engine to completion. Do not copy the 4,096-update 9×9 horizon into this
much smaller population.

For AdamW, use peak LR 0.001, final LR 0.0003, 40 warmup updates, betas 0.9/0.95,
epsilon 1e-8, decay 0.01 and gradient clipping 1. The peak and floor follow the
larger9 control; the shortened warmup covers approximately two million actual
positions in this pilot. A higher peak failed the paired larger9 screen.

For the source-derived arm, set the fixed position-batch reference to 53,028,
the rounded expectation for 128 uniformly sampled training games. Start its
sample clock at zero and advance it by actual live training positions. Retain
source effective LR scale 1, source group factors, the two-million-position
stepped warmup, Lookahead period six and interpolation 0.5. Six epochs of 18
updates approximate the source's million-position epoch and align each flush
with a Lookahead synchronization. Observe pre-update norms every update;
the source's 100-update print cadence would never run within these short
complete-game epochs. These batch/cadence/epoch conventions are explicit
adaptations, not claims about a historical checkpoint's exact settings.

Track position/family policy KL and value MSE, each opponent, each move phase,
value constant-predictor baselines, gradient clipping, update acceptance,
exposures, learning time, padding, compilation, evaluation and checkpoint cost.
Retain the existing three-observation, 0.5%-margin overfit flags alongside the
training probe. They are diagnostic flags, not an automatic best-checkpoint
selector. If validation worsens while the training probe improves, review the
curves before extending the horizon or launching another seed. Preserve all
observations and distinguish a weak/noisy value stratum from global overfitting.

The source CNN has passed full-size execution and exact all-rank recovery.
The fixed corpus is verified on all hosts. Transformer execution with a
16-frame chunk exhausted HBM before its first update; the retry used eight
independent encoder frames per chunk and passed all four updates and its
complete checkpoint audit. This
execution adjustment leaves the model, decoding FLOPs and global game batch
unchanged. CNN chunks remain at their qualified size of 16 frames.

The final launch inputs and storage reservations are pinned in
`pilot-registration-001.json`; the source-optimizer CNN arm has started.
The fixed validation hash
covers all 146 games; the 128-game training probe contains 53,743 positions.
Each completed arm retains two verified peer checkpoint copies and an owner
restore locator, keeping RAM use bounded across the sequence. Then use trained
joint checkpoints for paired-colour KataGo matches; validation loss and the
small interoperability fixtures do not establish playing strength.
