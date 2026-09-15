This is a design note, not a registered experiment or permission to launch.
The fixed scientific cohort and final transformer still depend on completed
evidence. The prepared 19×19 execution fixtures cannot serve as learning data.

The first learning comparison should measure learning on the same frozen
teacher-labelled games. Both models receive the same complete histories,
augmentation draws, policy distributions and signed player-to-move values.
Splits stay grouped by opening family. Keep the test population closed during
architecture and optimizer selection. Validation reports must include policy
KL and value MSE per position, per family, per opponent and per move phase,
alongside a fixed training probe. Teacher self-play deserves explicit counts:
an overall metric dominated by games against weak opponents can hide poor
decisions in balanced positions.

The full models already prepared have 233,220,870 training parameters for the
CNN and 232,011,540 for the transformer. The latter keeps its width and reduces
the one-token spatial connector to 16 channels at board size 19. Complete
deployed policy/value decoding, including the encoder and readout, is within
one percent of the CNN's analytic FLOP budget at the declared batch/context.
The encoder initialization used for the scientific transformer must follow
the completed paired 9×9 decision. Execution qualification does not select it.

For the CNN, retain the qualified source Muon/AuxAdam, adaptive decay, Lookahead
and BN-free FSON implementation. Its source defaults include Lookahead period
6, interpolation 0.5 and a nominal million-position epoch. Its early LR warmup
changes at 250,000-position intervals through two million positions. Our short
recovery fixture deliberately overrides these clocks; those overrides must
not leak into a scientific configuration.

There is an unresolved batch adaptation to settle explicitly. A batch of 128
complete games contains tens of thousands of training positions, whereas the
source schedule takes a fixed position-batch reference. Retaining the
qualification value of 256 would silently change the intended learning-rate
and clipping scale. A reasonable primary adaptation is a fixed reference
computed from the frozen cohort's expected live positions per complete-game
batch, while advancing the source sample clock by the actual count. This
preserves uniform game draws and efficient full-history training, but it is
not exact fixed-position KataGo sampling. Record the resulting reference,
rates, decay, clipping and epoch boundaries before launching. An AdamW CNN
control would help separate effects of this optimizer transfer from those of
the architecture; it must not be conflated with the source-optimizer arm.

Choose training length in position exposures after the cohort freezes, rather
than copying the 4,096-update 9×9 horizon. The larger9 study used about three
passes over its training positions. A much smaller 19×19 cohort and longer
games make 4,096 updates a substantially different experiment. Budget enough
updates to pass the source warmup and observe a useful post-warmup interval,
validate regularly, and preregister how sustained train/validation divergence
will affect continuation. Do not shorten the data itself in response to a
single noisy validation observation.

After the learning comparison, benchmark actual trained joint checkpoints
through the qualified Rust search and joint policy/value adapter. Use paired
colours and shared openings, keep search work and inference cost explicit,
and report uncertainty against pinned intermediate KataGo checkpoints. Small
untrained-model interoperability games and lower validation loss are neither
substitutes for this strength test nor evidence of faster RL improvement.
