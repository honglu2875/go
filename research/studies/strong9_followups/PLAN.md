The user authorized continued autonomous work on 2026-09-15: let the ongoing
comparison finish, then try simple learning-rate and encoder adjustments if
the transformer's result is unsatisfactory. Keep the current goal, fixed-data
learnability comparison, parameter budget and complete decoding-FLOP constraint.

Finish the existing larger-9×9 registration and its paired seeds first. Its
guarded continuation owns the accelerator. An intermediate same-step gap is a
diagnostic, not a reason to change a running schedule or declare a winner.
At update 1,024 the first-seed position-KL gap is 3.16%, down from 11.92% at
768. Both models' validation and training-probe losses are still decreasing.
The final two-seed criterion and regular overfitting checks remain in force.

If the transformer fails the registered improvement criterion, inspect all
position- and family-weighted validation/probe curves, learning-rate schedule,
gradient clipping and group updates before choosing the next intervention.
Higher training and validation losses motivate an optimization/learnability
probe; sustained falling training loss with rising validation loss motivates
a separately paired horizon or regularization study instead. Position and
equal-family metrics have different populations and must each be compared to
their own matching control.

The first candidate is a learning-rate-only change. A narrow upward range
around the current 1e-3 peak is reasonable, but rates and the decision rule
must be frozen after the current result is reviewed and before any follow-up
launch. Preserve the 64-update warmup and 0.3 final/peak ratio when isolating
the rate scale. Reuse the exact numerical source and matched initialization,
whole-game/D4 draws, full validation and training-probe populations. A shorter
screen needs a fresh contemporaneous control with the same cosine horizon;
the first 1,024 steps of the current 4,096-step schedule are not that control.
Use one intervention at a time and confirm a selected gain with the paired
second seed. Do not call a selected rate globally optimal.

Only after reviewing that result should an encoder intervention be selected.
Possible simple controls include residual-scale initialization or the balance
of convolutional and attention blocks. Prefer a single changed mechanism.
Changing channel counts, unique depth or shared passes requires a fresh full
parameter/encoder-plus-decoder arithmetic check and a stated compensating
temporal-depth adjustment. Retain one causal board token, width matching,
shared spatial readout and normalization without batch statistics in the
transformer unless a separately justified experiment explicitly changes them.
The failed rank-128, nonlinear-readout and refinement screens remain negative
evidence, not automatic candidates to rerun on the larger corpus.

This file records research direction, not a learning registration or a queued
job. Freeze exact source/configs, seeds, horizon, metrics, decision threshold,
storage reservation and launch prerequisites before scheduling. Keep test
targets closed, retain negative results and checkpoint continuation state,
and distinguish supervised losses from actual KataGo playing strength.
