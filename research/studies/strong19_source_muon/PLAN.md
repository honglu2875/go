Integrate the separately qualified KataGo Muon/AuxAdam kernel, Lookahead,
sample schedule and print-batch norm cadence with the existing pure-JAX
joint CNN policy/value learner. Preserve the numerical network, losses,
whole-game sampler, bounded gradient path and held-out evaluation code.

The immutable execution plans are `harness-plan-001.json` and
`harness-plan-002.json`. Plan 001 was superseded before execution to record
pre-update norms and independently replay every applied schedule setting.
Only plan 002 ran. This document explains their scope; it does not replace
or retroactively register either plan.

Use the existing small real-19×19 fixture on four simulated CPU devices.
Eleven uninterrupted updates must match a six-update prefix plus five
fresh-process updates exactly. Put the restart immediately before a
Lookahead synchronization, with distinct fast and slow weights. Exercise
uneven subepochs, an epoch flush, a warmup boundary and norm snapshot lag.
Retain all parameters, moments, slow weights, device hyperparameters,
host schedule state, sampler RNGs and diagnostic histories.

The 256-position schedule reference is separate from the variable number
of live positions in four complete games. Actual live positions convert
mean gradients to sums and advance the sample clock. The initial sample
offset, short epoch lengths, print cadence and Lookahead period are explicit
execution fixtures, not selected scientific hyperparameters.

After exact recovery passes, use the unchanged qualified serving recipe
to export the trained main parameters and compare full, cached and native
Rust search inference. These CPU checks consume no new accelerator run.
Full-size TPU qualification, the scientific 19×19 batch convention and
learning registration remain separate prerequisites.
