# Whole-history D4 and linear-precision experiments

This clone owns its complete pure-JAX model, AdamW learner, held-out evaluation and checkpoint recovery. D4 transforms entire histories, actions, policy targets and ownership together. Its independent per-host PCG64 stream is saved; episode sampling and learning schedules match the unaugmented control.

The fixed 1,024-update 233M run reduced validation behavior CE / expert KL / value MSE by 39.33% / 31.31% / 25.05%. CPU and multi-host recovery reproduced parameters, moments, samplers and D4 draws exactly. This is an established baseline and one weak-teacher seed, with fresh real-game evaluation required. The separate DEFAULT-versus-HIGHEST precision screen gave 1.139x and failed the speed gate; HIGHEST remains the learning control.

Freeze each resolved configuration before execution. Existing snapshots and failed attempts remain immutable. Clone this complete recipe for a scientific intervention; shared environment and artifact helpers are imported from `gozero`.
