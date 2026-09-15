The LR follow-up is prepared, but no rate is selected, registered or queued.
Complete and review the current two-seed comparison first, as required by
[PLAN.md](PLAN.md).

[lr_source.py](lr_source.py) clones an intervention from the retained frozen
parent and permits only the peak/end learning-rate endpoints to change. It
preserves the 4,096-update horizon, 64-update warmup and final/peak ratio 0.3.
All 1,310 source files remain identical. Its CPU qualification rejects 14
confounders and reproduces the unchanged parent's exact snapshot identity.

[launch_lr.py](launch_lr.py) accepts an explicit two-seed plan only after the
completed-study conclusion and a rate-specific review are pinned. It checks
both parent audits, identical numerical sources and the existing full-shape
qualification. It permits the second seed only after the first meets the
registered screen criterion. It checks exclusive pod ownership and reserves
room for a full checkpoint plus producer growth before launch. No plan or
process receipt for this operator exists yet.

[lr_compare.py](lr_compare.py) compares the full endpoints and last three
validation points against the matching transformer parent. It rechecks all
audited inputs, initialization, per-rank game/D4 draws, position exposures and
evaluation populations. Both position- and equal-family KL must meet the
plan's explicit screen threshold; the tail must not regress and a sustained
overfitting flag prevents confirmation. Per-seed gains remain evidence for
that rate and dataset, not a claim of global optimality or Go strength.

The [comparison qualification](lr-comparison-cpu-001.json) reads actual audited
records, verifies identity and known synthetic gain arithmetic, and rejects
ten corrupted schedules, populations or metric cases. Those altered curves
are fixtures, not experimental learning results. The launch operator still
needs its inspect gate exercised against the eventual real registration.
