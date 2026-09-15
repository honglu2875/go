The completed two-seed comparison was reviewed in
[lr15-review-001.md](lr15-review-001.md). The first LR-only intervention is
registered in [lr15-registration-001.json](lr15-registration-001.json): peak
`1.5e-3`, end `4.5e-4`, with every other setting unchanged. Seed 1 started on
2026-09-15 at 13:24 UTC; its immutable launch receipt is
[lr15/seed1-process-001.json](lr15/seed1-process-001.json).

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
room for a full checkpoint plus producer growth before launch. The inspect
gate passed against the real registration before the first launch.

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
are fixtures, not experimental learning results.

[finalize_lr.py](finalize_lr.py) binds the actual attempt to its launch receipt,
retains a verified peer copy of the complete checkpoint, runs the frozen
learning audit, and computes the paired contrast. Seed 2 is permitted only if
both endpoint KL measures improve by at least 0.5% against seed 1's transformer
parent, neither final-three mean regresses, and no sustained overfit is present.
Both seeds must meet that criterion to accept this LR intervention. Another
rate or encoder change requires a new evidence review and registration.

[continue_lr.py](continue_lr.py) now owns the running first seed's closure and
conditional second-seed confirmation under
[lr15-continuation/plan-001.json](lr15-continuation/plan-001.json). It updates
the monitor and three-arm validation/probe plots every five minutes, audits
each completed stage, and stops after this intervention. Ten decision checks
cover the exact threshold, either primary measure failing, tail regression,
overfit, wrong identities and malformed results. Its real registration inspect
and first live monitor/plot cycle passed. Source changes or failed evidence and
resource gates stop the continuation while the active learner retains its own
controller. Do not run a second finalizer alongside it.
