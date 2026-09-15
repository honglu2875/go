The existing continuation queue ends after producing both paired contrasts.
`conclude.py` then reproduces each contrast from its independently audited
inputs before writing a combined JSON and Markdown report. It schedules no
jobs and does not choose a checkpoint from the best validation turn.

`conclusion-plan-001.json` makes the existing prose criterion explicit: average
the per-seed relative endpoint gains; require improvement on both primary
metrics in each seed, at least 0.5% mean relative gain on both metrics, and
nonnegative last-three-mean gain for each seed and metric. These choices were
recorded before the first transformer run. If a gate fails, the report says
the criterion was not met; that alone does not prove CNN superiority.

Seven local decision tests passed, covering contradictory seeds/metrics,
unequal baseline scales, small effects, tail versus endpoint disagreement,
best-turn substitution and malformed evidence. The full real-run re-audit
cannot execute until all four learning runs and both contrasts are complete.

Inspect readiness without mutations:

```bash
.venv/bin/python -B research/studies/strong9_scaling/conclude.py \
  --plan research/studies/strong9_scaling/conclusion-plan-001.json \
  --plan-sha256 22cd6790fb7afae43506bacf91d911eda116385a32ef6a77f4464fa1c1513f59 \
  --inspect
```

When all evidence exists, replace `--inspect` with
`--output research/studies/strong9_scaling/conclusion-001.json`. The operator
requires new output paths and also emits `conclusion-001.md`. It reports all
registered validation and training-probe points, overfitting flags, observed
attempt reservations and deployed decode latency. It never reads test targets.
