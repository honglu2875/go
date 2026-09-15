The scalar settings surrounding standard Muon now match the pinned KataGo FSON
source across 320 cases and 7,680 comparisons. Rates and clipping caps match
exactly; the largest relative decay difference is 4.30e-16. Seven invalid-input
cases are rejected. The qualified snapshot is
`bddb832e7958c0a946fd5c46cc1a9bec5c95a32d60cee80073bff3d8c75ba012`.

The [reference exporter](../../recipes/katago_muon_schedule/export_reference.py)
executes the original nested functions and clipping statements as extracted AST
nodes after verifying the complete source hash. It does not import or launch
the upstream trainer. The [port](../../recipes/katago_muon_schedule/schedule.py)
requires explicit sample count, global position batch, effective LR scale,
group factors, norm ratios, lookahead alpha and warmup choice.

Coverage includes both sides of all eight warmup thresholds, four position
batch sizes, distinct LR scales and group factors, absent or changed running
norm ratios, and enabled or disabled lookahead LR scaling. These are source
equation checks, not selected research settings or estimates of a published
checkpoint's undocumented training arguments.

The source's world-size compensation before DDP averaging produces a gradient
of the global summed loss. Our joint loss is position averaged. A neural-learner
integration therefore needs the global-position multiplier before RepVGG
gradient scaling and clipping. The Muon clipping base is 11,000 in the source's
summed-gradient units, with batch/LR/explicit-factor scaling applied afterward.
It should not be copied directly as a threshold on position-mean gradients.

The [source runtime observation](source-runtime-observation-001.json) executes
the original metric functions and inspects the trainer's AST nesting. With
the default print-only norm setting, `set_snapshot_metrics` replaces the prior
sum/weight before logging: the ratio is the latest pre-update norm snapshot,
not an EMA. The subsequent 0.001 multiplier preserves that ratio. Lookahead's
counter resets at each subepoch entry; its slow-to-fast flush is outside the
subepoch loop, at epoch end. A missing `json` execution-namespace import in the
first observation attempt is retained in `source-runtime-execution-001.json`;
the corrected check executes the original equations unchanged.

The separate [joint Muon integration](../strong19_muon/README.md) now qualifies
neural-loss conversion and complete small-model recovery with fixed explicit
group settings. Source schedule/norm cadence, Lookahead slow weights/counter,
epoch behavior and measured full-size TPU overhead remain to be integrated.
Lookahead LR scaling here does not implement parameter averaging. The running
9×9 AdamW comparison is unchanged.
