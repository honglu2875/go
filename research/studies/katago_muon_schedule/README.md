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

Running norm observations/averages and their update cadence, lookahead slow
weights/counter and epoch flushing, complete learner recovery, and measured TPU
overhead remain required. Lookahead LR scaling here does not implement its
parameter averaging. The running 9×9 AdamW comparison is unchanged.
