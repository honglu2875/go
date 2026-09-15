The standard Muon/AuxAdam kernel now runs the unchanged joint CNN policy/value
learner and passes exact fresh-process continuation on real 19×19 games.
This is a small CPU execution qualification, with explicit fixture rates and
decays. It does not select the scientific 19×19 optimizer settings.

The [training recipe](../../recipes/strong19_muon/README.md) retains the qualified
model, loss, bounded gradient path, evaluation and whole-game/D4 sampler.
Its optimizer bridge converts globally averaged gradients to global position
sums, applies the source RepVGG center multiplier, then clips and calls the
reference-qualified Muon/AuxAdam kernel. Source/configuration identities are
recorded in [integration-source-001.json](integration-source-001.json).

The [independent bridge check](bridge-cpu-001.json) covers all six CNN parameter
groups, stacked and unstacked spatial kernels, input kernels that must not get
the center multiplier, and active clipping. It verifies the resulting first
and second moments against independent NumPy arithmetic and rejects six invalid
updates without changing parameters or optimizer state.

The [full harness](harness-cpu-001.json) passed in 472.84 seconds. Uninterrupted
four updates and fresh-process two plus two updates consumed the same 7,907
position exposures. All 202 checkpoint arrays (24,263 elements) match exactly,
as do all three sampler RNGs, counters, update metrics apart from timings,
and validation/probe/overfit histories. The model has 9,794 training parameters:
12 parameter leaves use Muon, and 59 use auxiliary Adam. Both 512- and
1,536-position runtime buckets are exercised without truncating games.

The [trained-inference check](inference-cpu-001/result.json) also passed. All
54 exported main-policy/value parameter arrays match their checkpoint bytes;
17 training-only helper arrays remain in the full checkpoint. The 9,297-parameter
export passes 48 policy/value comparisons and two actual Rust searches.
Cached/full policy disagreement is at most 6.71e-8. The shared export and
serving library needed no optimizer-specific change.

The source-derived schedule, norm-observation cadence and Lookahead state still
need integration and full-size TPU qualification before scientific learning.
The [source runtime observation](../katago_muon_schedule/source-runtime-observation-001.json)
clarifies that the default norm ratio uses the latest print-batch snapshot,
and that slow weights are copied to fast weights after the complete subepoch
loop. The current 9×9 AdamW comparison is unchanged.
