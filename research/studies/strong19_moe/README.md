# Matched-active-compute sparse experts

Initial learning screen complete, October 4. The user chose equal active FLOPs with more total
parameters allowed. [PLAN.md](PLAN.md) fixes data, scientific controls, safety
gates, the bounded autonomous window, and references.

[The completed round report](ROUND_20261004.md) combines learning, kernel and
systems results with the next experiment. Temporal-only experts add 36.6%
parameters with 0.30–0.38% update-time overhead in the short systems check;
their learning quality remains untested. All pod jobs are closed.

The implementation is in the cloned `research/recipes/strong19_moe` recipe,
with reusable routing/grouped matmuls in `gozero.moe`. No framework or new
runtime dependency is added. The original frozen dense recipe is unchanged.

The initial candidate has four experts and top-two token routing throughout
the spatial encoder and temporal transformer. Each expert has half the dense
FFN width. It has **420,991,764 parameters**, versus **232,011,540** dense;
complete cached-move matrix FLOPs (including encoder and routers) rise by
**0.0637%**. See `budget-001.json` for every tensor and context-length budget.
These are logical matrix FLOPs; measured runtime includes tile padding,
sorting, dispatch, router arithmetic, and the larger optimizer state.

`cpu-qualification-003.json` passed dense-parent exact equivalence, FP32/BF16
bounded/materialized losses and all gradients, padding invariance, causal
history/cached decoding, actual checkpoint serialization and continuation,
semantic optimizer decay, and four-device global router loss/gradients with
an empty shard. `cpu-qualification-004.json` repeats the checks after adding
explicit tiling and router diagnostics. Tests use small models; TPU resource
and learning evidence are separate.

The first TPU attempt failed because total rows were not tile aligned. The
fix pads zero rows to the final expert and removes them after the grouped dot.
The next attempt exposed a reference-precision mismatch: its router dot used
TPU default precision while the implementation requested highest precision.
Correcting the independent reference made every forward/backward case pass
(`kernel-qualification-003.json`), including empty groups and ragged rows.
Neither failure has been overwritten or counted as learning evidence.

An 18-case tile/backend sweep (`kernel-qualification-004.json`) showed that
Pallas `(256,512,512)` is a reasonable general tile for this workload, with
roughly 2–3x faster sparse kernels than `(128,128,128)`. Sparse kernels still
trail the dense FFN in these small synchronized forward/backward benchmarks.
The native JAX ragged-dot path passed correctness but was slower for encoder
shapes. Full-model systems qualification passed (`system-qualification-001.json`
and `system-review-001.json`): approximately 144 seconds / 29.19 GiB for the
512-position bucket, and 187 seconds / 29.93 GiB for the 768-position bucket.
The contemporaneous dense 512-position control took 64 seconds / 24.60 GiB.
Every shared dense first-update metric exactly matched the historical reference.
The present sparse implementation is about 2.26 times slower on that bucket.

A registered 64-update learning screen completed from the immutable
`balance010-001-plan.json`. It preserved the 512-update LR schedule and the
dense run's complete-game draws, with validation every 16 updates and durable
checkpoints at 32 and 64. All-rank update, draw replay, optimizer/RNG checkpoint
and disk-peer audits passed. Each model received 3,596,975 position exposures.
See [the endpoint results](balance010-001-RESULTS.md) and full curves in
`balance010-001-comparison.json`.

Policy KL was **1.086170 versus dense-transformer 1.133431** (4.17% lower),
but value MSE was **0.195272 versus 0.169567** (15.16% higher), and top-one
accuracy was lower. Recorded learning time was **2.245 times greater**.
This is a mixed quality result and does not meet the time-efficiency goal.
It is one paired seed and a short prefix with 40 warmup updates, not evidence
of convergence or playing strength. The dense reference is a transformer,
not the earlier CNN baseline. The held-out test split remains closed.

The separately cloned [activation-memory follow-up](../strong19_moe_remat/README.md)
has passed unchanged-chunk CPU equivalence checks; the larger-chunk BF16
first-moment check failed its registered tolerance and is not qualified.
The [kernel study](../strong19_moe_permute/README.md) and
[temporal-only systems arm](../strong19_moe_temporal/README.md) are separate
follow-ups; neither changes the completed learner. Router coefficients in this first screen are averaged
over distinct layers; their numeric values are not an exact reproduction of
another paper's layer-reduction convention.

Dataset files remain in RAM on each host, backed by the existing verified disk
copies. Two older, closed checkpoint bundles were fully archived and verified
on two disk peers before retiring their duplicate local payloads; all research
metadata remains. See `archive-20261004-result-001.json` and associated receipts.
Machine addresses, private paths, and these operational receipts are local
artifacts and must be sanitized before any public export.
