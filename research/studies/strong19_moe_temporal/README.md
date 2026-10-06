# Temporal-only MoE

See [PLAN.md](PLAN.md) for the controlled comparison. The frozen four-case
TPU systems qualification passed after the all-expert learner and the
70-case kernel screen passed their audits. See `system-stage-001/audit.json`
and [the systems review](system-review-001.json). This variant has no
learning curve or playing-strength result.

`cpu-qualification-001.json` passed ten complete-model checks.
`cpu-qualification-002.json` passed eleven checks, additionally rejecting an
unsupported execution optimization. FP32/BF16 gradients, exact dense encoder
initialization, complete causal history, cached decoding, checkpoint
continuation and global router reductions all passed.
`budget.py` derives tensor counts without allocating the full model and
cross-checks the original dense/all-expert arithmetic.

The rationale is to add approximately 85 million temporal expert parameters
while retaining regular dense matrix operations throughout the expensive
spatial encoder. The verified count is 317,001,492 parameters and the logical
cached-move matrix FLOP ratio is 1.00000132. Full-size measured update times
are 63.864 versus dense 63.672 seconds at the 512-position bucket, and 84.198
versus 83.877 seconds at the 768-position bucket. This is 0.30–0.38% overhead
for 36.63% more parameters. Compiled memory is 25.56/26.30 GiB, about 0.96 GiB
above dense. Inference latency has not been measured for this complete model.

Each systems case ran two fresh-state updates; reported timings are the
second update's slowest host. Every rank agreed on metrics and all 36 common
fresh dense first-update metrics exactly matched the historical reference.
This short timing result is promising, not a long-run throughput guarantee.

`prepare_system.py` freezes a four-case TPU qualification. `execute_system.py`
requires a successful closure of the first learner, an idle pod and resource
headroom. The registered job has now closed successfully. A separate frozen
learning registration and checkpoint-space reservation are required before
the proposed 128-update learning comparison; no new learner is queued.
