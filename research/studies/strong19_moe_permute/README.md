# Routing execution study

[PLAN.md](PLAN.md) describes two independent hypotheses: aligned grouped-matmul
tiles and permutation-based dispatch/combination gradients. The original
learner completed unchanged. The 70-case TPU screen also completed and passed
its complete audit; [results are here](kernel-review-003/RESULTS.md).

The default-off implementation is in `gozero.moe`, with configuration wiring in
the cloned `strong19_moe_permute` recipe. Six core unit tests pass, covering
oracle gradients, padding, empty experts, both activations, FP32/BF16, and
activation recomputation. `cpu-qualification-001.json` also passes complete
small-model loss/gradient checks, three AdamW updates and exact restored
continuation. Largest full-model gradient differences were 8.94e-8 (FP32) and
7.45e-9 (BF16). Largest differences anywhere in the three-update reports were
2.38e-7 and 9.31e-10 respectively; these maxima include scalar diagnostics.

The initial standalone bitwise gate failed on compiled input-gradient sums,
with maximum error 4.77e-7; forward outputs and parameter gradients remained
exact. An unfused control was fully exact. The revised unit contract preserves
exact forward/parameter checks and explicitly permits FP32 rounding in input
gradient accumulation. Both the failed gate and its diagnostic are retained.

The current preparation is `kernel-registration-003.json`, superseding the
unlaunched earlier preparations. It covers 70 cases: two encoder shapes, both
temporal training buckets, eight-token decoding, and balanced/concentrated
routing. Controls and alternatives share a physical device within each group.
[PLAN-003.md](PLAN-003.md) adds 384-aligned tiles and a stricter whole-gradient
error gate. Runtime kernel sources match on all configured hosts. Every case passed;
the largest complete-leaf relative-L2 discrepancy against the independent
all-expert oracle was 0.0036862, below the registered 0.01 limit.

Permutation gradients at the original tile were generally fastest for
forward/backward work: observed speedups were 1.22–1.27x at the temporal
training shapes and 1.06–1.16x at encoder shapes. The 384-aligned alternatives
were not a universal improvement. These are synchronized kernel medians,
not complete-learner speedups. Repeated dense controls varied by about 10–24%
across encoder candidates and 3–13% across temporal candidates, so small
differences need caution. Timing samples and control ranges are retained.

The temporal-only full-model systems comparison takes priority over adopting
an all-expert kernel change: the completed all-expert learner costs 2.245x
dense learning time for a mixed quality result. Full-model state, memory and
timing qualification remains necessary before adopting permutation gradients.

`review_kernels.py` summarizes the completed audit with forward and backward
timing ranked separately. It does not alter any model or queue learning.

[EXECUTION_PLAN.md](EXECUTION_PLAN.md) records a prepared complete-learner
comparison for a later kernel winner. Its source entrypoint compares full
optimizer states in RAM; it has no registered candidate or TPU result yet.
