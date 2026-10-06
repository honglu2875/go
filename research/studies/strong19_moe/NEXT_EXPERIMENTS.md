# Decisions after the initial MoE round

All three registered TPU stages passed and closed. The all-expert learner
improved endpoint policy KL by 4.17%, worsened value MSE by 15.16%, and cost
2.245x learning time. It does not meet the efficiency goal. The 70-case kernel
screen found useful permutation-gradient speedups, but those are not complete
learner gains. The temporal-only systems arm adds 36.63% parameters with just
0.30–0.38% measured update-time overhead. Its learning quality is untested.
See [the complete round report](ROUND_20261004.md).

1. Prioritize a fresh **128-update temporal-only learning arm**. Reuse its
   qualified model/source, exact original dense-reference draws, seed,
   optimizer, objectives and full 512-update LR schedule. Keep the per-router
   auxiliary-loss convention from the systems arm. Validate every 16 updates
   against the fixed validation population and training probe. Preserve full
   optimizer/RNG checkpoints at 64 and 128. This extends the post-warmup
   comparison without mixing an optimizer change into expert placement.
2. Before registration, recheck every host and reserve checkpoint space. A
   temporal-only FP32 model plus AdamW moments requires about 3.55 GiB per
   checkpoint, excluding small metadata. Two payloads plus the owner's 2 GiB
   reserve require at least about 9.1 GiB, with a larger margin preferable.
   The current owner has about 8.28 GiB free. Archive verified duplicates before
   launch; do not assume RAM alone supplies durable recovery. Reserve the
   disk peer's capacity and floor independently. Keep prior critical endpoints.
3. Judge policy KL, value error, top-one accuracy, fixed-train-probe separation
   and learning time together, using the registered endpoint rather than the
   best intermediate validation. The current dense reference is a transformer,
   not a CNN. Its fresh first update reproduced all 36 common historical
   metrics in the latest systems check. A promising result needs a second
   paired seed and a longer horizon before architecture promotion.
4. Keep execution optimization separate. Original-tile permutation gradients
   were generally fastest in the kernel screen, especially at temporal shapes.
   The complete-learner state/memory/timing gate is prepared in
   `strong19_moe_permute/EXECUTION_PLAN.md`, but no candidate has passed it.
   The 384-aligned tile alternatives were not a universal improvement. Do not
   silently switch kernels in a continuation or call the microbenchmark an
   end-to-end speedup.
5. If the temporal learning arm shows router instability, isolate router LR
   in a separate matched-seed arm. Initialization and layer-averaged auxiliary
   coefficients have different scales from common paper conventions; see
   `OPTIMIZATION_NOTES.md`. Do not simultaneously change initialization,
   balancing, architecture and optimizer.

Preserve the failed larger-chunk BF16 first-moment check. Unchanged-chunk
activation rematerialization has CPU equivalence evidence only; neither
proposal has a full-size TPU resource/performance qualification. A future
numerical study must justify a new contract explicitly, retaining the failure.

No new learning job is queued. The pod has been released after the bounded
round. Dataset collection stays stopped and the test split stays closed.
Real search/KataGo evaluation remains a later promotion gate; none of this
round's supervised or timing measurements establishes stronger play.
