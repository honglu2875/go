# Routing permutations and aligned expert tiles

This is a separate execution study prepared during the frozen first MoE
learning screen. It does not change that learner, its initialization, data,
objective or routing decisions. No new accelerator job is queued yet.

The optional `permutation_vjp` path computes the inverse of the stable expert
sort once. Dispatch backpropagation gathers into original assignment order
and sums the K copies of each token. Combination is a permutation gather in
both directions. This avoids repeated-index feature scatter-add in dispatch
and unique-index feature scatters in combination; expert bias gradients are
unchanged. Scalar index inversion is still a scatter. No token is truncated.
The option defaults off. This does not change logical active matrix FLOPs.

The technique follows the ordinary transpose of a permutation. Google's
[MaxText MoE implementation](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/moe.py)
also offers inverse-permutation custom gradients when automatic gather
backpropagation is inefficient. Our small pure-JAX implementation does not
import MaxText or Flax.

First check both activations, FP32/BF16, concentrated routing, empty experts,
padding, full-model gradients, AdamW trajectories and checkpoint continuation.
The initial bitwise unit gate failed only for compiled input-gradient sums
(maximum 4.77e-7). The unfused control is exactly equal; forward results and
all parameter gradients of the single layer are exact. Preserve this failure
and report floating-point input-gradient association separately. Full-model
numerical behavior still needs its own qualification.

The finite `kernel-proposal-config-002.json` compares the current qualified
tile, three aligned-tile alternatives, and the permutation VJP on the current
tile. It covers both encoder FFNs, temporal FFNs at 512 and 768 positions,
and eight-token temporal decoding. Balanced random routing and concentrated
top-two routing are both included. Each candidate and its exact-shape control
run on the same physical device; explicit logical-rank assignments spread
independent shape/routing groups over the four hosts. Ten synchronized timing
samples cover forward-only and forward/backward latency. The independent
all-expert reference checks outputs and every gradient before timing.

Run this bounded microbenchmark only after the registered learner releases
the pod and current device/storage checks pass. Select a candidate from
measured results, then validate the complete learner's memory, state drift
and timing before adoption. Neither a source-level operation count nor a
microbenchmark win establishes end-to-end speed or a learning improvement.
