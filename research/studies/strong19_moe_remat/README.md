# Sparse activation-memory follow-up (prepared)

The [plan](PLAN.md) proposes activation-only rematerialization as an execution
optimization. No TPU follow-up has started; the original learning run retains
its immutable source and configuration.

CPU qualification passed in `cpu-qualification-001.json`. At unchanged chunk
size, full model outputs, metrics, all gradients, and three AdamW updates were
exactly equal in both FP32 and BF16. Actual checkpoint restoration continued
exactly. Doubling the small fixture chunk gave maximum absolute differences
of 1.96e-8 in FP32 and 5.19e-4 in BF16; this requires a separate full-size TPU
numerical check before adoption. All five core MoE unit tests also passed.

The subsequent `chunk-rounding-002.json` diagnostic separated gradient norms
from scalar metrics/counters. Larger chunks have a gradient-only relative L2
difference of 7.68e-8 in FP32 and **0.001878 in BF16**. The BF16 three-update
trajectory fails the proposed conservative first-moment drift limit (0.001),
while FP32 passes. Same-chunk rematerialization remains exactly equal.
Consequently chunk 16 is **not qualified for adoption**. Isolate chunk-size
rounding against a same-chunk control before considering a fresh learning
comparison. The failed diagnostic wrapper invocation (missing a keyword)
remains recorded as `chunk-rounding-001.json`; it did not train a model.

Abstract JAXpr inspection reduced saved convolutional GELU scan outputs of
shape (4,5,5776,1536) in FP32 from ten arrays (7,097,548,800 bytes total extent)
to two (1,419,509,760 bytes). This is **not** a measured peak-HBM reduction.
The concrete next test compares compiled memory, full-state update differences,
and timing for original chunk 8, activation remat chunk 8, and remat chunk 16,
with a 31 GiB allocation guard. Expert matrix multiplies are not rematerialized
by this option; the pre-existing whole-chunk rematerialization remains.

A separate aligned-tile proposal is recorded in
`aligned-kernel-config-002.json` (24 cases; version 001 was the initial
12-case balanced-routing proposal). Width 768 is not divisible by the current
K/N tile width 512. The Pallas source rounds irregular dimensions up and masks
unused lanes; exact-width tiles may reduce padding work. Testing 256/768-aligned
tiles is a systems hypothesis, not a proven speedup. Preserve the current
qualified tile as a contemporaneous control. These tests are not queued while
the scientific learner occupies the pod.

Version 002 adds concentrated top-two routing for each shape/tile, because the
current learner's expert usage is uneven. A balanced random-input benchmark
alone would not adequately represent that regime. Both empty-expert gradients
and selected-expert outputs must still agree with the independent oracle.

`system-proposal-config-001.json` and the cloned recipe's `qualify_remat.py`
specify two full optimizer updates for each of three variants and both buckets.
Reference states stay in host RAM, and only small scalar diagnostics are saved.
At unchanged chunk size, the preregistered requirement is bitwise equality of
every parameter, optimizer moment, counter and metric. Chunk 16 uses the prior
throughput study's explicit groupwise drift limits, with coordinate failures
reported separately. No full-size numerical result is implied by this proposal.
