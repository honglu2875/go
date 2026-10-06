# Spatial and temporal sparse experts

This recipe clones the qualified fixed-data causal visual transformer and
replaces spatial and temporal FFNs with independent routed experts. The first
study uses four experts, top-two selection and half-width expert FFNs to match
active matrix FLOPs while adding total capacity. The shared implementation is
`gozero.moe`: pure JAX, FP32 routing and BF16 grouped expert matrix operations.
Routing is dropless and independent of unrelated tokens or future positions.

The 24-block visual encoder is shared across two passes; the 18-layer temporal
transformer consumes causal board/action history. Attention, spatial policy
readout, signed-value head and training-only first-pass auxiliary supervision
remain unchanged. Dense non-FFN initialization draws are preserved. AdamW
decays expert/router matrices under the explicit name-based parameter policy.

See [the study](../../studies/strong19_moe/README.md) for exact budgets,
qualification, frozen learning configurations and results. Kernel tests,
full-model systems checks and supervised learning are separate evidence.
Old dense diagnostic entrypoints are inherited for reference; they are not the
registered sparse learning experiment.

The mutable adapter rejects temporal-only scope and the permutation-VJP
optimization; those variants have separate cloned recipes. Historical frozen
snapshots remain unchanged.
