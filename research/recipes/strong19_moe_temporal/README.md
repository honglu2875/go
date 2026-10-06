# Dense encoder with temporal sparse experts

This clone retains the qualified dense visual encoder and replaces only the
temporal SwiGLU FFNs when `model.moe.scope` is `temporal`. Four independent
experts, top-two routing, and half-width expert FFNs increase total capacity
while matching the dense active FFN matrix work. The default `all` scope
retains the parent recipe's all-expert behavior.

Board encoding, spatial readouts, causal attention, auxiliary draft tokens,
signed-value objective, data pipeline and optimizer stay unchanged. The
shared sparse implementation is pure JAX with the bundled Pallas grouped
matrix kernels; no Flax dependency is introduced. Router statistics exclude
the dense encoder's empty rows. Per-layer auxiliary coefficients are recorded
explicitly in the experiment configuration.

See [the study](../../studies/strong19_moe_temporal/README.md) for qualification
and registration. A CPU correctness result is not a TPU throughput or learning
result. The separate permutation-VJP optimization is rejected by this recipe.
