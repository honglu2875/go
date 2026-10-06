# Additional aligned tiles and stronger numerical qualification

This supersedes the unlaunched 50-case preparation with 70 fixed cases. Keep
every original control and candidate, and add `(256, 384, 384)` and
`(512, 384, 384)` in each of the ten shape/routing groups. Controls and
candidates within a group still use the same physical device. There are ten
synchronized repetitions for each forward and forward/backward measurement.

The installed JAX grouped kernel has a parallel output-column grid dimension.
Its other two grid dimensions are sequential. A 384-wide tile divides both
768 and 1536 and yields an even number of output tiles for these widths.
This could reduce padding while retaining parallel work on a two-core TPU.
A 768-wide output tile instead yields only one output tile at width 768.
These are source-level reasons to test the alternatives, not measured speedups.

[JAX's TPU kernel guide](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
explains the role of parallel grid dimensions and balanced core work.
The installed implementation, rather than an assumed newer API, is inspected
and hashed in the preparation report. No dependency or runtime is changed.

Strengthen the independent oracle gate: each complete output/gradient leaf
must have relative L2 error at most 1%, in addition to the existing coordinate
tolerances. The previously qualified cases were all below 0.37%; this new gate
prevents a loose absolute tolerance from concealing errors in small gradients.
It is fixed before observing any new TPU result. Failures remain failures.

The grouped-matmul, routing and custom-gradient functions are unchanged from
the previous frozen preparation; their AST identities are recorded. The shared
validator additionally accepts the separately implemented temporal-only model
scope, which is not used by this kernel benchmark.

After the original learner closes successfully, run this bounded benchmark
with the same all-rank case-coverage audit. Only numerical-passing cases may
inform performance selection. Any candidate still needs a complete learner
state/memory/timing qualification before adoption. No learning follows
automatically from a kernel result.
