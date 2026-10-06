# Varied-batch runtime qualification

Registered before executing the new batches. Screen 001 remains **rejected**
under its original coordinate-wise state criterion. The conditional diagnostic
passed: including versus skipping padded chunks in one compiled executable
gave exactly equal parameters, both AdamW moments and scalar metrics after two
updates on all four ranks.

The original-versus-conditional discrepancy already appears with every board
computed. At update one, first-moment relative L2 differs by 0.0111%, while
the largest parameter discrepancy is 4.53e-5. The affected coordinates are in
three weight tensors; their near-zero gradients are sensitive to AdamW's
normalization on its first update. This motivates checking gradient/moment
direction and magnitude separately from parameter coordinates. It does not
retroactively make the original screen pass. The observed difference is in
compiler-generated conditional execution; no individual primitive has been
identified as its source. JAX documents that algebraically equivalent compiled
programs can have different floating-point results in its
[numerics FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).

## Fresh benchmark batches and fixed scientific scope

Use original registered draw turns 3, 4, 9 and 13: the first two unused draws
of each length, 512 and 768. Replay the preceding sampler and augmentation draws
exactly. These batches were used in the historical learning comparison, but
have not been used to measure the new runtime. They are not a new validation
dataset or a learning-quality test. Each variant starts from identical initial
parameters and AdamW state, then takes four consecutive updates on these
batches. Model, original optimizer schedule, all objective coefficients and
complete causal histories are fixed. All numerical model source bytes match
the CPU-qualified diagnostic; only the benchmark driver changes.

## Prospective acceptance

- Original dense, conditional dense and conditional skip execute serially.
  Conditional dense and skip reuse the same executable for each length.
- Require exact equality of every parameter, both moments, optimizer counter
  and scalar metric between the two conditional paths at every update/rank.
- Independently compare both conditional paths with the original executable.
  Keep the original scalar-metric bounds (rtol 0.005, atol 0.0001). Require
  relative L2 at most 1e-4 for parameters, 1e-3 for first moments and 0.005 for
  second moments, both overall and in every semantic model group. First-moment
  cosine must be at least 0.99999 where defined; counters must match exactly.
  Record every leaf and the original coordinate-wise gate, even if it fails.
- Require all updates accepted and finite, identical initialization hashes,
  exact registered draws, unchanged configuration and 232,011,540 parameters.
- Enforce compiled peak below 31 GiB across all ranks using overflow-safe KiB
  collectives. Keep only one input batch resident on the device at a time.
- Require at least 5% lower median maximum-host update latency in **both**
  buckets relative to original dense execution. Two samples per bucket remain
  a short performance qualification, not a precise long-run throughput estimate.

A pass selects an optional execution path for subsequent research. It does
not establish bitwise identity with the historical compiler, playing strength,
MFU, or unchanged long-run learning curves. Existing immutable learning results
and checkpoints are not altered. No checkpoint arrays or large profiler traces
are persisted by this qualification. One bounded attempt runs at a time.
