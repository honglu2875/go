The comparison charges one complete neural move decode. The reference
operating point is a 9x9 board, global batch 128, and 128 completed prior
moves. This point is explicit because CNN inference cost is effectively
constant in history length and causal attention cost is not. Also report
histories 0, 32 and 256 and the position-weighted cost over the dataset.
Do not label the entire history curve or 19x19 inference equi-FLOP from a
match at the reference point alone.

The CNN reference performs 37,348,255,744 dense multiply-add FLOPs per move.
Its 232,431,872 trainable parameters include the encoder and the
training-only helper; 232,134,784 parameters participate in inference.
These two parameter counts must remain distinct in subsequent tables.

The transformer's charged decode includes the current board encoding,
new visual tokens, policy readout, and amortized action-token processing.
Its cache updates are observable outputs. A teacher-forced qualification
may append the actual played action, but the predicted policy for that
move must be independent of that action. Report the additional sampling
operation separately if a benchmark selects its own action.

Distinguish allocated KV capacity, attention key extent, and number of live
keys. Masking unused cache entries does not remove their matrix arithmetic
in a dense kernel. A 512-move allocation cannot be charged as 128 moves
unless the executed attention really reads a bounded slice or skips the
unused blocks. Count query/key padding performed by the selected kernel.
Observe the returned cache and record alias/memory information so XLA
cannot remove supposedly benchmarked action processing or cache writes.

Count all encoder convolutions/projections/pooling, trunk matrices,
attention score/value products, embeddings with arithmetic, and the
policy head. Dense FMA is two FLOPs. List reductions, nonlinear functions,
normalization, gathers and cache writes separately; transcendental counts
are not converted into invented hardware-equivalent FLOPs. Record
prefill, neural decode, CPU feature generation and data transfer costs
separately. Complete inference latency and throughput ultimately matter
in addition to the arithmetic match.

The primary parameter and dense-decode arithmetic tolerances are 1% of
the CNN reference. Architecture selection to satisfy these constraints
uses parameter schemas and operation counts before training. Preserve
the first valid configuration and its cost receipt in each iteration's
registration; do not retune the budget after seeing validation results.

XLA's cost estimate for the scanned CNN undercounts loop execution. The
independent JAX graph counter expands static scan trip counts and matches
the analytic count exactly. Keep the raw compiler estimate for inspection,
but do not use it to match this CNN against an unrolled transformer. For
opaque attention kernels, use a separately qualified arithmetic model of
the actual executed blocks and compare cached predictions to a transparent
reference implementation.
