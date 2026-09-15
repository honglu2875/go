# Fused exact-board speculative packet

`speculate.py` executes shallow autoregressive decoding and exact JAX Go state advancement, followed by full expert verification in one dispatched HLO graph. Expert-only and alternating expert/observed-behavior proposal roles are distinct; the target uses the expert head for both actual players. `resolve` implements maximal-coupling acceptance and residual replacement with independent random streams.

CPU distribution and exact native replay checks passed. The trained 233M TPU screen resolved only 1.69–1.95 moves per root at horizons4/8 and failed the amortized speed gate, including an optimistic calculation excluding CPU audit. This prototype has no continuous cache-repair loop or MCTS-equivalence claim. See the speculation registration, repair and result under `research/studies/visual_causal`.

Freeze each resolved configuration before execution. Existing snapshots and failed attempts remain immutable. Clone this complete recipe for a scientific intervention; shared environment and artifact helpers are imported from `gozero`.
