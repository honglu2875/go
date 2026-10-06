# Sparse experts at matched active compute — 2026-10-04

Authorized autonomous window: approximately 13:49–17:49 UTC. This is an initial
fixed-data learning and systems screen, not a playing-strength claim. Preserve
all prior frozen studies and the completed 256-update dense comparison.

## Scientific contract

Use the recovered, immutable 19x19 mixed-strength dataset (manifest
8e1ab17423f083137367adec93d215d7dcea2440dfafa1c448f0cea834470855), the same seed,
128-game global batches, complete histories, D4 augmentation, AdamW schedule
(512 updates, 40-update warmup, peak 1e-3), signed value objective, and fixed
validation population. Never inspect test labels. Reference: the completed
flat-connector dense run's matched prefix, after dense-clone equivalence checks.

Replace the encoder and temporal FFNs with four independent experts, choosing
two per token. Each expert has half the original FFN hidden width: 1536 in the
convolutional encoder blocks, 512 in spatial attention blocks, and 1024 in the
temporal transformer. Keep width 768, shared two-pass encoder, flat connector,
spatial readout, normalization and all non-FFN initial draws unchanged. Total
parameters may grow; match active matrix FLOPs, including routers and encoder.

Dropless routing is required. Capacity truncation would introduce batch/future
dependence into causal decoding. FP32 routing selects and stably groups tokens;
BF16 grouped matmuls execute only selected assignments. Use JAX's bundled TPU
Pallas grouped kernels after forward/backward qualification; CPU ragged-dot is
the reference. Expert parameters are initially replicated over the existing
data-parallel mesh. This avoids expert all-to-all but increases optimizer and
gradient communication; measured time and memory decide practicality.

## Sequential gates and ablations

1. Check independent expert oracle, all gradients, empty/skewed groups, padding,
   batch independence, causal/cached decoding, bounded/materialized loss,
   distributed normalization, optimizer decay, and checkpoint continuation.
2. On the TPU, compare grouped kernels to reference BF16 matmuls and gradients,
   profile balanced and skewed groups, then qualify a complete finite update
   with the existing 31 GiB compiled-memory guard. Count routing and sort cost
   in measured latency; theoretical active FLOPs are not achieved MFU.
3. Register a short learning horizon before seeing learning results. Prefer 64
   updates if measured runtime allows, otherwise a 16-update multiple matched
   to a recorded dense endpoint. Preserve the full 512-update LR schedule.
4. First screen: balancing coefficient 0.01, router z coefficient 0.001. If
   healthy and time permits, run a same-seed screen with balancing 0.001.
   Keep architecture, active FLOPs, initialization, draws, LR and horizon fixed.
   Qualify and interpret each pass before scheduling the next. No broad sweep.
5. Report policy KL, signed-value MSE, train/validation separation, positions,
   wall time, compiled HBM, router entropy/load/skew, zero dropped tokens,
   and checkpoint/audit status. A short warmup screen cannot establish final
   convergence or superiority to KataGo. Longer comparisons remain follow-up.

## Storage and closure

Use /dev/shm for dataset, transient compilation and staging. Preserve durable
full optimizer state and all-rank RNG state on disk with a verified disk peer.
Before retiring any historical owner payload, preserve the complete bundle on
two checksum-verified disk peers and retain source metadata. Bound new full
checkpoints and reserve disk space before launch. Never prune active logs,
latest checkpoints, datasets, unrelated projects or unverified sole copies.
Leave exact snapshot/config/source hashes, immutable attempt outcomes, receipts
and a clear handoff by the end of the window. Failed qualifications remain
recorded and cannot become successful learning evidence.

## Primary references

- V-MoE, Google Research (2021): https://arxiv.org/abs/2106.05974
- ST-MoE, Google Research (2022): https://arxiv.org/abs/2202.08906
- MegaBlocks, Stanford/MIT/Databricks (2023):
  https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf
- JAX ragged-dot contract:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html

These motivate sparse capacity, router stability and dropless execution. Their
hardware speedups are not evidence of a speedup for this Go implementation.
