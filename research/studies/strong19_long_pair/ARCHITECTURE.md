# Current 19×19 causal transformer

This is the 232,011,540-parameter transformer used by the longer common-AdamW
comparison. Its implementation is the cloneable, pure-JAX recipe
[`strong19_long_ce`](../../recipes/strong19_long_ce/). The exact scientific
configuration is [`transformer-config-001.json`](transformer-config-001.json).
The generated [parameter inventory](transformer-architecture-001.json) includes
every tensor shape and the original source hashes, checked against the active
immutable model. Flax is not used.

## Board encoder and one-token connector

Each observation contains a `19 × 19 × 22` spatial feature tensor and 19 global
features from the pinned KataGo V7 adapter. All convolutions have stride 1 and
SAME padding. The encoder retains a `19 × 19 × 768` feature map throughout.

| Component | Operation and weight shape |
| --- | --- |
| Stem, applied once | `3 × 3` convolution, 22 → 768 channels; weight `[3,3,22,768]`, bias, affine LayerNorm and GELU |
| Convolutional residual block | Depthwise `3 × 3`, weight `[3,3,1,768]`; affine LayerNorm; channel MLP `768 → 3072 → 768`, GELU; per-channel residual scale |
| Spatial attention block | Full attention over all 361 board locations; 12 heads of width 64; Q/K/V/output weights each `[768,768]`; row/column RoPE |
| Spatial attention MLP | Gated SiLU, `768 → 1024 → 768`; separate affine LayerNorm and residual scales |
| Block order per pass | Four repetitions of five convolutional blocks followed by one spatial attention block: 20 convolutional + 4 attention blocks |
| Shared refinement | Execute the same 24-block stack twice, sharing all stack weights between passes; residual scales initialize to 0.01 |
| Connector | Per-location projection `768 → 16`, flatten `361 × 16 = 5776`, project `5776 → 768`, affine LayerNorm |
| Global features | Project `19 → 768` and add to the board token with a learned type embedding |

The result is **one 768-dimensional soft token per board**, plus the original
361 spatial feature vectors retained for the policy readout. The connector
does not replace those vectors with a reshaped temporal latent. Its flattening
projection is specific to the configured board size.

Spatial attention is unrestricted within the current observed board. Temporal
causality is enforced separately by the history transformer. There is no
quadrant or raster-order mask on the spatial feature map.

## Causal history transformer

Normal inference processes `[board, action, board, action, ...]`. The policy at
a board token can use all preceding board/action tokens and its current board.
The current played action and all future frames are hidden from that policy.

| Setting | Value |
| --- | --- |
| Layers / width | 18 / 768 |
| Attention | Grouped-query attention: 12 query heads, 4 KV heads, head width 64 |
| Per-layer attention weights | Q `[768,768]`, combined KV `[768,512]`, output `[768,768]` |
| Feedforward | Gated SiLU; gate/up `[768,2048]`, down `[2048,768]` |
| Normalization | Pre-attention and pre-MLP RMSNorm, epsilon `1e-6` |
| Positions | RoPE, theta 10,000 |
| Capacity | 1,536 board frames; two normal inference tokens per frame |
| Arithmetic | BF16 matrix/activation paths with FP32 parameters and accumulations where declared in code |
| Execution | Splash attention for training/prefill; explicit cached attention for incremental decoding; rematerialized encoder and temporal blocks |

[`causal.py`](../../recipes/strong19_long_ce/causal.py) implements prefill,
the opening position and cached move advancement. Cached advancement processes
the previous action and the exact new board together, and returns the next
policy plus updated KV state.

## Policy and value readout

The policy has 362 outputs: 361 intersections and pass. The normalized temporal
latent is projected through the transpose of the shared `[362,768]` action
embedding. Each board intersection additionally receives:

1. A learned local projection of its 768-dimensional encoder feature.
2. A contextual score `dot(Wq h, Wk f_i) / sqrt(64)`, using the temporal query
   and that intersection's spatial feature.

The pass output uses the temporal projection. Legal-move masks are applied by
the policy loss and inference interface. This registered model uses the linear
bilinear contextual score; the optional extra query-refinement module is off.

The shared value head reads the normalized temporal latent through
`768 → 256 → 3`, with Mish between the two affine layers. Its signed prediction
is `softmax(logits)[0] - softmax(logits)[1]`, in player-to-move convention.

## Shared first-pass auxiliary supervision

Training obtains a draft board representation after the first encoder pass and
the full representation after the second. Both reuse the same connector,
temporal blocks, spatial policy parameters and value head. No extra encoder is
run to produce the readout features.

The packed training sequence is `[draft board, full board, action]` per frame.
Its mask gives both board readouts the same preceding **full-board/action**
history. The current full board is hidden from the draft; draft tokens never
enter another frame's history. Draft and full board tokens use the same
temporal rotary position. Deployment retains the normal two-token frame.

The current joint objective is:

```text
policy_loss = 0.75 * CE(full_policy, teacher_policy)
            + 0.25 * CE(draft_policy, teacher_policy)
value_loss  = 0.75 * CE(full_value_logits, q)
            + 0.25 * CE(draft_value_logits, q)
loss        = policy_loss + 0.7 * value_loss
q           = [(1 + y) / 2, (1 - y) / 2, 0]
```

Here `y` is the teacher's stored signed value. The target preserves its
expectation; it does not reconstruct unavailable teacher WDL probabilities.
This fixed-data ablation uses one teacher-policy head, shared between full and
draft outputs. Opponent-behavior prediction remains a separate research task.

## Exact parameter accounting

| Group | Parameters |
| --- | ---: |
| Encoder, including connector/global projection | 118,161,424 |
| Temporal transformer blocks | 113,273,856 |
| Policy readout and action embeddings | 378,625 |
| Value head | 197,635 |
| **Total** | **232,011,540** |

The shared second encoder pass adds computation without adding parameters.
The comparison's complete decoding budget includes both passes, connector,
temporal decoding and policy/value readouts. Training also pays for the draft
tokens and auxiliary objectives; deployment FLOPs do not describe training cost.

The [registered study](README.md) specifies AdamW, the common 512-update schedule,
frozen data, validation, and checkpoint audits. The architecture inventory was
generated by CPU shape inspection of original snapshot
`a1239d3c1d41b4765a13832b22525ef2f335d21e5eed0e3386d64307f42d0827`;
it did not initialize model tensors or allocate an accelerator. Historical source
hashes identify the original experiment; published deployment examples must be
configured and snapshotted afresh for a new run.
