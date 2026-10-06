# Single board-token attention pooling, stage 001

## Question and scope

Does a content-dependent single-token summary improve the selected causal
transformer when the encoder, width, temporal network, spatial move readout,
targets, optimizer, data draws and decoding compute budget are held fixed?

This is a fixed-data learnability experiment. It does not measure playing
strength, self-play sample efficiency, achieved MFU or speculative acceptance.
The prior CNN comparison remains a historical reference; this pair compares
two transformer input connectors. One paired seed cannot establish a small gain.

## Model

Both arms keep the two shared passes through the width-768 encoder (20
depthwise-convolution/MLP blocks and four full spatial-attention blocks per
pass), the 18-layer width-768 causal temporal transformer, 12 query/4 KV heads,
FFN width 2048, and rank-64 bilinear spatial policy readout. Each board produces
one temporal token. Full and first-pass auxiliary paths share the connector.
No batch normalization or extra policy/behavior objective is introduced.

The control projects each 19x19x768 feature grid to 19x19x16 and maps its flattened
5776 features to width 768. The candidate normalizes each grid location, adds
learned row/column embeddings, and applies a learned width-768 query split into
12 heads. Key/value projections have shape [12,768,64], output projection
[768,768], and the residual GELU MLP is 768→1728→768. A raw learned-query residual
and pre-MLP LayerNorm precede the existing final token LayerNorm, global-feature
projection and board-type embedding. The shared spatial readout still receives
the full grid; no reshape of the single token is used for move locations.

Because the learned query is independent of the board, its key contraction is
performed before the spatial dot product. Since each value projection is
linear, it is applied after that head's weighted spatial sum. In real arithmetic
this equals standard single-query cross-attention. FP32 outputs and all
derivatives are checked against a separately expressed projected-K/V reference.
BF16 reassociation is not claimed bitwise equal to that reference; the optimized
order defines the candidate on both training and inference paths. Key/query
factorization has the usual parameter redundancy of a constant query; raw
parameter matching does not imply equal effective function-space dimension.

All common initial weights retain their original RNG draws. New pooling tensors
use an independent folded key. The baseline configuration reproduces the
selected parent's model. Parameters are approximately 232M in each arm; the
machine-readable budget records exact tensor counts, including the connector.

## Budget convention

Count each dense/convolution multiply-add as two FLOPs. Include both encoder
passes, connector, previous-action and current-board temporal tokens, all
executed cached attention positions, spatial policy readout, and value head.
The opening uses one temporal token. Report multiple cache extents and include
the constant-query contraction once per board, a conservative upper bound when
the compiler hoists it across a batch. The candidate is required to differ by
less than 0.1% in both raw parameters and complete decoding matrix FLOPs.

These are analytical matrix/convolution FLOPs, not total hardware operations.
Normalization, bias/position additions, GELU, softmax and cache traffic remain
outside this conventional count and are disclosed, not called free. Training
also computes the shared draft connector and temporal auxiliary tokens, with
backward/rematerialization overhead; measured step and learning time are
reported separately. No unused parameters or dummy FLOPs are added for matching.

## Frozen comparison

- Dataset: the existing frozen 19x19 scientific release; 2,634 training games,
  1,091,214 training positions. Both arms use exactly the registered game draws
  and D4 symmetries, complete causal histories, and raw teacher policy/value labels.
- Global batch: 128 games, 32 per process, eight per device; lengths 512/768.
- Same pure-JAX AdamW: peak LR 0.001, final LR 0.0003, 40-update warmup,
  beta1/beta2 0.9/0.95, weight decay 0.01, global gradient clipping at 1.
- Same full/draft policy and repaired signed-target value objective, auxiliary
  weight 0.25 and value coefficient 0.7. No loss-target changes.
- Each arm starts fresh, seed 91312427. Run **256 accepted updates per arm**
  using an immutable **512-update cosine schedule** and an explicit stage stop.
  This is a continuation-ready midpoint, not a complete 512-update result.
- Order is attention then flat, sequential full-pod use. Estimated paired wall
  time is approximately 10 hours, to be updated from observed rates. Each arm
  has a bounded 24,000-second execution timeout; no unbounded retries.
- Evaluate at initialization and every 16 updates on the same 146 held-out games
  (60,284 positions), plus the same fixed 128-game training probe. Test targets
  remain closed. Verify evaluation identity and counts at every observation.

The primary readout is held-out policy KL at update 256, supported by family KL
and the last three validation means (224,240,256). Value MSE and family-weighted
value MSE are secondary. Retain complete training/validation curves and compare
the latest completed validation points at a common learning-time budget without
interpolation. Report overfit flags; do not choose a flattering endpoint or
alter this horizon based on observed loss. A winning candidate needs longer
continuation, another seed and ultimately KataGo matches before a strength claim.

## Execution and retention

Source/config snapshots, operator hashes, parent qualification, exact draw replay
and CPU checks are pinned before training. The first two accepted training
updates also exercise both real sequence lengths. They are part of the frozen
learning run, not discarded warmup optimization steps. Nonfinite or rejected
updates fail the run; a collective compiler-memory gate checks all processes
before each new training/evaluation executable is used, with a 31 GiB/device
limit. A failure cannot automatically advance to the next arm.

Reserve both endpoint checkpoints and peer copies before the first launch.
Uncompressed model plus AdamW state is approximately 2.784 GB per endpoint.
The checkpoint minimum-free-space reserve is explicitly 56 GiB for this study,
with a further 2 GiB growth allowance in preflight. This permits both endpoints
without deleting previous critical checkpoints or datasets. Disk remains for
small logs/manifests; no repeated full parameter exports. Each final checkpoint
gets an independent peer copy and full file/tensor hashes. Both copies are RAM
and are volatile; selected models need later durable promotion.

There is one checkpoint at the end of each 256-update stage. An interruption
before that endpoint loses that arm's in-memory progress; this deliberately
bounded storage choice is recorded. A full state audit covers parameters, both
AdamW moments, optimizer counter, RNGs, all-rank metrics, game draws, symmetry
draws, validation histories, and overfit observations before the next stage.
The controller writes live observations, completion reviews, numeric curves and
a final comparison automatically. Transient live-read failures do not poison a
successful final audit, as happened in the previous storage incident.
Every ten minutes it checks all hosts for at least 3 GiB disk, 56 GiB shared
memory and 24 GiB available system memory. Exhausted reserves, repeated monitor
failures or 30 minutes without an accepted update request cancellation through
the existing attempt-token supervisors. The next arm cannot start after a
cancellation. The startup timer also permits the initial compilation/evaluation.

The CPU development attempt 001 failed on an unsupported BF16 batched-dot
operation. Its failure record is retained. No accelerator learning was started
from that version; the compatible implementation has a separate qualification.
