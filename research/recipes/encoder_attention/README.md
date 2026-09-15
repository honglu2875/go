This candidate replaces four of 24 unique encoder block positions with
within-board attention, keeping two shared passes, width 768, one historical
board token, 18 temporal layers and the single expert-policy target. The
selected parent is the context-readout transformer, whose KL gain passed
both paired seeds (11.83% improvement in mean KL). The readout stays fixed
while this experiment changes four encoder blocks.

Blocks 6, 12, 18 and 24 use bidirectional attention over the 81 board points.
Each has affine LayerNorm, bias-free Q/K/V/O matrices [768,768], 12 heads of
64 channels, 2D RoPE split equally between row and column coordinates,
residual LayerScale initialized at 1e-6, and another affine LayerNorm plus
bias-free SwiGLU projections [768,1024], [768,1024], [1024,768] and LayerScale.
Both residual boundaries use BF16. Attention never mixes distinct boards.
The remaining 20 depthwise-convolution/GELU blocks and connector retain the
previous shapes. A nested scan executes four groups of five convolutions and
one attention block; the same parameters execute again in the second pass.

Each replaced convolutional block has 4,732,416 parameters; each new attention
block has 4,723,200. The selected configuration includes the previously
replicated 64-dimensional context readout: 231,181,121 parameters and
37.428121728 GFLOP per complete cached 9x9 move at global batch128 and
128 prior moves. The reference CNN costs 37.348255744 GFLOP. Training and
physical TPU padding/latency require separate accounting and qualification.

Fourteen CPU tests cover causality, cache equivalence/guards, rematerialized
and shared-pass gradients, direct spatial readout, context readout, distant
point communication, board independence and relative 2D position scores.
The initial arithmetic/test failures and their fixes are retained in the
study: use audited broadcast/reshape instead of the tile primitive, and use
spatially varying random inputs for an attention-gradient assertion (affine
ramps normalize to identical vectors and nearly zero query gradients).

Run full-shape TPU qualification on all history buckets before 1,024-update
learning. Compare identical actual game/D4 draws and validation population.
A >=1% endpoint KL reduction with <=15% initial decode slowdown triggers
paired second-seed replication. Do not combine an unreplicated gain. These
are fixed-data learning measurements, not playing-strength or MFU claims.

The completed first seed reached KL 0.403794 but narrowly failed that
latency screen. An explicit adaptive addendum then selected one independent
quality replication, requiring >=5% paired KL improvement in both seeds.
The second seed reached 0.412231; paired improvements were 11.00% and
7.52%. Both original latency screens failed, with ratios 1.152 and 1.154.
The attention model is therefore a selected research parent with replicated
validation improvement and a retained latency cost. See the immutable
records under `research/studies/spatial_followups/` and the completed
`runs/spatial-followups-attention-002/replication/REPORT.md`.
