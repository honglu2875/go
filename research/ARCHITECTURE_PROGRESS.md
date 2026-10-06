# Architecture research through October 6, 2026

The current research default is the 232,011,540-parameter dense causal visual
transformer, trained with AdamW at peak learning rate 1e-3 and 64 complete games
per update. The next registered ablation adds temporal mixture-of-experts
layers under those tuned settings. It has no completed learning result yet.
These are fixed-data learnability experiments; a strong Go engine and faster
reinforcement learning than KataGo remain goals.

## Current architecture

| Component | Specification |
| --- | --- |
| Board input | 19x19x22 spatial planes and 19 global features |
| Stem | 3x3 convolution, stride 1, 22 to 768 channels |
| Shared encoder | 24 unique blocks, applied twice; 20 ConvNeXt blocks plus 4 spatial-attention blocks |
| ConvNeXt block | 3x3 depthwise convolution, normalization, pointwise expansion 768 to 3072 and projection to 768 |
| Spatial attention | 12 heads, width 768, SwiGLU hidden width 1024 |
| Board token | Project each spatial location 768 to 16, flatten 361x16=5776, project to one 768-dimensional token |
| Temporal transformer | 18 causal layers, width 768; 12 query heads, 4 KV heads, head width 64; SwiGLU hidden width 2048; RMSNorm and RoPE |
| Policy readout | Reuses the 361 encoder feature vectors with the temporal query, contextual readout dimension 64, and a learned local spatial correction |
| Value head | Hidden width 256, three logits trained with signed-target cross-entropy |
| Auxiliary supervision | First encoder pass, weight 0.25; value objective weight 0.7 |

The [cloneable recipe](recipes/strong19_moe_batch64/README.md) includes the
complete dense and sparse configurations. Pure JAX owns model initialization,
forward computation, loss and optimization; no Flax or batch normalization is
used. Training skips wholly padded encoder chunks while preserving complete
game histories and the registered real-position stream.

## Completed comparisons on the current fixed corpus

The corpus has 3,200 completed games and 1,356,897 positions. Training contains
2,906 games / 1,233,366 positions; validation contains 152 games / 64,371
positions; 142 test games remain closed. Repeated training samples make
position exposures larger than the number of unique corpus positions.

At 128 updates of 128 games, all arms consumed the same 7,001,181 position
exposures, using AdamW, identical game draws and D4 augmentation:

| Architecture | Parameters | Validation policy KL | Value MSE | Learning hours |
| --- | ---: | ---: | ---: | ---: |
| BN-free KataGo-style CNN | 233,220,870 | 1.034096 | 0.225195 | 1.835 |
| Dense causal transformer | 232,011,540 | 0.842229 | 0.129620 | 2.362 |
| Temporal-only MoE | 317,001,492 | 0.844658 | 0.127651 | 2.370 |

The temporal MoE result is a near tie with dense, with 36.6% more total
parameters and matched active decoding matrix FLOPs. It replaces the temporal
SwiGLU FFNs with four independent experts, selects two per token, and gives
each expert hidden width 1024. The dense visual encoder stays unchanged.
Pallas grouped matrix kernels execute token dispatch without token dropping.

A separate 421M all-expert model improved policy KL by 4.17% at the shorter
64-update endpoint but worsened value MSE by 15.16% and took 2.245x the recorded
learning time. It was not extended to 128 updates. There is no inference
latency or MFU claim from these measurements.

## Dense LR and batch tuning

The LR screen retained 1e-3. Validation policy KL after the same 7,001,181
position exposures was 0.875002 at 6e-4, 0.842229 at 1e-3, 0.856248 at
1.22e-3 and 0.841816 at 1.5e-3. The highest rate's 0.05% gain did not pass the
registered 0.5% selection threshold.

At LR 1e-3, the batch comparison preserved the exact game/D4 stream and scaled
the LR and evaluation schedules with game exposure:

| Games per update | Updates | Validation policy KL | Value MSE | Learning hours |
| ---: | ---: | ---: | ---: | ---: |
| **64** | 256 | **0.728001** | **0.121956** | 2.437 |
| 128 | 128 | 0.842229 | 0.129620 | 2.362 |
| 256 | 64 | 1.024584 | 0.169125 | 2.431 |

Batch 64 improves endpoint policy KL by 13.56% and value MSE by 5.91% for
3.17% more learning time. Its last-three validation KL mean is 14.42% lower.
No sustained-overfit flag was observed in this batch screen. Smaller batches
receive more optimizer updates at fixed data exposure, so moment and per-update
weight-decay dynamics differ. This is not an isolated hardware-batching effect.

The [defaults](defaults/README.md) keep the exposure-based cosine schedule:
1024 updates, 80 warmup updates, end LR 3e-4, validation every 32 updates.
The 256-update screen is a prefix, not a completed annealing schedule.

## Current MoE ablation

The [new protocol](studies/strong19_moe_batch64/PROTOCOL.md) compares the audited
dense batch-64 control with two fresh temporal MoE arms: the previous router
balance coefficient and a three-times-weaker coefficient. Router z-loss,
architecture, initialization, LR, data and exposure remain fixed. Both MoE
arms have 317,001,492 parameters. Encoder-inclusive active cached-decode
matrix FLOPs are approximately 1.00000132x dense; actual dispatch and training
time are measured separately.

Full-size systems qualification precedes sequential 256-update learning runs.
The protocol requires complete state/replay audits, fixed validation and
training probes, router utilization diagnostics, and preserved checkpoints.
No outcome from these newly registered runs is asserted here.

## Interpretation and provenance

Learning time excludes compilation, evaluation and loading. Dense timing in
the matched tables is historical. The comparisons use one seed and fixed
teacher data; they do not establish trained playing strength, convergence or
reinforcement-learning sample efficiency. Earlier long-pair comparisons used
a different dataset and must not be merged into these tables.

Public source and configurations are sanitized. Historical source hashes refer
to private original experiments; obtain the referenced data, adapt deployment
examples and freeze a new snapshot to reproduce the procedure. Full model and
optimizer arrays, live logs, credentials and machine inventories are omitted.
