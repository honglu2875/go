The approved one-board-token, shared-encoder study is complete: two allocation runs followed by one LR-only follow-up, each trained for 1,024 updates. All five TPU attempts, including two full-shape qualifications, passed.

The 1:1 allocation at LR 6e-4 is the best shared model tested. Doubling LR reduces its final validation KL by 10.2%. Relative to C128, it trains 2.16x faster and its warm neural decode is 9.60x faster, but final KL is 8.2% higher. The CNN remains the accuracy reference. Moving from 1:1 to 1:2 did not improve KL at LR 3e-4; the small top-1 difference went in the other direction.

| Model | Parameters | Peak AdamW LR | Validation KL | Top-1 | Learning min | Warm decode ms |
|---|---:|---:|---:|---:|---:|---:|
| KataGo-derived CNN | 232,431,872 | 1e-03 | 0.420361 | 0.6666 | 31.98 | 16.26 |
| C128 encoder + transformer | 234,281,728 | 3e-04 | 0.521731 | 0.6268 | 43.62 | 81.96 |
| Six-layer encoder + transformer | 232,551,168 | 3e-04 | 0.511634 | 0.6289 | 40.44 | 72.58 |
| Shared 1:1, control | 231,118,912 | 3e-04 | 0.628923 | 0.5920 | 20.22 | 8.67 |
| Shared 1:2 | 231,017,536 | 3e-04 | 0.631134 | 0.5934 | 20.38 | 9.55 |
| Shared 1:1, higher LR | 231,118,912 | 6e-04 | 0.564571 | 0.6109 | 20.22 | 8.54 |

Decode times are initial-weight medians, matching the historical measurement protocol: global batch 128, 9x9 boards and 128 prior moves. The best model also passed cached/full inference checks after training at 8.72 ms. These are warm neural graphs, including the encoder and pending action; CPU rules/features, transfers and pointer reset are excluded. Learning time excludes compilation, evaluation and checkpoint writing. This does not measure full-rollout throughput or MFU.

*Figure omitted from the source export: Learning curves by update and measured learning time. Recreate it with the study plotting script.*

Both new models use one width-768 observation embedding and one action embedding per move. A causal transformer predicts action_t from the output at observation_t, before action_t is visible. There is one tied policy head and no behavior, value or return-conditioning objective in this ablation.

The encoder starts with a stride-one 3x3, 22-to-768 convolution. Each shared residual block has a stride-one 3x3 depthwise convolution, affine LayerNorm, pointwise 768-to-3072, GELU, pointwise 3072-to-768, and learned channel scaling. LayerScale starts at 1e-6. The residual stream is BF16 at block boundaries, with full gradient accumulation through every shared pass. The connector applies 768-to-64 channel compression, flattens 9x9x64, and projects 5184-to-768 once. All new models are batch-normalization-free.

| Allocation | Unique encoder blocks x passes | Temporal layers | Encoder + connector parameters | Temporal parameters | Complete decode GFLOPs/move |
|---|---:|---:|---:|---:|---:|
| Shared 1:1, control | 24 x 2 | 18 | 117,780,544 | 113,273,856 | 37.2676 |
| Shared 1:2 | 16 x 3 | 24 | 79,921,216 | 151,031,808 | 37.4281 |

Both execute 48 spatial blocks. Temporal layers use width 768, 12 query heads, four KV heads, head width 64, SwiGLU width 2048, RMSNorm and RoPE. The fixed 9x9 connector does not claim weight compatibility with other board sizes. Encoder/connector work is included in the traced budget. Total parameters and complete decoding arithmetic are within 1% of the frozen CNN reference (232,431,872 parameters; 37.3483 GFLOPs/move). Training arithmetic is separately recorded and is about 19% below the CNN reference; decoding equality does not imply equal training work.

The compiler-accounted decode memory is 1.404 GiB/device for 1:1, 1.610 GiB for 1:2, and 16.749 GiB for C128. This is arguments + outputs + temporaries - aliases, not a measured peak-memory trace. The learned residual-scale median magnitude rises from about 0.0080 at LR 3e-4 to 0.0205 at 6e-4. Parameters are learning beyond initialization; this alone does not establish that repeated passes improve prediction.

The data are the same fixed weak native-MCTS teacher samples used in the earlier comparisons, not new self-play or KataGo expert labels. There are 9,466 training games / 836,486 positions, and 1,170 validation games / 102,339 positions. Each full run consumes exactly 11,469,333 position exposures, using 128 complete games per update and identical game/D4 draws. AdamW beta=(0.9,0.95), epsilon=1e-8, weight decay=0.01, gradient clip=1, 64-update warmup and cosine decay to 0.3x peak are retained. The CNN reference retains its existing FSON/Mish and training helper head; the new models use a single policy objective. All are compared on the main policy KL against the same validation targets.

This is a single-seed, finite-horizon learnability comparison. The allocation experiment changes encoder uniqueness, sharing and temporal depth together; it does not isolate weight sharing. The LR contrast verifies identical numerical source, library/lock hashes, architecture, initial weights and samples. The test split remains closed, and these runs do not establish playing strength, convergence or RL improvement.

Six CPU model tests cover causality, cached/full inference, cache guards, rematerialized gradients, accumulation across shared passes and the fixed-board/BF16 contract. Abstract schemas and traced arithmetic passed. Each allocation completed full-batch TPU qualification for all 128/256/384 training buckets. All full runs passed finite-update, exact-draw, replica, persistent-checkpoint and trained-cache audits. The LR audit also rejected an allocation change mislabeled as an LR intervention.

| New full run | Frozen configuration | Independent audit |
|---|---|---|
| Shared 1:1, control | 4761ce55 (external or omitted experiment artifact) | audit (external or omitted experiment artifact) |
| Shared 1:2 | ba7333c3 (external or omitted experiment artifact) | audit (external or omitted experiment artifact) |
| Shared 1:1, higher LR | c83cca0a (external or omitted experiment artifact) | audit (external or omitted experiment artifact) |

All new checkpoints, including optimizer and RNG state, are on persistent storage. The five attempts used 22.458 recorded attempt chip-hours; no further job is queued. Historical six-layer-encoder metrics and audits remain available, but its checkpoint arrays were lost in the earlier RAM-storage interruption, as documented in the preceding handoff.

The next useful isolated experiment would test LR 1e-3 on the 1:1 allocation. The current evidence supports keeping its speed benefits as a research reference; it does not support production promotion or calling the LR optimal.

Implementation: cloneable recipe (external or omitted experiment artifact). Evidence: machine-readable study index (external or omitted experiment artifact), complete comparison (external or omitted experiment artifact), LR contrast (external or omitted experiment artifact), SVG figure (external or omitted experiment artifact). Regenerate this summary with `python3 research/studies/visual_katago/shared_study_report_20260913.py`.
