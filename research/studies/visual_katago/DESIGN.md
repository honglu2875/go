This study proceeds sequentially. First qualify and train a KataGo nested
bottleneck CNN on the existing expert policy samples. Inspect its learning
curve and numerical diagnostics before running the equal-width causal
transformer. Choose each subsequent patch/encoder intervention from the
previous result, and register it before training. There is no simultaneous
architecture sweep and no test-set selection of encoder ideas.

The archived source manifest is
`9a4ef7a1034897f0991e197b5c92f5303d777a5cfb8a23bfa6dcbbd3b7c8f0b6`.
It contains 11,871 expert games and 1,047,681 expert positions; training has
9,466 games and 836,486 positions. Keep the original whole-game splits and
MCTS policy distributions. No new games, behavior targets, value targets,
KataGo teacher logits, or pretrained weights enter this screen. Reconstruct
KataGo V7 spatial/global inputs from the same past moves for both arms and
verify every pre-action board and legal mask against the original cache.

The CNN reference is KataGo `b40c768nbt-fson-mish-rvglr-bnh` from source
`92ee95c0a4b25fec214da00951ab69e97e207729`. Its trunk width is 768, its
bottleneck width 384. Preserve its fixed scaling, nested residual blocks,
global pooling, Mish, initialization, and training-only batch-normalized
policy helper. Retain only current-policy channel 0, with the same target
for main and helper outputs. This is a policy-only architecture adaptation,
not a reproduction of KataGo's complete multitask training recipe. The
official PyTorch code is used in an isolated qualification environment only.

Our transformer and all proposed image encoders are free of batch
normalization. The first transformer uses RMSNorm and a simple learned
patch projection. Later convolutional encoders use fixed residual scaling
or normalization over channels within one example. No cross-example batch
statistics or running averages are introduced into our architectures.

Matching contract: trunk width 768; count every trainable parameter,
including board encoder, embeddings, policy readout and any training-only
helper. Also report inference-active parameters separately. Count FMA as
two FLOPs. The primary decode budget is one move at 9x9, batch 128, after
128 completed prior moves with a populated KV cache. Include the new board
encoder, all visual/action/readout tokens, projections, attention over the
actual cache capacity, and policy output. Report elementwise operations,
padding, cache bytes/updates, and compiler estimates separately from dense
multiply-add arithmetic. Cache updates must be observable outputs of the
benchmark. A warm, fixed prefix must not hide repeated board encoding.

Require absolute relative differences of at most 1% in both total
parameters and algorithmic cached-decode FLOPs before calling an arm
matched. Verify compiler cost and execution; disclose physical padding
overhead and any disagreement instead of redefining the budget after
learning. Report the history-length curve (0, 32, 128, 256), prefill cost,
training FLOPs and wall time separately: equality at one history length
does not imply equality everywhere or equal latency. If discrete viable
architectures cannot meet both limits, record that before training and
compare under explicitly separate parameter/compute controls.

Use pure JAX and the locked uv runtime. Freeze source/config/data hashes
before every qualification and training attempt. Preserve failures and
resource receipts. Log the exact episode/D4 draws, accepted updates,
gradient clipping, semantic parameter-group gradient/update norms, held-out
policy KL/CE, cumulative training FLOPs, and compile/train/evaluation time.
Reserve the test split for frozen endpoints. Later recipes are cloned;
results already published by the preceding visual study remain unchanged.

No full-scale production training is authorized by a successful offline
screen alone. This study measures learnability on fixed weak-teacher data;
it does not establish self-play improvement rate, MFU, or strong Go play.
