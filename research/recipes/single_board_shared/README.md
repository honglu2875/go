This clone implements the approved single-board-token weight-sharing study.
The encoder uses stride-one residual blocks with depthwise 3x3 spatial mixing,
affine LayerNorm, GELU and 4x pointwise channel expansion. Its 24 or 16 distinct
blocks are reused across two or three passes with full gradient accumulation.
LayerScale starts at 1e-6, a declared initialization choice. The residual stream
is BF16 at each block boundary; checkpointing recomputes from these exact inputs.

A 768-to-64 pointwise bottleneck and a whole-board linear projection produce
one width-768 embedding. The 9x9 connector is explicit: these configurations do
not claim unchanged weights can handle another board size. A causal transformer
processes [board, action] pairs and predicts the action from the board output.
There is one tied policy head, the same fixed teacher targets and no return,
value or opponent auxiliary objective. Current actions remain causally hidden.

The first allocation has 24 shared encoder blocks, two passes and 18 temporal
blocks. The next has 16 shared encoder blocks, three passes and 24 temporal
blocks. Both use width 768, SwiGLU width 2048, 12 query and four KV heads.
Total unique parameter and complete decoding budgets must pass the existing
one-percent checks against the frozen CNN at 9x9, batch128 and 128 past moves.
Changing allocation also changes weight sharing, which must remain explicit.

The scientific training configs retain the previous seed, exact game/D4 draws,
128 complete games per global batch, 1024 updates and full validation. Learning
rates are candidate peak AdamW rates with the established 64-update warmup and
cosine decay to 0.3x peak. The test split remains closed. Checkpoints use the
ordinary persistent filesystem writer at every configured boundary; temporary
RAM-only checkpoints are not used by these configs.

Before a full run: test_shared.py validates causal and cache semantics, gradient
accumulation across shared uses, and rematerialization. qualify_shared.py checks
the actual parameter schema and traced decoding/training graphs. A four-update
full-model TPU probe must exercise all three training buckets, validate cached
inference, finite distributed learning and a durable replicated checkpoint.
Frozen source/config snapshots and every failed attempt remain separate.

The first [completed study](../../studies/visual_katago/shared_encoder_study_20260913.md)
contains both allocation runs at peak LR 3e-4 and an LR-only 1:1 follow-up at
6e-4. Final validation KL was 0.628923, 0.631134 and 0.564571 respectively.
The higher LR improved the unchanged 1:1 model by 10.2%. Its measured learning
time was 20.22 minutes, with trained cached decoding at 8.72 ms for batch 128.
It remains behind C128 and the CNN on endpoint KL. All three full runs and
both full-shape qualifications passed; complete checkpoints are persistent.
No further run is queued, and these results do not establish Go strength.

For reproduction, use the study's frozen configuration/snapshot links.
The editable clone also contains `report_comparison.py`, `compare_lr.py`
and `inspect_refinement.py` for audited comparisons and checkpoint diagnostics.
