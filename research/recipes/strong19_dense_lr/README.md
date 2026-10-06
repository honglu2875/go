# Dense learning-rate screen

Exact numerical clone of `strong19_moe_comparison`, used without MoE for the
current 232M dense parent. The study varies only AdamW LR scale; the model,
objectives, precision, data and augmentation remain fixed.

See `research/studies/strong19_dense_lr/PROTOCOL.md`. The initial grid has
peaks 6e-4, 1e-3 (existing audited reference), and 1.5e-3. Each fresh arm
runs 128 updates of the unchanged 512-update schedule, with full validation
every 16 and a durable endpoint optimizer/RNG checkpoint.
