This clone tests a small convolutional visual encoder after reviewing the fixed endpoint of `visual_token_linear`.

The active encoder applies two 3×3 SAME convolutions, 22→64→64 channels, with per-point RMSNorm and SiLU after each. It keeps the parent’s overlapping 2×2 grid (36 visual tokens on 9×9), 34-layer width-768 causal trunk, SwiGLU width 2304, 12 query/4 KV heads, and tied policy head. New random draws occur after the parent’s existing parameters so unchanged weights initialize identically. There are no batch statistics or temporal convolutions.

The ablation retains the immutable data, full histories, exact episode and D4 draws, common AdamW schedule, one policy task, and 1,024-update endpoint. `linear_reference.py` is the exact pinned parent implementation used only for initialization qualification. The cache runtime experiment remains in its separate recipe and is not folded into this learning comparison.

`test_causal.py` checks an independent NumPy convolution/patch reference, inherited initialization, causality, cache/full agreement, all encoder gradients and globally weighted gradients. `qualify_budget.py` traces the complete cached decode including the encoder and compares it to both the linear parent and qualified KataGo CNN. Full TPU learning is gated on those checks and bounded small/full TPU qualification.

The numerical matching contract applies to 9×9, batch 128 and 128 past moves only. Learning results are a single-seed fixed-data screen, not a playing-strength or RL-efficiency claim. Selection rationale is recorded in `research/studies/visual_katago/linear_review.json`.
