# Board attention architecture control

This is a complete clone of `score_utility`: Rust owns legal moves and search;
the recipe owns pure JAX parameters, forward passes, losses, optimization,
replay and recovery. No Flax or attention framework is used.

The attention trunk starts with a 3x3 convolution and group normalization, then
uses pre-layer-normalized attention/MLP blocks over all board intersections.
Each block has four attention heads and a GELU MLP at four times trunk width.
Residual branches are scaled by `1/sqrt(block_count)`. Attention dot products,
softmax, normalization and residual accumulation use float32 accumulation;
matrix operands follow the configured float32/bfloat16 compute type.

A shared learned relative-position bias ties offsets with the same unordered
absolute row/column displacement. Its triangular table supports square boards
through the explicitly configured maximum (26 here), and starts at zero. The
bias is D4 invariant; the unconstrained convolutional stem means the **whole
network is not exactly D4 equivariant**. Training uses the inherited random D4
augmentation. Each board size still needs its own compiled shapes.

CNN and attention variants use exactly the same policy, pass, win, ownership
and bounded-score heads, losses and SGD implementation. The draft 9x9 configs
use width 64 with four CNN blocks or six attention blocks, giving similar
parameter counts. Both currently use score utility factor zero, score loss
weight 1 and ownership loss weight 1.5. These choices are provisional controls,
not promoted hyperparameters. Matching parameters does not match FLOPs, latency
or optimization difficulty. A comparison must budget tuning equally, report
both samples and wall time, and retain real pinned KataGo evaluation.

The attention design is a research candidate, not a reproduction of
[ResTNet](https://arxiv.org/abs/2410.05347) or
[Chessformer](https://arxiv.org/abs/2409.12272). It evaluates global spatial
attention before adding a causal history decoder. It does not yet implement
the proposed expert/opponent heads or fused multi-ply speculative execution.

The inherited pilot filenames are draft configurations only. No architecture
pilot or strength claim has been registered or completed. CPU gradients/arithmetic and single/four-device updates passed. Fresh CPU
continuation matched 82 arrays and 26 subsequent games; fresh multi-host TPU
continuation matched 328 arrays and 443 subsequent games, plus every actor
state. These are 3x3 execution/recovery qualifications, not Go strength.
Receipts are in `research/studies/attention_control/qualification.json`.
