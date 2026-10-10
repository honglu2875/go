# Temporal fuzzy TopK

A cloneable pure-JAX fork of `strong19_moe_batch64`. Only the 18 temporal FFNs
change. The 768-wide, two-pass spatial encoder, causal masks, shared policy/value
readouts, signed-value objective, first-pass auxiliary supervision, optimizer,
and fixed-data replay remain unchanged. `dense.json` is the unchanged reference.

For `H = G*K` scores, partition into `K` fixed groups of `G`, choose each group's
largest score, apply ReLU, and sum its original dictionary row into the residual
output. There is no independent router or balancing/reconstruction objective.
Both projections have zero-initialized biases; norms/biases are not decayed.
UP uses standard fan-in initialization; DOWN uses fan-in and residual-depth
scaling. Non-FFN initial draws remain identical to the dense control.

| Arm | Group size G | Selected groups K | Features H |
| --- | ---: | ---: | ---: |
| Dense ReLU control | 1 | 3072 | 3072 |
| Fuzzy G2 | 2 | 2048 | 4096 |
| Fuzzy G4 | 4 | 1228 | 4912 |
| Fuzzy G8 | 8 | 683 | 5464 |

The logical active FFN decoding budget is `2*D*(H+K)`, compared with
`6*D*2048` for dense SwiGLU. The difference is under 0.07% for every arm.
The initial choicewise implementation executes `4*D*H` matrix FLOPs, plus
selection and diagnostics. It is not an efficient selected-row kernel. The study
reports both budgets, encoder-inclusive totals, compiled memory, and measured
cached decoding at global batch 128. Training has three temporal tokens per
position; inference has two. Padding and rematerialization must not be hidden.

The implementation lives in the importable `gozero.fuzzy_topk` module and uses
ordinary JAX autodiff through a rematerialized loop. A literal selected-row
oracle, BF16/FP32 gradient checks, causal cache checks, bounded-gradient checks,
and optimizer round trips are in `qualify_fuzzy.py`. Activity diagnostics exclude
padded tokens; a feature unused in a batch is not necessarily permanently dead.

Inspiration: [rig's pinned grouped dictionary implementation](https://github.com/honglu2875/rig/blob/5912ca24af3b6452bcffe272bd375bf726dbeb7d/rig/kernels/fuzzy_topk.py).
This implementation is independently expressed. Published rig text results do
not establish an advantage over Go SwiGLU or playing strength.
