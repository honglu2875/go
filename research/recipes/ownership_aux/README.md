# Terminal ownership experiment

Complete clone of `local_inference`: this folder owns model, loss, SGD momentum,
configuration, replay sampling, training and recovery. Rust owns all rollouts,
terminal labels and PUCT. No Flax, external weights, teacher evaluations or game
labels enter training.

The additional 1×1 spatial head predicts ownership in the current player's
perspective. Its pretanh output is trained with binary cross entropy on twice
its output and targets `(ownership + 1) / 2`, averaged over board and batch.
The control uses coefficient 0; the candidate uses 1.5, as in the
[pinned KataGo loss](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/python/katago/train/metrics_pytorch.py).
Neutral intersections have target probability 0.5. Inference returns policy and
value; ownership does not directly modify search. Both arms have identical
initial weights, shapes, L2 penalty and all other configured budgets.

`actors.scoring` explicitly selects `raw_area` or `pass_alive_area`. The latter
matches the pinned KataGo area adjudication with suicide allowed, safe and unsafe
large territories, non-pass-alive stones, and no tax. Board stones and superko
history are unchanged by adjudication. Outcomes, search terminal values and
ownership labels all use this same setting. Rows from truncated games remain
excluded. This is one component control, not a reproduction of KataGo's full
training procedure.

Ownership is stored in native completed rows and the replay ring, transformed
with policy/features under D4, and saved in full checkpoints. Native ABI 2 is
required. Each run must use its own source snapshot and qualified native build.

`smoke.json` is a 3×3 CPU integration/recovery check. `pod_smoke.json` and
`pod_9x9_bootstrap.json` are bounded integration configurations. Study protocols
and results belong in `research/studies/ownership_aux/`; passing an integration
check does not establish a strength or sample-efficiency improvement.
