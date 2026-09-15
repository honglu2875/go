# Trained paired causal execution

This cloneable recipe executes the final, checkpoint-verified causal student
from `causal_distillation`. Its play head distilled native MCTS policies; its
separate behavior head learned observed actions. This execution study performs
policy sampling and exact native legality correction, with no MCTS decisions.

Both player views and independent forecast samples execute within one JAX scan.
Rust resolves legal actions and accepts continuations only when they match the
actual prefix. Own random draws depend on game, episode and absolute ply, so
packet boundaries and branch selection cannot change them. The full trained
weight hash, checkpoint lineage, model implementation, configuration, HLO,
native library and per-game event streams are recorded.

`cpu_greedy.json` and `cpu_stochastic.json` fix own-policy temperatures to 0 and
1 respectively. Each compares four conditions with a one-ply sequential control:

- Trained behavior forecasts, four independent samples per view.
- Initialized behavior parameters with the trained own policy held fixed.
- Known other-player policy probabilities with independent forecast draws.
- Joint continuation of known self-play policies using their actual draws,
  with one sample per view.

The last condition is implementable when both self-play policies and samplers
are available. An external opponent's future randomness and unknown policy are
not available. Neither condition automatically preserves decisions made by
running MCTS at every future root.

Every mode must match the sequential per-game event prefix, including terminal
scores and resets. Cached append and full prefill can differ numerically in
bfloat16; any changed action fails this exactness qualification and is retained.
The four speculative conditions use the same fixed number of packets. The
sequential control runs far enough to compare their longest event streams.
Consequently elapsed times are diagnostic, not a matched-work speed comparison.

Counters include accepted depth, own-policy legality corrections, distinct
first forecast actions, padded prefill and append token slots, bytes and host
dispatches. They do not measure all forecast legality, TPU MFU, CPU scaling,
Go strength, or end-to-end learning efficiency. Full actual history is prefetched
anew for every packet; the native batch resolver is serial in this prototype.

Run `test_model.py` on CPU, freeze one of the two configs, build that snapshot's
native extension, and invoke its `train.py` with `--config`, `--native-receipt`,
`--artifacts-root`, and a fresh `--output`. This conventional entry-point name
does not imply that this recipe trains weights.
