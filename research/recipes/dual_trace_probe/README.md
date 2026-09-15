# Paired causal policy execution probe

This complete cloneable recipe tests the user's two-view, 2N-ply decoding
mechanism before adding a learner or search. Both player views and k independent
opponent forecasts execute in one JAX `lax.scan`, with per-branch KV retained on
device. Rust resolves exact legal moves and filters continuations against only
the actual prefix. It never selects a trace by future agreement. A wrong final
opponent forecast does not invalidate the other player's last actual move.

The pure-JAX transformer has a play head, value head and separate behavior head.
The behavior loss uses observed actions and cannot change the play trunk; the
play loss accepts separate policy targets. The execution probe uses untrained
weights and produces no MCTS targets. Historical action counts supplement the
causal move tape as a deliberately simple opponent context. Learned opponent
calibration and MCTS distillation remain subsequent experiments.

Own Gumbel draws use game, episode and absolute ply, independently of packet
boundaries or selected branches. Exact legal masking in Rust uses those same
draws and play logits. An illegal unmasked proposal is corrected; predictions
that no longer match cannot continue. Tickets bind generation, root, episode,
model versions and context versions. Malformed batches are rejected before any
game advances. Terminal games and capped games remain distinguishable.

The CPU probe compares three multi-ply modes against one sequential causal
control: the untrained behavior head, an oracle play-head predictor with
independent draws, and an oracle with shared draws. The latter is an explicit
unavailable-information upper bound. Exact per-game event prefixes, including
resets and terminal scores, are required. Accepted plies are dispatch diagnostics,
not a speedup or strength claim. HLO, parameters, events, config, native binary
and source identities are retained. This version prefills the whole actual tape
per packet and resolves the native batch serially.

Run `test_model.py` on CPU, freeze the recipe with `cpu.json`, build its frozen
native extension using `ops/build_native.py`, then invoke the frozen `train.py`
with `--config`, `--native-receipt` and a fresh `--output`. The conventional
entry-point name does not imply that this probe trains weights.

`cpu_root_mask.json` tests a known-root legality optimization against the updated
`cpu.json` control. Trace ABI2 returns tickets, tokens, lengths and a boolean
root mask. In the masked variant both views apply it only at decoding depth0;
Rust validates that declared sampler and still checks every real move. Raw
own-policy logits remain available for exact legality corrections at later
depths. The first qualification remains frozen with its original three-part
frame; new source/config/native identities distinguish this iteration.
