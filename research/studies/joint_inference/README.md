The joint 19×19 CNN and causal transformer now accept exact native V7 features
and return policy plus a signed value for the player to move. The preparation
recipe is `research/recipes/strong19_joint/inference.py`; shared replay and board
identity checks live in the importable `gozero.v7_replay` and `gozero.v7_state`
modules. This does not change the frozen larger-9×9 comparison.

`native-boundary-001.json` passed independent Rust/C++ board and legality
comparisons on 12,021 positions from 30 real training games, plus 150 pending
Rust search leaves. Native six-plane observations are used to audit identity;
the model receives KataGo's 22 spatial and 19 global V7 inputs. A complete
action history identifies a cache entry, so equal boards with different
histories cannot accidentally share temporal state.

`model-cpu-002.json` passed with small complete joint models on four simulated
CPU devices. The CNN passed 48 output comparisons and eight rejected-input
checks; the transformer passed 50 and eight. Cached values and policies agree
with independently evaluated full histories through branches, shortened
prefixes, passes and reordered requests. Maximum absolute output discrepancy
was 8.65e-7. Both models also completed two real Rust MCTS searches, validating
the network value's perspective, legal policies and native pending histories.
These randomly initialized small models establish interface correctness, not
playing strength or full-size numerical tolerances.

The first attempt, `model-cpu-001.json`, failed because the test requested search
diagnostics after `Game.finish()` had consumed the search. The rerun moved that
inspection before `finish()`. The failed receipt and log remain; the inference
implementation did not change between attempts.

The transformer retains one cached path per slot, rewinds to the shared prefix
for a sibling, and appends several supplied legal board/action steps in one
compiled scan. Prefix hits use stored outputs without a device dispatch. The
native worker persists, but still replays complete action histories to validate
them before encoding requested suffixes. This is cached batched evaluation of
known legal paths; speculative action generation and trace acceptance are
separate future work.

Full-size TPU memory, latency and optimizer qualification, a checkpoint export
and serving entry point, and actual KataGo matches remain open. No trained
19×19 checkpoint, MFU improvement or strength improvement is claimed here.

`pod-stream-staging-001.json` records copying the qualified host-0 native worker,
its C++ source and build receipt to the three peer hosts. all configured hosts passed
the same four-row protocol probe with byte-identical responses, including pass
and suffix cases. The worker is 768,912 bytes; probes ran on CPUs 59 and 119 and
started no persistent service. These are staged copies of one build, not four
independent compilations.
