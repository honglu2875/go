# Causal visual observation/action model

This recipe owns a plain-JAX decoder, continuous board-patch tokenizer, separate
expert/behavior objectives, functional AdamW update, and bounded model profiling.
It imports no Flax or another research recipe. `train.py` is currently the pod
harness entrypoint for **qualification**, not an online Go training loop.

The large configuration has 18 layers, width 1024, 16 query heads, 4 KV heads,
head dimension 64, SwiGLU hidden width 3328, RMSNorm, temporal RoPE and learned
two-dimensional patch-position embeddings. Parameters and residuals are float32;
matrix operands and stored KV use bfloat16. Full-sequence attention uses JAX's
TPU Splash kernel; the CPU reference and cached block scorer use explicit XLA
grouped-query attention. Attention implementations need numerical qualification
on the actual hardware. `parameter_schema` counts the full tree without creating
233M parameter elements. Gradient rematerialization is explicit.

Observations contain absolute black/white stones and native side-to-play, signed
komi/area, consecutive-pass and legality planes. A seventh internal channel marks
on-board intersections. A 3x3 patch is flattened and linearly embedded, with no
image rendering or discrete whole-board codebook. On 9x9 each position contributes
nine patch tokens, one readout token, and one action token. On 19x19, padding gives
49 patches and 51 tokens per position. All attention is causal. The readout comes
after every patch of its own observation and before its action, so predictions
see the entire current board and earlier history, but never their target action.

The configured 512-position limit gives 5,632 tokens on 9x9 and 26,112 on 19x19.
Board-size support in arithmetic is not evidence that these full lengths fit or
perform well. Context is never silently truncated. Cache buckets can be smaller
than the configured limit; they must hold the supplied prefix and continuation.
Cache state is versioned, has per-game lengths, and flags invalid phases, capacity
overflows and stale model replies while preserving affected KV buffers.

The expert output is tied to the action embedding and learns MCTS policy targets.
Behavior has its own adapter and output and learns actual observed actions. The
default lets behavior update the shared trunk; a flag provides the stopped-gradient
control. Value, score and spatial ownership heads are also present. Target masks
are separate: an unfinished observed game need not have outcome or MCTS labels.
The recipe does not equate the expert/behavior log-probability difference to Q-V.

`score_continuation` accepts supplied pairs `(proposed action, next exact board)`
and scores all positions in one causal block. Root-action scores come from the
preceding root prediction. This is the target-scoring primitive for speculative
execution, with cached one-edge scoring as its numerical reference. It does not
generate future boards, choose an acceptance algorithm, certify MCTS decisions,
or implement an early-exit drafter. These are separate research interventions.

The qualification harness uses explicitly synthetic tensors. It records frozen
source/config identity, parameter schema, per-device batches, compiled memory,
prediction/cache comparisons, supplied-path timings and finite gradients. Its
timing excludes proposal generation, native transitions, queueing and acceptance;
it cannot establish rollout speedup or MFU. It writes no large untrained weights.
Scientific learning and throughput experiments require a separate registration
with real trajectories, retained failures and real pinned KataGo benchmarks.

The model and optimizer are cloneable by copying this directory. `fixtures.py`
exists only for qualification. `observations.validate_inputs` is the data boundary;
compiled forward passes assume its shape/value validation has been performed.
