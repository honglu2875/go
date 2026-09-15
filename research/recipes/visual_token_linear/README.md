This is the first causal transformer in the sequential fixed-data study.
Its trunk width is 768, equal to the qualified KataGo CNN. It uses 34 layers,
12 query heads, 4 KV heads, a 2,304-wide SwiGLU feed-forward layer, RoPE and
per-token RMSNorm. Neither the transformer nor its board encoder uses batch
normalization. JAX implements the model and optimizer directly.

Each exact V7 board observation becomes 36 overlapping 2x2 visual tokens on
9x9. Patch projections, 19 global features, row/column embeddings and token
types are learned. A readout token follows the board, then the recorded
action token. The next observation follows that action. Every layer has
causal attention over the complete recorded game prefix. The readout sees
its current board and earlier actions, but cannot see the action it predicts
or later observations. There is one tied action-embedding/policy head and
one cross-entropy objective on the same stored MCTS policy distributions as
the CNN. There are no behavior, value or auxiliary targets.

The complete model has 234,409,728 trainable parameters (+0.851% versus the
CNN, including its training helper). One cached move processes the pending
previous action, 36 new board tokens and the readout. At 128 prior moves,
9x9 and batch 128 it costs 37.238 GFLOP of dense multiply-add work per move
(-0.294% versus the CNN). Encoder, head and attention to all 4,901 actually
addressed keys are included. The separate unit-cost count of floating
operations is within 1% too; transcendentals are not claimed to have the
same hardware cost as additions. Cache allocation holds 512 moves but the
profiled graph reads the registered 129-move extent. Cache outputs must be
observed, and the compiled executable reports donation/alias and memory
statistics. Matching is specific to this reference history and board size.

The grid covers every board point and is closed under reflection. For 19x19
it uses 196 tokens; this shape is not FLOP-matched to the CNN. Other board
sizes and history lengths must be measured explicitly before deployment.

The fixed learning screen uses the CNN's exact dataset, whole-game splits,
episode and D4 draws, global batch of 128 sequences, AdamW schedule, 1,024
updates and validation cadence. The test split remains closed. Equal
decoding work does not imply equal full-sequence training work or latency.
Record both curves by update and by measured training time. The CNN keeps
its original same-target training helper, so this compares architecture
packages rather than isolating attention as the sole changed operation.

CPU checks cover independent attention/patch arithmetic, causal non-leakage,
ragged cached/full equivalence, invalid-cache rejection, encoder gradients
and the globally weighted objective on four devices. Small and full TPU
qualification must pass before training. The full profile compares cached
policies with full-sequence Splash attention on real game prefixes.

Training and prefill use 512x512 Splash tiles, selected by a bounded kernel
timing and output/gradient qualification before the first learning screen.
This preserves the architecture and cached-decoding arithmetic; the latter
uses transparent dense attention. Training buckets with 4,864, 9,728 and
14,592 tokens pad attention to 5,120, 9,728 and 14,848 tokens respectively.
The earlier 128x128 kernel timings and full-model qualification are retained.

All runs use immutable source/config snapshots. The inherited KataGo
reference and dataset tools are retained to make the recipe self-contained.
Encoder interventions are cloned only after reviewing this arm's results;
see research/studies/visual_katago/ for registration and evidence.
