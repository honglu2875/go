# Visual self-play and full Adam-state continuations

This clone owns the pure-JAX decoder, losses, optimizer and experiment drivers.
The 233M model implementation is identical to the trained D4 visual recipe.
`generate.py` uses persistent per-slot KV and native MCTS. Rust workers are pinned
to physical cores; Python still schedules per-leaf requests, and individual games
can move between worker threads. Fixed native ownership remains future work.
Expert search targets and observed behavior labels remain separate.

CPU and multi-host controlled checkpoint-boundary recovery passed for generation
and complete Adam-state forks. Forks preserve parameters, both moments, the
absolute optimizer step, every host sampler and D4 stream, and exposure counters.
No host-loss or mid-round kill qualification is established for this visual path.

The full generator produced 1,024 games: 1,015 terminal and nine capped, with
99,250 moves. The merger independently replayed every board, legality mask,
ending and outcome. A separate KataGo check reproduced 5,978 positions and 32
completed scores. Caps receive no outcome target and cannot enter expert
training. Generated collection holdouts are D4-grouped. Historical behavior and
validation/test data remain unchanged, with an explicit sparse-bucket fallback.

The paired 256-update continuations both restored the same full D4 checkpoint.
Their complete training-state audit passed. The fresh self-play arm did not
improve the fixed real-KataGo endpoint. See the study's online learning and
KataGo reports; this is one generation, not an established RL speedup or a
continuous production learner. `evaluate_collection.py` is a separately
registered post-hoc diagnostic on 213 generated games excluded from training.

Use frozen source/config snapshots and their pinned candidate/data closures.
The root pod controller stages those closures before distributed startup.
