`go-core` implements strict Tromp–Taylor rules on runtime-sized square boards: padded storage, circular incremental chains, distinct liberties, captures and permitted multi-stone suicide, positional superko with exact collision checks, a reversible mutation journal, and direct terminal area scoring. `discard_undo` releases committed move journals while preserving superko history. The default allocation guard allows 1,048,576 playable points and is configurable; arithmetic/indexing limits are checked.

`go-gtp` provides a rules bring-up endpoint with random `genmove`, explicit alternating turns, two-pass terminal scoring, and undo. Standard single-letter GTP vertices limit this endpoint to 25 columns; the core supports larger boards. It is not yet a full learned playing engine.

`go-search` provides worker-owned PUCT trees, explicit neural requests carrying search/network identities, perspective-correct backup, exact completed-visit accounting and root restoration. Its baseline has no tree reuse, transpositions, virtual visits or neural cache. Tests cover adversarial toy trees, invalid/stale responses, terminal leaves and resource failures. The root evaluation is distinct from completed simulations. See the [DeepMind MCTS implementation](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/search.py) for related reference semantics; this implementation has its own explicit contracts.

`go-actors` owns persistent optionally pinned workers, multiple games per worker, Dirichlet sampling, observations, game records and terminal training rows. Move-limit truncations emit no training rows. Checkpoints restore full history, unfinished targets and RNG positions; tests compare every subsequent request and completed target after changing worker count.

`go-bridge` is a coarse PyO3/NumPy extension. Search and replay preparation release the GIL. Rust transfers feature/replay vector ownership into NumPy; neural replies are copied before releasing Python to prevent mutation races. `ops/build_native.py` builds locked frozen sources and writes a binary/source receipt consumed by the Python wrapper and pod launcher.

Scoring is explicit: `raw_area` remains the default, and `pass_alive_area` follows the pinned KataGo area adjudication. Ownership never mutates actual stones or superko history. The actor/search position uses the selected profile for terminal values, game outcomes and ownership labels. Native ABI 2 includes spatial ownership in completed rows, signed for the player to move; truncated games have no targets. Historical ABI 1 libraries remain usable by the GTP loader with raw scoring, while current actor bindings require ABI 2. See [scoring qualification](../research/studies/scoring_compatibility/README.md) and the adaptation's [license](go-core/KATAGO_LICENSE.md).

```bash
cargo test --workspace --release --locked
cargo test -p go-core --release million_transition_qualification -- --ignored --nocapture
cargo build --workspace --release --locked
target/release/go-bench 9 1000000 1 0 1
```

The independent oracle uses unpadded grids, flood fills, full copies, and exact position sets. Qualification checked 8,308,582 proposals, including 2,970,115 committed moves, across sizes 1, 2, 3, 5, 7, 9, 13, 19, 25, and 37, in 27.08 seconds on host 0. Fixtures cover ko, repetition-exempt passes, multi-stone suicide, area/neutral scoring, collision filtering, allocation guards, and history-preserving commit. The suite also samples full legal sets and randomized multi-move undo. Further SGF, symmetry, tactical, and external rules fixtures remain required.

The native benchmark has worker-owned boards and counters, deterministic seeds, warmup, and separate proposal/committed-move/mask counts. It measures a synthetic rules workload, not self-play throughput or playing strength. CPU affinity and topology must accompany scaling measurements.

Legality queries now use chain liberties and candidate position hashes without playing/undoing moves. Matching hashes still require exact packed-board comparison, so this optimization does not approximate positional superko. A frozen baseline/candidate comparison passed the full oracle qualification and measured 5.26×/6.41× improvements in the single-core 9×9/19×19 mask-heavy workloads. The [study](../research/studies/rules_qualification/README.md) retains specifications, measurements, limitations, and artifact identities.
