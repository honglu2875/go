The first causal-model MCTS integration completed all four registered real KataGo games. All 380 boards and four final scores matched; there were no illegal moves, process failures or capped games. The student lost both colors against each checkpoint. This passes integration qualification and provides no evidence for a strength improvement.

| Opponent | KataGo visits | Student W/L/caps | Checked boards |
|---|---:|---:|---:|
| Historical level 3, b6c96 | 1 | 0 / 2 / 0 | 209 |
| Pinned official 9×9 b18c384 | 1 | 0 / 2 / 0 | 171 |

These are two empty-board color pairs from one trained student. Their conservative paired intervals span [0, 1]; four games cannot estimate strength or learning efficiency. The stronger checkpoint is the project's pinned 2023 9×9 evaluation model, not a claim to cover the latest KataGo training.

`eval/causal_gtp.py` loads the original frozen model implementation and verifies all exported parameters against its committed checkpoint. Native `Game.request_history(request)` exposes the actual root moves plus the selected hypothetical leaf path, bound to the pending search/network ticket. Rust remains authoritative for legality, superko and terminal scoring. Every leaf uses the expert play/value heads for the player to move. The observed-behavior head supplies no MCTS priors.

Each requested move uses 16 Gumbel simulations excluding root, with zero search noise and the pinned teacher-style search settings. In total the panel made 3,024 simulations, 3,197 neural evaluations and 16 terminal evaluations. Inference prefills the complete causal tape at every leaf. This is the correctness baseline for future caching, with no speculative search or throughput claim.

The game cap is explicitly 256 plies. Reserving all 16 possible search extensions fits inside the student's 329-token context. Startup rejects incompatible caps/search budgets; commands cannot silently truncate history or force a pass at the limit. Offline student training had no native environment, so inference declares and verifies a separate native source and binary. Board size, komi and scoring must match the hash-verified teacher dataset.

- Registered protocol [spec.json](spec.json), SHA256 `fd7bf662d0e524e5d86a7fed3a1c9fe53c5711a7fda88591cb7331e58911fb67`.
- Execution source `0279695ad801cd9c90bae5f42f8e61e1986c3ee1ee0f1c6f7bb29d3a90fcbd20`; complete recursive panel/config/model input closure was registered before execution.
- Native source `4e6b09be6ca0ae7a39771b2bba19a2ddeb684e6d213983cf2ffdb8e4de418e35`, binary `a0d579784ef92a801d4c930bfdf7dcf65ca7a556b33733b049cd8c07b6686aaa`.
- Raw panel, SGFs, both engines' transcripts and readiness identities: `runs/eval/causal-mcts-0279695a`.
- Audited result.json (external or omitted experiment artifact), SHA256 `8351c01fae6ee8cf95a12819f808267b5aa913af6ad4c0fd4748c8a1ea9d7e32`, from analysis source `c8b18d4a27e126101e941f1aa78feeb832e80f11d364b4ca8cee8b58cedce691`.

The CPU panel took 38.51 seconds. Native history tests independently replayed deeper PUCT and Gumbel leaves and compared every feature, including stale-request rejection and owned-array isolation. Seventeen native binding tests, six causal model tests, five input-closure tests and two context-limit tests passed. A separate historical board-model adapter regression compared nine genmoves across three real-game prefixes; actions, boards and every non-timing search statistic remained exact (`runs/qualification/causal-gtp-board-regression-0279695a`, result SHA256 `4fe17162ce17cd8ca7e7706f9b1b691c32cff3e27ffef9783be25dc52eff4511`).

Reproduce the audit with its frozen `eval/analyze_causal_mcts.py --artifacts-root /workspace/go --output <fresh-path>`. The failed first analysis used an incorrect capsule-relative path for the external registered protocol; it is recorded in `analysis_failures.json`. The corrected analyzer pins the external protocol hash. No game was rerun for that correction.
