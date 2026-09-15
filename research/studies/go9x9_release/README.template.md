---
license: gpl-3.0
task_categories:
- reinforcement-learning
tags:
- go
- katago
- synthetic
- policy-distillation
size_categories:
- 10M<n<100M
---

# Go 9×9: fixed-teacher, mixed-strength games

The final 9×9 corpus contains **207,138 complete games and 19,598,695 pre-action labeled positions**. Generation stopped on 2026-09-14 UTC. Every published game passed full-history replay validation. No published games were removed, and none reached the 324-ply truncation cap. Unfinished in-flight work at shutdown was not published.

The strong teacher is **kata9x9-b18c384nbt-20231025**. Each game pits this teacher against one of eight KataGo checkpoints. Its color is selected deterministically from the game ID. The same teacher labels both players' pre-action positions. This is engine-generated behavior and teacher supervision, not human games or self-play from the research architectures.

Release identity: `{{RELEASE_ID}}`. Pin the Hugging Face commit as well as this identity when comparing models. `manifest.json`, `SHA256SUMS`, and the per-game indices identify the exact bytes.

## Targets and rules

The board is 9×9 with komi 7.5, positional superko, area scoring, multi-stone suicide allowed, no tax or button, no handicap, and `friendlyPassOk=false`. Native terminal outcomes use the pass-alive-area scoring implementation. The exact KataGo rules object and analysis settings are in `provenance/producer/data/katago.py`.

Behavior uses 16 search visits per move. In the first eight plies it samples a mixture of 75% normalized root edge visits and 25% raw policy; afterward it selects KataGo's first root move by reported order. There is no root Dirichlet noise or symmetry pruning. Complete causal move histories are supplied to KataGo.

`raw_policy` and `raw_value` are the fixed teacher's **raw neural outputs**, even when the played move came from a weaker opponent. Teacher search labels are valid only on teacher turns and only where `search_policy_valid` is true. On opponent turns, a one-visit teacher query supplies the raw targets. `actions` records actual behavior and must not be silently substituted for the teacher policy.

All neural values and scores use the **player-to-move perspective**: positive favors the player whose turn it is. `white_score` in game metadata is the separate terminal behavior result, positive for White, including komi. Terminal game outcomes are not raw teacher values. The teacher's strength here is limited by the stated inference/search budget; this corpus does not establish maximum-strength KataGo play.

## Splits and sampling

| Split | Games | Positions |
| --- | ---: | ---: |
{{SPLITS}}

Each opening family is the lexicographically smallest of the eight D4 transforms of the first eight actions, hashed using canonical compact JSON and SHA256. `int(family[:8],16) % 100` assigns 0–4 to test, 5–9 to validation, and 10–99 to train. There are **{{FAMILIES}} distinct opening families** and no family crosses a split. These are nominal 90/5/5 family buckets; repeated openings make the observed game proportions different. Shared later positions/transpositions are possible. Game IDs are unique, but this is not a guarantee of unique trajectories.

The snapshot retains every completed game. Faster opponent strata produced more games, so it is **not an equal-game mixture**. Use the opponent IDs and counts to register a balanced or weighted training view; keep the split assignment fixed and do not rebalance validation based on model results.

| Opponent index | Checkpoint | Games | Positions |
| --- | --- | ---: | ---: |
{{OPPONENTS}}

## Files and loading

`data/` contains 20 uncompressed tar shards of individually compressed, pickle-free NPZ games. The tar headers have fixed timestamps and ownership. `index/host-*.jsonl.gz` contains one row per game with its metadata, shard, member path, payload byte offset, payload size, and SHA256. Shards target 256 MiB of NPZ payload; tar headers/padding add overhead. The full release is approximately 5 GB. Reading does not require extracting 207,138 files.

Download to a filesystem with sufficient free space (for example `/dev/shm/go9x9`) and verify the checksums:

```bash
hf download quintic/go9x9 --type dataset --revision FINAL_COMMIT --local-dir /dev/shm/go9x9
cd /dev/shm/go9x9
sha256sum -c SHA256SUMS
```

Replace `FINAL_COMMIT` with the immutable Hub revision you intend to use. The included reader needs only NumPy:

```python
from read_games import games

for metadata, arrays in games('/dev/shm/go9x9', split='train'):
    board = arrays['stones']       # [plies, 9, 9], before each played move
    target = arrays['raw_policy']  # [plies, 82]
    value = arrays['raw_value']    # [plies], current-player perspective
    actions = arrays['actions']   # [plies], causal history in chronological order
```

The reader verifies each NPZ checksum by default. A game is loaded at a time; training batch construction, history features, D4 augmentation and balancing are intentionally left to a pinned training recipe.

## NPZ schema

Let `T` be game length. Board and action coordinates are row-major from the top left; action 81 is pass. Stone codes are 0 empty, 1 Black, 2 White. Black plays first and the player alternates each ply. No resign action is used.

| Column | Shape / dtype | Meaning |
| --- | --- | --- |
| `stones` | `[T,9,9] uint8` | Absolute pre-action stones |
| `legal` | `[T,82] bool` | Full-history native legal mask |
| `raw_policy` | `[T,82] float32` | Teacher raw policy, illegal entries zero, normalized |
| `raw_value` | `[T] float32` | Teacher raw signed winrate in [-1,1] |
| `search_policy` | `[T,82] float32` | Normalized teacher root edge visits; apply validity mask |
| `search_value` | `[T] float32` | Teacher searched signed winrate; meaningful search on teacher turns |
| `search_policy_valid` | `[T] bool` | Whether the teacher search target is eligible |
| `root_edge_visits` | `[T,82] int32` | Teacher root edge counts |
| `raw_score` | `[T] float32` | Teacher raw score when available; apply validity mask |
| `raw_score_valid` | `[T] bool` | Raw score field availability |
| `teacher_visits` | `[T] int32` | Requested budget: 16 on teacher turns, 1 on opponent turns |
| `actions` | `[T] int32` | Actual behavior moves |
| `metadata` | `[bytes] uint8` | UTF-8 JSON, never Python pickle |

Metadata includes `game_id`, `contract_id`, teacher/opponent SHA256, opponent index, expert color, host/worker/sequence identity, row count, terminal flag, terminal White score, opening family and split. The contract pins all checkpoint URLs and hashes.

## Reproducibility and validation

Every NPZ was audited at publication, then audited again after shutdown using the frozen producer environment. Checks covered full-history legality including superko, board states, legal masks, policy normalization and support, teacher query budgets, finite values, terminal outcomes, and split assignment. Additional release checks verified schema shapes, immutable game identities, artifact identities, nonnegative visits, unique game IDs, and disjoint opening families across splits. Shards were hashed after packaging and checked again after gathering from the four hosts.

`provenance/` includes each host's production config, stop receipts, final audit, and the exact relevant Python producer sources. The original native extension hash and NumPy version are recorded in `producer-snapshot.json`; its binary is not distributed here. KataGo's EIGEN binary SHA256 is in `contract.json`, built from commit `92ee95c0a4b25fec214da00951ab69e97e207729`. Concurrent search is not guaranteed to reproduce identical games if generation is rerun; this release preserves the actual generated bytes.

The repository's existing GPL-3.0 license declaration is retained. KataGo code and checkpoint sources have their own upstream terms: [KataGo](https://github.com/lightvector/KataGo), [network sources and licenses](https://katagotraining.org/networks/), and [9×9 teacher release](https://github.com/lightvector/KataGo/releases/tag/v1.13.2-kata9x9). Model weights are referenced by URL/hash and are not redistributed in this dataset.
