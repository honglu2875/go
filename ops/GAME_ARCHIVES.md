# Completed-game archives

`gozero.game_archives` and `ops/archive_games.py` provide reversible archival of
closed training runs. They preserve every JSON/SGF byte, file mode and nanosecond
modification time. A content-addressed manifest binds the archive to the exact
training result and source snapshot. Each member and the ZIP have SHA-256 hashes.
The format uses standard ZIP with deflate level 1 and no pickle.

Packing publishes and verifies the archive before publishing
`artifacts/games.archive.json`. Eviction verifies the archive and every remaining
original before deleting any original. Restoration checks existing files and
atomically publishes missing files without overwriting different content. An
exclusive per-artifact lock serializes these operations. Interrupted eviction
and restoration can continue explicitly from their retained state.

The qualification (external or omitted experiment artifact)
includes an actual 110-file round trip and continued agreement with the saved
training/recovery comparison. Ten unit cases cover streamed reads, corruption, changed
originals, incomplete runs, unsafe paths, lock contention and partial operations.

The migration catalog (external or omitted experiment artifact)
records all eight controller-side archives for the seed-27 and seed-28 Gumbel CNN
attempts. It preserves 628,144 files representing 314,072 complete or capped
games: 650,997,542 logical input bytes, 385,672,826 ZIP bytes and 98,017,698 manifest
bytes. Filesystem block savings are larger than logical compression savings for
these small files; free-space deltas during concurrent work are not a controlled
storage benchmark. Completed and capped game outcomes remain distinct.

The subsequent seed-27 attention migration (external or omitted experiment artifact)
preserves another 391,180 files / 195,590 games. Its four archives contain
219,770,081 ZIP bytes and 60,981,008 manifest bytes for 347,614,628 logical input
bytes. The same verification and restoration contract applies.

Historical frozen analyzers that read `artifacts/games/*` require explicit
restoration. For example, from the workspace root:

```bash
archive_source=.gozero/snapshots/fc927ee97ad3ed8a5d9794f1c872ec3f68ea23f89eb7996097a03a83f776dd59
archive_artifacts=runs/pod-20260911T111515Z-fd49f349/rank-0/artifacts
.venv/bin/python -B "$archive_source/ops/archive_games.py" restore \
  --workspace-root /workspace/go \
  --artifacts "$archive_artifacts" \
  --output runs/maintenance/gumbel28-rank0-restore-001.json
```

Use a new receipt path for each operation. `pack`, `verify`, `evict` and `restore`
share this interface. The operator must run from a verified source snapshot;
artifact paths must remain inside this workspace's `runs/` directory.

This migration covers the controller filesystem. Remote originals were retained.
The local archives are not externally durable backups, and host-loss recovery
has not been qualified. Checkpoints, model exports, training results and frozen
source were not modified by archival.

When invoking `ops/pod_run.py` from a frozen snapshot, pass
`--workspace-root /workspace/go`. The controller's source root determines its
code and attestation; the workspace root determines local run records. The
controller rejects artifact roots inside either frozen source before creating
an attempt. This keeps operational logs outside content-addressed capsules.
# Streaming training data

`gozero.game_archives.open_records(artifacts, store)` verifies the complete archive and closed training result before exposing selected records. Its context holds the archive operator lock, checks every selected member again, and rejects archive/result changes before successful exit. Consumers publish derived datasets only after that context exits. This avoids restoring hundreds of thousands of small game files.

The seed28 attention migration preserved417,654 files /208,827 games; the completed wide-prefetch seed27 migration preserved202,852 files /101,426 games. Their read-only catalogs are `research/studies/runtime_qualification/attention28_game_archive_migration.json` and `prefetch27_game_archive_migration.json`. Remote original game records remain available; local archives do not establish external durability.
