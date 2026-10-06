# Recovery after the September restart

**Completed September 28:** both 256-update arms and all-rank endpoint audits
passed. See [RESULTS-003.md](RESULTS-003.md) and `sequence-003/result.json`.
The launch narrative below records the earlier restart and is historical.

The user authorized restarting on September 27 after checking that all TPU
hosts were free. The attention arm is running as
`pod-20260927T215053Z-92eb8d47`; the flat control is queued after its audit.
Follow `sequence-003/` and [RESTART_20260927.md](RESTART_20260927.md). The same
frozen scientific settings, data and draws are retained. The lost historical
corpus and weights remain separate from this new frozen population.

`startup-qualification-003.json` verifies the first accepted update on every
rank, exact registered draws, the expected TPU holders and live controller,
and initial validation/probe and update metrics matching the interrupted
initialization run. Both old failures remain separate from this running pair.

The collectors use the original strong teacher, all eight original opponent
checkpoints, original Eigen KataGo executable, and a newly qualified surviving
Rust native library. Sixteen independently replayed 19x19 histories (2,496
positions) matched KataGo board states and legal moves exactly. All eight
networks passed real engine queries. Fresh-process collector recovery passed
after the RAM working directory was removed, preserving completed games and
monotonic admission IDs.

The 32 collection workers completed the fixed quota of 3,200 terminal games,
balanced across the eight opponents, and exited. There are 1,356,897 positions.
Completed games, checksums and admission counters are written and fsynced to
disk, and the monitor finished verified disk peer copies before exiting.
The corpus is capped at 3 GiB per host, with an 8 GiB disk free-space floor and
an additional admission reserve. Logs and observer output are small; no old
research runs were deleted by this recovery pass.

The follow-up controller froze the predeclared eligible complete-game view,
checked V7 feature replay and exact labels, published two disk copies of the
packed corpus, and registered a fresh pair on exactly the same replacement
dataset and draws. Each arm has a fixed
256-update stage of the existing 512-update AdamW schedule, validation every
16 updates, and full-state disk checkpoints every 64. Attention runs first,
then flat, with authoritative audits between them. The September 26 attempt
completed 55 accepted updates before immediate cancellation; its first
checkpoint at 64 was not reached. A September 27 startup attempt failed on
idle lockfile permissions before any updates. After guarded lock repair,
sequence 003 restarts both arms from initialization in fresh directories.
Both earlier attempt records and the frozen scientific source remain unchanged.

## What was verified

- Both small model variants passed CPU uninterrupted-versus-2+2 training after
  removal of working inputs and the primary checkpoint. Complete parameters,
  AdamW moments, rank RNGs, draws, metrics and diagnostic histories matched.
- A real multi-host, 16-device TPU test passed the same comparison after all
  primary prefix checkpoint copies and all RAM inputs were erased. It restored
  from a different host's disk; all rank states, array hashes and subsequent
  update metrics matched exactly, except elapsed timings. This was a 14,896
  parameter execution fixture, not a new full-size model-quality experiment.
- The endpoint auditor and independent peer verifier passed on these actual
  TPU artifacts. Launcher disk/RAM admission checks passed on all configured hosts.
- Retention tests preserve an unmirrored checkpoint and only remove old local
  array payloads after a peer-copy receipt exists. Atomic-write tests passed.

The first TPU restoration script expected a coordinator-only launch record on
every host. Its failed receipt is preserved as `pod-recovery-qualification-001`.
Version 002 checks the correct per-host execution receipt, reuses the two
successfully completed training attempts, and records the successful restore.
No failed test was relabeled as passed.

Host/JAX rank mapping changed at reboot. Recovery under the current mapping is
qualified; a future different mapping is deliberately rejected by the trainer
until a separately audited logical-rank migration is performed. Disk copies
protect against RAM loss, not simultaneous loss of every host's disk.

## Operational entry points

The private host configuration remains in `ops/hosts.json`. Commands run from
the workspace root. These operators use pinned local runtimes and credentials
already configured for the authorized hosts; no secret is embedded in sources.

```sh
.venv/bin/python -B ops/katago_corpus_durable/resume.py
.venv/bin/python -B ops/katago_corpus_durable/monitor.py
```

`resume.py` is a no-op for healthy or completed collectors and rejects manually
stopped/quarantined workers. The monitor has an exclusive lock and exits after
the quota and peer backups complete. Do not launch duplicate follow-up
controllers; consult the PID, boot ID and start-time receipts first. Background
processes themselves do not survive reboot, but their durable inputs and
completed records do. A future restart requires explicit process resumption.

Current state and evidence:

- `ops/recovery_20260925/collection-status.json`: live collection and mirrors.
- `RESTART_20260927.md`: current sequence and the storage/lock recovery record.
- `registration-003.json`, `pair-process-003.json`: current registration/controller.
- `sequence-003/`: current observations, closure and paired comparison.
- `startup-qualification-003.json`: all-rank first-update and process checks.
- `HANDOFF_20260926.md`: previous pause and restart requirements.
- `handoff-20260926-verification.json`: all-host process/device release evidence.
- `cohort-result-001.json`, `cohort-disk-backups-001.json`: completed data and copies.
- `collection-followup-status.json`: last controller phase; consult its error
  receipt and the handoff record for the intentional interruption.
- `collection-followup-001.json`: immutable bounded follow-up and prerequisites.
- `PROTOCOL.md`: prospective scientific and storage contract.
- `harness-qualification-001.json`: exact CPU recovery for both variants.
- `pod-recovery-qualification-002.json`: successful real TPU disk restoration.
- `pod-stage-audit-001.json`, `pod-peer-verification-001.json`: independent audits.
- `runner-inspection-001.json`: non-launching resource/admission qualification.

The new scientific dataset manifest and paired registration were published
after collection and backup checks finished. They must not be confused with
the historical dataset or the earlier completed attention arm.
