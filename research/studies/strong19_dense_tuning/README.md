# Bounded dense LR and batch tuning

This campaign follows `../strong19_dense_lr/sequence-001`. It waits for that
fixed grid's completed audits, chooses at most two additional rates according
to [PROTOCOL.md](PROTOCOL.md), and checks any new LR winner at 256 updates.
It then qualifies and compares physical 64/128/256-game batches on an identical
game/D4 stream with matched position exposure and schedule progress.

`registration-001.json` pins the decision policy, code and CPU evidence.
`campaign-events.jsonl` records progress and choices; `lr-decision-*.json`
preserve the full evidence for each chosen point. `lr-selection.json` records
the settled rate for this study. Each new run has its own config, frozen
snapshot, plan, source backup, full-state audit and checked disk peer.

`batch-system-audit.json` must pass before either batch learner starts.
`comparison.json`, `curves.csv`, `RESULTS.md` and `result.json` describe a
completed campaign. Until those exist with passed audits, measurements are
provisional. No production defaults or test labels are changed.

Storage admission remains strict. Original research/reference checkpoints are
protected. `retention/` explicitly distinguishes two-peer volatile RAM copies
from durable disk states and records any later restoration of a selected batch
winner. Model/configuration/metrics evidence stays on disk peers throughout.

CPU evidence includes exact source lineage, all canonical logical-rank draws,
padding and schedule equivalence, saved cursor rejection, a full independent
audit of the existing dense checkpoint, bounded-decision fixtures and a tiny
two-peer RAM transport round trip. These are qualification fixtures, not
additional model-training results.
