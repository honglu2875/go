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

Every existing checkpoint is preserved. New provisional trial state is staged
in RAM with two verified RAM replicas. Selected new LR and batch endpoints
receive two complete fsynced disk copies; all original files remain in place.
The full worst-case RAM and disk budget is admitted before follow-up work.
See PROTOCOL.md for the explicit durability and storage limits.

This replaces an unlaunched design rejected by automatic approval review for
checkpoint deletion. The replacement has no checkpoint-retirement operator.
Numerical, replay, state-audit and transport qualifications remain prerequisites.
