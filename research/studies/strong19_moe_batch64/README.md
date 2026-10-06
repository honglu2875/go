# Tuned-default temporal MoE ablation

This study compares the audited dense batch-64 control with standard temporal
MoE and a three-times-weaker load-balancing penalty. See [PROTOCOL.md](PROTOCOL.md)
for fixed budgets and selection rules. The registration freezes source,
configuration, inherited tests and data/replay identities before execution.

Follow `events.jsonl`, `system-audit.json`, `temporal-current.json` and
`balance_low-current.json`. Completed evidence is written to `comparison.json`,
`curves.csv`, `RESULTS.md` and `result.json`. A live metric is provisional until
the full state, replay and checkpoint-copy audits pass.
