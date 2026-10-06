# Matched CNN, dense-transformer and MoE learnability

See [PROTOCOL.md](PROTOCOL.md) for the fixed scientific scope.

Completed on 2026-10-05 at 04:48 UTC. Both fresh arms reached 128 updates;
all scientific/state/checkpoint audits passed. Temporal-only MoE nearly tied
its dense parent at the endpoint (policy KL 0.844658 vs 0.842229, value MSE
0.127651 vs 0.129620) at 1.0036× recorded learning time. Its top-one accuracy
was lower (46.44% vs 48.08%). The CNN reached KL 1.034096 and value MSE
0.225195. No sustained overfit flags were raised, and no further runs are
queued. See [results](sequence-001/RESULTS.md), [full comparison](sequence-001/comparison.json)
and [curves](sequence-001/curves.csv). This is one seed and a partial schedule.

The completed 421M all-expert MoE has a 64-update comparison with its dense
parent. The 317M temporal-only MoE previously had numerical and systems
qualification only. The historical 512-update CNN comparison used a different
corpus, so its losses are not a control for these MoE runs.

This study adds fresh temporal-only and CNN learners on the current immutable
19×19 corpus, sequentially for 128 updates each. The 232M flat dense and
attention-pooling dense runs already provide the exact matched prefixes. The
flat model is the primary control because it is the MoE parent; attention
pooling remains a secondary reference. All-expert MoE appears at update 64
only. All arms retain the 512-update AdamW schedule and the repaired value
objective, with full validation and a fixed training probe every 16 updates.

The encoder-inclusive active cached-decoding matrix budgets are within 1%;
total parameter count is allowed to increase for MoE. Training latency, router
overhead, padded work and optimizer cost are measured separately. Common
updates expose 3,596,975 positions at 64 and 7,001,181 at 128. The corpus has
2,906 training games and 1,233,366 training positions; validation has 152
games and 64,371 positions. Test labels stay closed, and collection stays
stopped.

The cloned recipe preserves learner mathematics. Its sole executable change
from the qualified temporal recipe is in the systems harness: it respects the
CNN's existing padding configuration. Six CPU checks cover the signed value
objective, CNN bounded/materialized gradients and distributed normalization.
The CNN additionally runs two fresh updates at each sequence length before
both learners are registered. `source-budget-lineage-001.json` binds parameter
schemas, source hashes and budget evidence.

`prepare_pair.py` freezes both arms before either learning curve is observed.
`execute_pair.py` monitors progress, resources and train/validation separation,
checks the first update against systems evidence, and runs the existing
all-rank sampler/state audit and independent disk-peer verification before
advancing. A failed stage stops the sequence. Checkpoints at 64 and 128 include
optimizer and RNG state. Earlier local payloads are retired only after the
audited endpoint and its peer proof; peer midpoint copies remain available.

The completed run history is in `sequence-001/events.jsonl`,
`sequence-001/temporal-current.json` and `sequence-001/cnn-current.json`. Final results are in
`sequence-001/comparison.json`, `curves.csv` and `RESULTS.md`. These include
policy KL, value MSE, top-one accuracy, family-weighted losses, last-three means,
train/validation gaps, router diagnostics and historical learning-time views.
One seed and a partial training schedule do not establish final playing
strength or MFU.

The two historical payloads retired for capacity have complete checksum-verified
archives on two disk peers; see `archive-20261004-result-001.json`. Current
strong-model endpoints remain available. Working datasets stay in RAM with
their existing disk backups. Operational receipts are private and must be
sanitized before public export.
