# Current dense learning-rate study

The earlier 9×9 sweeps tuned other architectures and objectives. This study
starts a current-corpus 19×19 dense LR screen: 6e-4 and 1.5e-3 versus the
existing audited 1e-3 reference, with 128 matched updates per arm. See
[PROTOCOL.md](PROTOCOL.md) for the fixed comparison and selection gates.

The initial round runs sequentially. Each learner uses the full pod, an
unchanged 512-update cosine schedule, identical starting weights and exact
logical-rank data draws. Only the complete LR scale changes. Full optimizer
and RNG state is saved at the endpoint with a checksum-verified disk peer;
the test split stays closed. Configurations, operators and source snapshots
are pinned before launch and copied to two peers.

`sequence-001/comparison.json`, `curves.csv` and `RESULTS.md` are the completed
round's results. Live observations are provisional. `source-qualification-001.json`
and `reporting-qualification-001.json` record the source, optimizer and report
checks. `snapshot-dedup-result-001.json` records storage reclaimed by sharing
identical read-only snapshot files; all contents and paths are preserved.

The user has also authorized adaptive LR refinement and subsequent batch
screens. Those are separate prospective registrations after reviewing this
round; they must not change this fixed registration. Early LR gains need
later confirmation, since a previous 9×9 LR advantage reversed with longer
training. Batch comparisons must preserve total training exposure, adjust
the learning schedule and report throughput alongside validation quality.
