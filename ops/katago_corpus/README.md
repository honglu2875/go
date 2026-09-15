# Mixed-strength 19×19 corpus

This isolated producer continues data collection while the architecture study
uses the frozen 9×9 release. It clones the exact Python producer used for 9×9,
reuses its hash-pinned Rust Go extension, and makes board size explicit in every
engine query, policy mask, board replay, and split calculation. It does not
modify or restart the original producer under `/dev/shm/gozero`.

The fixed teacher is `kata1-tf3-b11c768-s11500M-d6163M`, SHA256
`73f6454eba62d2f6d099af8ce66d8c3fde6225e223c55817da0627590e98b0ae`.
This is an available strong general-board checkpoint listed near the top of the
[official network table](https://katagotraining.org/networks/) on 2026-09-15.
The eight opponent strata range from early b6c96 checkpoints through this same
teacher (the final stratum therefore supplies self-play). This labels varying
quality behavior with one consistent raw policy/value target. Neither nominal
network rating nor 16-visit play is a measured strength rating for this corpus.

The contract uses 19×19, komi 7.5, positional superko, multi-stone suicide,
pass-alive area scoring, a 1,444-ply cap, and 16 visits for behavior. The first
16 plies sample a 75% search / 25% raw-policy mixture. Teacher color varies by
game ID. Teacher raw labels appear on every ply; searched labels are eligible
only on teacher turns. Both colors use full causal history. Schema v2 records
board size and komi per game. D4-canonical first-eight-action families determine
fixed train/validation/test assignments. Concurrent search is not bitwise
deterministic, so published game bytes and producer/config identities are kept.

`prepare.py` freezes sources, native hash, and model hashes. `qualify.py` checks
all eight distinct real networks on empty and nonempty 19×19 histories, native
publication/replay and D4 splits, then runs real teacher/opponent prefixes for
both teacher colors. Qualification prefixes are separate from production.
`deploy.py` requires a passed qualification, verifies the frozen artifacts on
each host, and refuses to replace or duplicate an existing run.

Data live only under `/dev/shm/go-corpus19/corpora/<contract-id>/host-N/`.
Runtime records are private to each configured deployment. Configure worker
concurrency, CPU affinity and inference thread counts for the target machine.
The producer reserves RAM, caps its owned files at 24 GiB per host, and pauses
admission below 64 GiB free shared memory or 96 GiB available RAM. It never
deletes old games automatically. These RAM files remain volatile until a later
explicitly managed archive. The completed 9×9 corpus remains in shared memory
and is backed up at [quintic/go9x9](https://huggingface.co/datasets/quintic/go9x9).

To stop generation, create `runs/expert19-v1/stop` on each host. The supervisor
allows up to one hour for in-flight 19×19 batches to drain before terminating
its owned worker and engine processes. Failed workers back off and quarantine
after five failures. Inspect `supervisor.json`, per-worker `status.json`, and
published file counts; session counters reset after a restart. A completed
terminal game is required before claiming end-to-end production success.
