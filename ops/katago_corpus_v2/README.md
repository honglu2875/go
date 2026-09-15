The continuous 19×19 producer preserves the teacher, opponent pool, rules,
16-visit behavior and per-ply raw teacher targets from `../katago_corpus`.
It admits eight independent games per worker and refills a slot as soon as its
game finishes. It persists admission counters before submission, so a restart
cannot reuse an ID, and reports unfinished plies separately from published data.

The supplied example assigns eight CPU cores per worker. Ordinary strata use
seven Eigen inference threads for the teacher and one for the opponent; teacher
self-play uses four for each process. Size these example allocations to the
available CPUs and measure completed-game throughput before selecting a change.

`tests-001.json` records seven controlled queue/handover tests.
`qualification-result-001.json` records 24 isolated real-engine games, 96 plies,
both thread allocations, native replay/publication checks, and restart recovery.
These capped qualification games are outside both production contracts.
The pinned source environment is `deployment.json`; changing workspace source
does not change the deployed implementation.

`handover-001/` records all-host preflight and each production handover. The old
supervisor is paused by PID/start identity so its one-hour shutdown timeout
cannot interrupt the existing games. Its workers and engine process groups
continue until their batch completes. The new supervisor starts each replacement
only after the recorded old worker and both engines exit. An independent guard
also resumes the old supervisor after all its work finishes, protecting recovery
if the new supervisor fails. No old game is intentionally killed or discarded.
Never rerun `deploy.py` blindly: its no-overwrite checks deliberately reject an
existing handover. Inspect the saved config, process identities, guard and logs
before any recovery action.

Run `monitor.py` to record both generations, live workers, unfinished v2 plies,
storage headroom, and unexpected overlapping process allocations. A requested
stop drains all admitted v2 games; a supervisor failure has a 24-hour shutdown
bound. Admission enforces the configured free-memory floors and producer
storage cap. It never deletes data to make space. Both production cohorts
remain distinct and carry their exact contracts.

