# Complete-learner gate after kernel selection

`qualify_execution.py` is a prepared fresh-state qualification entrypoint.
No candidate, configuration or TPU execution is registered yet. Select an
execution candidate only after the kernel comparison; freeze the candidate
and this plan before dispatch. Preserve the original learning run and all
earlier frozen kernel preparations.

Only a global grouped-matmul tile and the optional permutation VJP may change.
The 420,991,764-parameter model, initialization, data, seed, loss, router losses,
AdamW, chunk size eight and activation-rematerialization setting stay fixed.
Compare original/candidate at both 512- and 768-position buckets. Each case
starts from identical parameters and optimizer state and runs two updates on
its pinned real complete-game batch. Require identical initial tensor hashes.

Keep full reference states in host RAM, check at least 32 GiB host headroom,
and refuse compiled device allocations above 31 GiB. No learned checkpoint is
saved. Full parameter and first/second AdamW moments are compared after both
updates, globally and within each semantic model group. The predeclared
relative-L2 limits are 1e-4 for parameters, 1e-3 for first moments and 5e-3 for
second moments, with first-moment cosine at least 0.99999. Counters are exact.
Metrics use rtol 0.005 and atol 1e-4; the separate coordinate diagnostic is
retained. Do not relax a failed limit to adopt a candidate.

Report full-state drift, all-rank consistency, actual donated-buffer memory,
warmup/second-update latency and the qualified dense control separately. A
candidate can be slower or fail this gate despite winning a microbenchmark.
Passing two fresh updates establishes a bounded execution qualification, not
long-horizon numerical identity or a new scientific learning result.
