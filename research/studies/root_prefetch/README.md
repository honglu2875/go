# Batching Gumbel's initial root-child sweep

The cache implementation and cloneable trainer are in
[the root-prefetch recipe](../../recipes/root_prefetch/README.md). The native
comparison passed five simulation budgets (1, 7, 16, 31, 64), partial-cache
fallback, extreme logits, complete games and training targets, actor RNG state,
and continuation after restoration. Native stale and malformed responses were
rejected without accepting partial work. All 16 Python binding tests passed.

The [CPU qualification protocol](qualification_spec.json) is registered for
sequential, queued and wide execution, plus a queued continuation from turn 48.
The analysis (external or omitted experiment artifact) passed: the sequential control
reproduced all 60 legacy arrays and 55 games; queued execution reproduced all
60 arrays, complete scientific actors and all 55 games. Queued continuation
also matched all saved state and 28 subsequent games exactly.

Both exact arms evaluated 6,171 active neural positions. Queued execution used
3,792 cached predictions, all consumed, and reduced host fetches from 864 to
604. It increased padded neural slots from 6,912 to 16,352 because the 16-child
capacity exceeds this tiny test's eight-simulation budget and late-game legal
action counts. Its short CPU run was slower. These are qualification counters,
not a measured TPU throughput benefit.

Wide execution completed and its native state restored, but its floating-point
outputs changed training. It ended with 676 eligible rows versus 699 and
different final parameters. A changed rollout is expected to amplify small
numerical differences; this result neither qualifies exact execution nor
establishes worse learning. TPU numerical and learning validation remain open.

The [multi-host 9x9 qualification](tpu_qualification_spec.json) also completed.
Its analysis (external or omitted experiment artifact) verifies all 336 saved arrays,
complete scientific actor/trainer state and all 753 games between sequential
and queued execution. Continuation from turn 128 matched all 336 final arrays,
full actors and 362 subsequent games exactly. All four registered attempts
completed, using 0.5572 attempt chip-hours in total.

| Global work, 131,072 real moves | Sequential | Queued | Wide |
|---|---:|---:|---:|
| Actual active neural outputs | 2,216,799 | 2,216,799 | 2,214,961 |
| Host fetches | 17,408 | 10,973 | 11,014 |
| Neural slots including padding | 2,228,224 | 3,370,624 | 3,375,872 |
| Cached predictions consumed | 0 | 1,900,507 | 1,874,987 |
| Unused active cache predictions | 0 | 0 | 0 |
| Maximum training segment, seconds | 11.3034 | 13.4780 | 9.8012 |

Queued execution preserved actual work and learning, but its additional padded
executables outweighed fewer host synchronizations in this short run. Remaining
sequential requests also keep the batch active after many actors have finished.
Wide execution finished sooner but followed different self-play and learning;
its final parameter difference is not a fixed-input forward error. These
single, short repetitions do not establish a throughput improvement or worse
learning. Next work should measure fixed-input numerical differences, reduce
padding and remaining sequential work, then register repeated timings and a
learning comparison for approximate execution.
