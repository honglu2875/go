# Root-child inference prefetch

This full recipe clones the qualified Gumbel CNN trainer. Rust prepares legal,
history-aware root-child observations for the initial Gumbel sweep. Predictions
are cached inside the owning actor. The unchanged sequential search consumes a
prediction only when it requests that exact direct child; deeper or unpredicted
leaves use ordinary inference. Native terminal values are never predicted.

`root_prefetch.mode` selects `off`, `queued`, or `wide`. Queued execution retains
the original batch executable and actor-to-slot assignment, dispatching branch
batches before one host fetch. Wide execution compiles one larger batch and may
change floating-point outputs. `limit` bounds speculative children per actor.
The native API allows one prefetch round per move and rejects stale identities,
invalid dimensions and nonfinite replies before workers mutate search state.

Actual nonpadding neural outputs, padded slots, executable dispatches, host
fetches and cache hits are counted separately. Unused predictions are extra
work. Replay metadata continues to count neural evaluations actually consumed
by the sequential search. Cache state is discarded at real-move boundaries;
complete scientific checkpoints and restoration are unchanged, while request
rounds differ from the sequential control. Process-local prefetch counters are
carried into the trainer's saved aggregate counters on restoration.

CPU qualification uses `smoke.json`, `smoke_queued.json` and `smoke_wide.json`.
See the registered study in `research/studies/root_prefetch`. No TPU throughput,
learning-efficiency or playing-strength claim follows from CPU equivalence.
