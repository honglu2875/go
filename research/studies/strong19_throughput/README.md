# Fixed-model 19×19 training throughput

The user selected throughput first, preserving architecture and objective.
The parent is the completed 232,011,540-parameter transformer in
`../strong19_long_pair/`. Architecture, encoder sharing, both full/draft losses,
AdamW settings, complete causal histories and training-example weights are fixed.

The original run spends 70.47 / 105.83 seconds per update in the 512 / 768 frame
buckets. Only 77.38% / 71.50% of their frame slots are live. The bounded encoder
currently evaluates every padded frame. Host sampling is 132 seconds against
38,338 learning seconds, so loader changes are not the first intervention.
Original compiled temporary memory is 28.4 / 29.2 GB; a blind chunk-size increase
would risk capacity. See `baseline-analysis-001.json`.

## First experiment

The separate clone `../../recipes/strong19_throughput/` optionally skips encoder
chunks containing no live frame. Original chunk boundaries, partial chunks and
frame indices are retained. Padded frames follow each game's live prefix and
cannot influence a live causal prediction. The encoder has no batch-dependent
normalization. Loss, complete gradient and optimizer equivalence must still be
verified; mathematical masking alone is not a runtime qualification.

CPU gates cover FP32/BF16, partial chunks, empty games/shards, all-live/all-empty
populations, full parameter gradients and two AdamW updates on four devices.
The unmodified encoder path is retained as `compact_reference.py`.

The full-size experiment uses the exact first two registered training draws,
one from each bucket, and all original parameters/AdamW settings. Each runtime
and bucket starts from the same initialization, repeats that fixed real batch
for one warmup plus two measured updates, then compares complete parameter and
optimizer trees. Repeated-batch results are timing/numerical evidence only.
No checkpoint payload, giant HLO dump or profiler trace is persisted. Record
compiled memory, synchronized update latency, live positions/second and numerical
differences. Enforce the configured memory limit collectively on all hosts.

`benchmark-config-001.json` prospectively pins draws, runtime variants, numerical
tolerances and the memory gate. Freeze the source after CPU qualification.
A bounded pod attempt owns peer cancellation and timeout. Run one attempt at a
time and preserve the completed comparison's checkpoint replicas/datasets.
No long learning run is scheduled by this screen.

A speed improvement qualifies only if both buckets and all ranks pass numerical
checks. Compare against the measured in-attempt baseline; the old run provides a
separate sanity check. Two timed repeats are a screening measurement. Confirm a
winning implementation with a longer, varied-draw check before production use.
Compiler cost estimates through nested scans are not achieved MFU.

## Screen 001 outcome

Both TPU buckets failed the registered state tolerance despite faster timings.
The optimization remains disabled. See [SCREEN_001.md](SCREEN_001.md) for the
rejected result and a discovered overflow in the collective memory summary.
A separate conditional-control diagnostic is being qualified; it preserves the
failed screen and does not change its acceptance criteria.

## Diagnostic and final qualification

[The conditional diagnostic](DIAGNOSTIC_001.md) passed exact all-state checks
with padding included versus skipped in the same compiled program. Numerical
drift from the historical compiler occurs even when no boards are omitted.
The [separately registered varied-batch protocol](VARIED_PROTOCOL_001.md) now
tests four fresh runtime batches across both lengths, exact conditional controls
and explicit original-compiler group/metric bounds. Its bounded attempt is
`pod-20260921T002611Z-672dc811`; completion writes `VARIED_001.md`. No model or
objective changes are part of this pass.

## Completed result

The final varied-batch qualification passed and the optional runtime is selected
for future research. See [RESULTS.md](RESULTS.md) and `selection-001.json`.
`selected-training-config-001.json` changes only `training.skip_padding` to true;
`training-snapshot-001.json` records its frozen future-run source/configuration.
All TPU jobs are closed, and no long learning job was started.
