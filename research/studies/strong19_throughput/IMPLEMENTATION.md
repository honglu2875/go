# Padding-aware encoder execution

The candidate is an optional training runtime in
`../../recipes/strong19_padding_varied/`. The numerical model is unchanged:
same parameter tree, width, encoder passes, temporal transformer, spatial
readout, full/draft losses, value objective and AdamW settings. The historical
`strong19_long_ce` recipe and its snapshots are untouched.

The implementation flattens game/frame indices into the same chunks of eight
boards as before. It builds a live-frame mask from each game's count. A chunk
containing any live frame follows the original encoder function. A completely
padded chunk returns correctly shaped zero outputs through `jax.lax.cond`.
Partial chunks, frame indices, full histories, augmentation and example weights
remain unchanged. The outer `lax.map` stays sequential; vectorizing the
conditional over chunk predicates can turn it into a select that evaluates
both branches, defeating the optimization. See the
[JAX conditional contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html).

This is valid because the encoder operates independently on each board, without
batch normalization, and the omitted frames lie after each game's live causal
prefix. Temporal attention still processes the full padded shape. The loss
and readouts mask padded positions. A future noncausal operation, cross-frame
normalization or unmasked pooling would invalidate these assumptions and needs
new qualification.

`training.skip_padding` selects the optional path; its default remains `false`
for compatibility with existing configurations. `chunk_frames` remains eight.
The diagnostic-only `encoder_counts` batch argument permits forcing every
encoder chunk on while retaining the actual loss/history counts. It enables
the strongest control: one compiled program, identical model inputs, and only
the runtime choice to execute unused padding changed. Ordinary training uses
the actual counts automatically. The scalar crosscheck between ordinary and
diagnostic interfaces is in `diagnostic-interface-check-001.json`.

CPU qualification covers full outputs, gradients and optimizer state, FP32 and
BF16, partial/empty chunks, empty shards and four-device reductions. The TPU
screens compare full replicated parameters and both moments on every host,
hash initial state, bind data draws and source, and synchronize before recording
update latency. Complete state copies stay in host RAM; only small reports are
persisted. Compiler memory estimates are recorded separately from measured
update time and are not runtime HBM telemetry or achieved MFU.

The original coordinate-wise gate failed on the TPU. The follow-up isolated
that difference to conditional compiler execution even when every board is
computed. Skipping padding within the same executable is a separate exact
control. The [varied-batch protocol](VARIED_PROTOCOL_001.md) fixes its acceptance
criteria before running further batches; original failures remain visible.

Even after a systems qualification, a later long learning run must retain normal
validation and checkpoint audits. A new execution recipe is not an exact resume
of a historical source-bound checkpoint. Loading one requires an explicitly
audited migration or initialization step; do not bypass its identity checks.
