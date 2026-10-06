# Non-destructive dense tuning recipe

Clone of the established dense LR or matched-exposure batch recipe. Numerical
model, objective and optimizer code is unchanged. The configuration validator
admits explicitly reserved temporary RAM trial checkpoints. Selected endpoints
receive two complete fsynced disk bundles; every original checkpoint remains.

See research/studies/strong19_dense_tuning_keepall/PROTOCOL.md for the bounded
adaptive LR decisions, later-horizon confirmation, exact batch replay, system
qualification and full campaign storage admission. A temporary trial state is
explicitly volatile until selected and promoted; its metadata and curves stay
on disk peers. The library staging/reader and copy-only promotion are qualified
with tiny labelled fixtures before the campaign is registered.
