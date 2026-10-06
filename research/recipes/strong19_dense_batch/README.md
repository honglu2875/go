# Dense matched-exposure batch screen

Clone of `strong19_dense_lr`, preserving every model, objective and optimizer
computation. `batch_replay.py` changes only complete-game grouping: split or
merge the canonical 128-game stream into physical batches of 64/128/256.
`train_joint.py` records the replay cursor and keeps evaluation grouping fixed.
The schedule and evaluation cadence follow the canonical game clock.

The separately registered `strong19_dense_tuning` study controls configurations,
CPU qualification, full-size TPU shape admission, comparisons and retention.
`qualify_batches.py` measures both nonreference batch sizes at both sequence
buckets before any scientific batch learning. It is a systems qualification,
not an extra training result. No expert, router or model architecture changes
are made by this recipe.
