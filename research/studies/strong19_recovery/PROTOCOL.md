# Replacement 19x19 collection and paired comparison

The September restart erased RAM-only 19x19 games and learned tensor payloads.
Preserved historical measurements remain separate; this is a fresh experiment
on a newly identified frozen population, not continuation of those weights.

Collect 100 terminal games per opponent lane on each of four hosts: 3,200
terminal games across eight equally represented opponent checkpoints. Preserve
the previous exact strong teacher, opponent hashes, 16-visit search, rules,
komi, full histories, raw labels and D4 opening-family splits. Archive all
completed games. The scientific view includes every terminal training or
validation game of at most 768 plies. This metadata-only limit was declared
before collection. Longer games remain archived and test labels stay closed.
Report eligibility exclusions and opponent/split counts with the final view.

All completed games and admission counters are fsynced to disk. A monitor
verifies immutable game receipts and mirrors new games to a different host's
disk. RAM holds disposable working data. On quota completion, the controller
replays native V7 features, checks exact boards/legal masks and unchanged
targets, freezes the cohort, and publishes two checksum-verified disk copies.

The matched comparison repeats the latest unresolved attention-pooling versus
flat-connector question, using the same 232M models, width, encoder, spatial
readout, causal backbone, AdamW, signed-target value objective, loss weights,
seed and qualified padding optimization as the original study. Full encoder
and cached-decoding FLOP accounting is carried forward only if the mathematical
source files are byte-identical to the qualified version. The sole model
difference is the registered board-pooling connector.

Both arms start from initialization and consume identical independently
replayed game draws and D4 augmentations on the replacement dataset. Batch size
is 128 complete games, length buckets 512 and 768, with bucket probabilities
determined by training-game counts. Evaluate all validation games and a fixed
128-game training probe every 16 updates. Compare a fixed 256-update stage of
the same 512-update cosine schedule, attention first then flat; retain all
overfit observations. No adaptive horizon or additional architecture sweep is
authorized by this controller. Endpoint KL is primary, with family KL,
last-three-validation means, value errors and elapsed training time reported.
No supervised comparison establishes Go playing strength.

Training requires successful CPU exact continuation for both architectures
and a tiny multi-host TPU peer-disk restore qualification. Full parameters,
AdamW moments, update counts, all rank RNGs and evaluation histories are saved
every 64 updates. Each checkpoint is fsynced on disk and the complete rank
bundle verified on a disk peer before further updates. Keep the latest two
local full states during an arm and its audited endpoint afterward; earlier
states remain on the disk peer. All byte/retirement receipts survive.
Host/JAX rank changes are explicitly rejected until an audited logical-rank
migration is performed. Stable-mapping recovery qualification is not a claim
of arbitrary topology migration or protection against simultaneous disk loss.

The collection follow-up and paired run operators are hash-pinned before they
start. Fail closed on failed qualifications, missing opponent strata, source
changes, insufficient storage, mismatched draws or invalid state. Preserve
failed attempts and their original receipts. Resource/stall monitoring and
the existing token-aware pod cancellation protocol remain enabled.
