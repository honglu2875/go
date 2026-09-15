This study extends the completed spatial-followup comparison. Its first stage
keeps the original fixed 9x9 weak native-MCTS corpus, the expert-only objective,
width 768, and the approximately 233M parameter / complete-decoding budget.

The ordered research questions are a rank-128 contextual readout, a nonlinear
contextual readout at the same internal width, and one spatial cross-attention
refinement before policy scoring. Each intervention is reviewed before choosing
the next parent. Combining first-pass and full-pass features is a later option,
not an automatically queued fourth intervention. The full model retains one
historical board token and the first-pass auxiliary policy loss.

Record minibatch training CE/KL and a deterministic training-evaluation subset
alongside regular complete validation. Compare the same checkpoints, legal-mask
semantics, and full-policy objective; auxiliary training loss is reported
separately. Flag sustained validation deterioration accompanied by improving
training fit. The fixed-data screens retain their prospectively fixed horizon;
the larger-data stage will have a separately registered horizon/early-stop rule.
Do not extend or shorten one arm after seeing its validation endpoint while
calling it a fixed-budget comparison.

The complete 9×9 `expert-v1` corpus is now stopped and frozen at Hugging Face
`quintic/go9x9`, tag `final-9x9-v1`; see `../go9x9_release/completion.json`.
After a few architecture screens, register its training view. This is a different teacher
and target distribution from the old weak native-MCTS corpus. Audit teacher
policy semantics, legal masks, histories, game-level splits, opponent strata,
and provenance before training CNN and the selected transformer from scratch on
the same release. Do not silently mix raw and searched policy targets. Maintain
separate immutable validation and closed test sets. 9×9 generation will not
resume. 19×19 varied-strength generation now runs separately under
`/dev/shm/go-corpus19`; the model comparison on that corpus remains the subsequent
study after 9×9 decisions are settled.

Storage and scheduling: assign separate CPU sets to data generation and
training controllers, respecting the deployment's resource allocations.
Use bounded `/dev/shm/gozero-staged-checkpoints` for temporary run checkpoints,
with verified copies on another host and persistent source/metadata/results.
RAM copies remain volatile; selected final checkpoints need persistent
promotion. Check filesystem and available-RAM floors before every launch.
Only manage files and process groups owned by the current experiment.

The rank-128 run completed and passed the provenance, draw, checkpoint and
validation audits. Endpoint KL was 0.40705645 versus the paired parent's
0.39701045; the last-three-checkpoint mean was also worse. The registered screen
did not qualify for another seed. See `rank128-decision-001.json` and the complete
curve in `rank128-first-seed-progress.png`.

The next intervention is the additive rank-64 readout. Its registration is
`additive-registration-001.json`. Full TPU qualification passed, including
cached/full logits, first-pass auxiliary semantics, complete decoding arithmetic,
all training buckets, and two verified RAM copies. The 1,024-update first seed
is `pod-20260915T020031Z-703c7e91` and completed with KL 0.41474593,
4.47% worse than its paired parent. Its early lead disappeared.
`additive-decision-001.json` records that it did not qualify for replication.

The final planned small-data intervention is a single rank-128 spatial attention
refinement on the unchanged rank-64 parent. See `refinement-registration-001.json`.
After its conditional replication decision, proceed to the larger-data comparison.

The refinement qualification passed all four ranks, full/draft cache equivalence,
CPU arithmetic and replicated checkpoint auditing. The zero-initialized new
projection preserves the parent initialization exactly. Its first 1,024-update
run `pod-20260915T030118Z-71aa88c0` passed the full audit but missed the improvement
threshold. See `RESULTS.md` for all three closed screens; rank64 remains selected.
