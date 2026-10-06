# Matched CNN, dense-transformer and MoE learnability

The user requested the same controlled learnability ablation used for the CNN
and best transformer. The old 512-update CNN comparison used a different
corpus, so its losses cannot be inserted into the current MoE table.

Run two fresh 128-update arms sequentially on the current immutable 19x19
corpus: temporal-only experts first, then the established BN-free nested CNN.
Both retain seed 91312427, 128 complete games per update, identical logical-rank
game/D4 draws, AdamW, the full 512-update LR schedule with 40-step warmup, and
the repaired signed-target value CE objective. Validate every 16 updates on
the full fixed validation population and fixed training probe. Keep test labels
closed. Preserve architecture-specific helper losses (CNN 0.8, transformer
first-pass 0.25) and all previously qualified numerical settings.

The existing flat dense transformer supplies the exact matched 128-update
reference; the attention-pooling transformer is a secondary reference, not a
post-hoc replacement for the sparse model's parent. At update 64, compare all
four primary architectures, including the completed all-expert MoE. At update
128, compare CNN, flat dense and temporal-only MoE. Do not extrapolate the
all-expert model to an unrun endpoint or retrain it merely to fill a table.

Reuse temporal-MoE numerical and systems evidence through exact source
lineage. The only executable change in this clone is that its systems-only
qualification respects the existing CNN/transformer padding configuration;
learner mathematics is unchanged. Run CNN loss/gradient/distributed checks and
two full-size updates at each sequence bucket before registering its learner.
Record shapes, parameter counts, encoder-inclusive active matrix FLOPs,
compiled memory and measured timing. Extra MoE parameters are permitted by
the user's matched-active-FLOP choice.

Register both fixed endpoints before either learning curve is observed. Report
policy KL, equal-family KL, top-one accuracy, value MSE, last-three validation
means, train/validation separation, positions and learning time. Preserve the
registered sustained-overfit observations; do not select the best intermediate
checkpoint. Compare historical timing explicitly as historical. One seed and
supervised losses do not establish playing strength or MFU.

Preserve all original frozen studies. Full optimizer/RNG checkpoints are
written at 64 and 128 with verified disk peers. Admission reserves both raw
payloads plus metadata and the existing owner/peer floors. Only retire local
historical payloads after two complete checksum-verified disk copies; retain
source metadata and restore locators. Keep datasets in RAM with their existing
disk backups, and keep collection stopped. The sequence stops on execution,
resource, audit or backup failure. No further arms are selected automatically.
