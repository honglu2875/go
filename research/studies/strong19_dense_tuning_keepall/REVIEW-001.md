# Completed dense LR and batch campaign

Completed 2026-10-05 at 19:53 UTC. All scheduled learning runs, full-rank state/replay audits, batch shape qualifications and selected-checkpoint disk publication passed. Post-completion review on 2026-10-06 re-verified both selected disk bundles. No research attempt remains open; the current inspection found no visible accelerator holders (foreign descriptor access was limited).

The LR screen retained 1e-3. All four points below used 128 updates and 7,001,181 position exposures, with the existing audited 1e-3 reference reused.

| Peak LR | Validation policy KL | Validation value MSE |
| ---: | ---: | ---: |
| 0.0006 | 0.875002 | 0.153700 |
| 0.001 | 0.842229 | 0.129620 |
| 0.00122 | 0.856248 | 0.142589 |
| 0.0015 | 0.841816 | 0.128236 |

The 1.5e-3 endpoint improves policy KL by only 0.05%, below the registered 0.5% selection threshold. Its last-three mean is modestly better, but this does not establish a different optimum. The finer 1.22e-3 point did not improve the selected incumbent, so the bounded refinement stopped. No new 256-update LR confirmation run was launched: that step was conditional on selecting a different rate. The retained control already has an audited 256-update reference. The generic scope sentence in comparison.json/RESULTS.md describes the planned later check, not an additional completed run.

At peak LR 1e-3, all batch arms used exactly the same 7,001,181 position exposures and canonical game/D4 stream. Schedule and evaluation progress were matched by game exposure.

| Global games per update | Optimizer updates | Policy KL | Value MSE | Learning hours |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 256 | 0.728001 | 0.121956 | 2.437 |
| 128 | 128 | 0.842229 | 0.129620 | 2.362 |
| 256 | 64 | 1.024584 | 0.169125 | 2.431 |

Batch 64 lowers endpoint policy KL by 13.56% and value MSE by 5.91%, with 3.17% more recorded learning time than batch 128. Its last-three policy KL mean improves by 14.42%. Batch 256 is worse in both losses and does not save learning time. Batch 64 is the selected quality candidate; batch 128 remains the throughput reference. Learning time excludes compilation, evaluation and loading, and the batch-128 timing is historical.

No sustained-overfit flag was observed. These are single-seed fixed-data learnability results, not Go playing-strength or MFU measurements. Batch 64 gets twice as many optimizer updates for the same data; moment and per-update decay dynamics therefore differ. Longer matched-exposure confirmation is the next useful research step before carrying this choice into a new dense–MoE comparison.

The selected batch-64 optimizer/RNG state is preserved in three RAM copies plus two verified fsynced disk bundles. Every prior checkpoint remains intact. No production setting was changed, and no additional run is queued. See selected-batch-disk-promotion.json, post-completion-review-001.json, the per-stage audits and curves.csv for evidence.
