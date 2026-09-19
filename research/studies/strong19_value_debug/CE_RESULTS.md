# Completed 19×19 objective repair

The cross-entropy continuation completed all 108 updates on September 16,
21:40 UTC. Complete all-rank state/sampling audits and checkpoint replication
passed. The endpoint comparison was reviewed on September 19. No further
learning runs were launched between completion and that review.

Every arm consumed the same 5,787,025 live training-position exposures from
the frozen 1,091,214-position training population. Full validation contains
60,284 positions / 146 games; the fixed training probe contains 53,743 positions
/ 128 games. The repair shares the original transformer's initialization,
draws, D4 transformations, optimizer and learning-rate schedule. Only the
value objective and added diagnostics changed. All results use one seed.

| Arm | Validation policy KL | Equal-family KL | Validation value MSE | Equal-family value MSE | Learning minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN, source optimizer, MSE | 0.801642 | 0.823066 | 0.157816 | 0.168851 | 92.51 |
| CNN, AdamW, MSE | 0.989842 | 1.014182 | 0.159603 | 0.167390 | 92.24 |
| Transformer, AdamW, MSE | 0.965713 | 0.988983 | 0.542233 | 0.543564 | 136.26 |
| Transformer, AdamW, CE | 0.935420 | 0.958728 | 0.150406 | 0.159253 | 136.32 |

Lower is better for all losses. Learning time sums both CE segments and
excludes compilation, evaluation, checkpointing and controller overhead.
The CNN source optimizer is the qualified Muon/AuxAdam/Lookahead recipe;
the CNN architecture is KataGo-derived with the registered normalization
substitution. These controls do not reproduce the complete KataGo loss stack.

![Update and learning-time curves](ce-comparison-001.png)

## Interpretation

- The objective repair lowers transformer value MSE by 72.26% and policy KL
  by 3.14% against the identical original transformer. Mean signed prediction
  is now -0.000265 versus target mean +0.001179. Mean third-outcome probability
  is 0.00000669, with no validation positions above 0.99. This resolves the
  measured collapse and aggregate bias; it is not a full calibration assessment.
- The repaired transformer is 5.50% better in policy KL than the AdamW CNN,
  but 16.69% worse than the source-optimizer CNN. It takes 47.36% more learning
  time than the latter. Equal decoder FLOP budgets have not delivered equal
  training time. No MFU or trained playing-strength measurement is available.
- The CNN source optimizer reduced policy KL by 19.01% relative to its AdamW
  control. That is direct evidence that the recipe matters on this CNN, not
  proof that copying it will provide the same transformer benefit.
- The final transformer training-probe policy KL is 0.909444 versus 0.777411
  for the source CNN. The gap appears on both train and validation, and on all
  eight opponent strata. Current evidence favors investigating optimization
  and representation learnability before increasing regularization.
- No arm triggered the registered sustained-overfit rule. Repaired value MSE
  rose from 0.146893 to 0.150406 during the last interval; probe MSE also rose
  from 0.121345 to 0.124611. This last change is not the train-down/val-up
  divergence pattern. Policy continued improving on both splits. The endpoint
  is reported as registered, without selecting the lower update-99 value.
- Value error is lower than the CNNs in aggregate, but not in every opponent
  stratum. For example, the repaired transformer has value MSE 0.383643 versus
  source CNN 0.323635 in opponent stratum 5. Mixed-strength aggregate losses
  cannot establish strength against the teacher or a calibrated search value.

CNNs still used signed MSE while the repaired transformer used signed-target
CE. Thus the new cross-architecture differences mix architecture and objective.
The CE target is `[(1+y)/2, (1-y)/2, 0]`; it preserves the stored teacher
expectation but does not reconstruct unavailable three-outcome teacher labels.

## Proposed sequence

1. Qualify and run an objective-only CE control for the AdamW CNN, followed by
   the source-optimizer CNN, on this same frozen cohort and draw schedule.
   Keep value-loss weight, architecture and inference FLOPs fixed. This gives
   a common-objective comparison and shows whether CE helps the CNN policy too.
2. With the repaired transformer as the parent, screen the stronger optimizer
   family with explicit parameter routing/scales and checkpoint qualification.
   Keep architecture, draws and horizon fixed; do not combine an optimizer
   change with an encoder redesign. A smaller value-head initialization is a
   subsequent stability ablation only if the early oscillation remains costly.
3. Confirm a promising result with an independent paired seed, then register
   a longer comparison on a newly frozen subset of the expanded corpus. Match
   exposure budgets and report measured time as well. Monitor the unchanged
   validation/probe protocol, preserve family splits, and keep test targets closed.
4. Qualify full-size TPU inference and compare retained models through the
   real KataGo harness. Measure cached decoding latency and training memory/
   recomputation cost before selecting a throughput intervention. Production
   training and speculative-rollout claims remain later milestones.

Operational prerequisite: collection has reached 18,205 completed games /
7,545,677 positions at the September 19 16:38 UTC observation. Three hosts
are generating and one is paused by its storage guard. Archive selected
checkpoints and reclaim verified redundant caches before further allocation;
the fixed scientific cohort and closed test population remain unchanged.

## Evidence and reproduction

`compare_endpoint.py` creates `CE_RESULTS_001.json` by checking the closure,
plan, audit and replica hash chain, matching all-rank batches/D4 clocks, and
checking every validation/probe population. It sums learning time over the
prefix and continuation rather than treating the resumed segment as a full run.
`plot_endpoint.py` produces the PNG, PDF, CSV and hash receipt from that report.
Original immutable recipes, controls and observations remain unchanged.

Final CE checkpoint retention passed on the owner and one peer. Both copies
are in volatile RAM; persistent checkpoint promotion remains outstanding.
The existing public source backup contains the repair and prefix result but
does not yet include this final report or the model weights.
