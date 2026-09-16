# Value-head saturation diagnosis

The completed 19×19 transformer pilot learned policy but spent updates 9–54
predicting nearly zero value. Final validation value MSE was 0.54223, versus
0.15960 for the AdamW CNN; the training probe was similarly poor. All three
pilot arms used the same signed-expectation MSE objective. The CNN label denotes
its architecture and optimizer lineage, not the complete KataGo training loss.

## Confirmed failure mode

`diagnostic-analysis-001.json` inspects the retained final transformer on 16
complete games / 6,643 positions. Select the smallest game identity separately
within each of eight opponent strata in train and validation, without looking
at predictions. Test data stays closed. This is a small diagnostic sample,
not a replacement for the pilot's full validation population.

The head produces three probabilities and reports `v = p(win) - p(loss)`.
Its average probabilities are `[0.407487, 0.00003450, 0.592479]`. The losing
outcome is suppressed almost everywhere. For 3,064 positions with teacher
value below −0.5, mean predicted value is +0.04354 and mean third-outcome
probability is 0.95640; 84.40% have third-outcome probability above 0.99.
Median per-position MSE gradient norm with respect to logits is 0.00007042
on that losing subset, despite its value MSE of 1.08976.

The early training logs independently show value-head gradient norm falling
from 2.3565 on update 1 to 0.0006857 on update 4. No intermediate checkpoint
was retained, so the final logit measurement does not directly measure the
early third-outcome probabilities. It does establish the final saturation
mechanism. Training and inference reproduce the same failure: their mean
absolute logit difference is 0.00123 on this sample. This does not prove
bitwise parity; the largest logit difference is 0.01624.

For outcome scores `s = [1, -1, 0]`,
`dv/dz_i = p_i (s_i - v)`. If the third probability approaches one, both the
signed prediction and the gradients approach zero. The scalar MSE cannot
distinguish a balanced win/loss prediction from a saturated third outcome.
The latter obstructs representation learning because its Jacobian vanishes.

The proposed objective is cross-entropy against
`q = [(1+y)/2, (1-y)/2, 0]`, where `y` is the stored signed teacher value.
Its logit gradient is `p - q`, which remains corrective in this failure mode.
This preserves the target expectation and existing inference head. It is an
explicit signed-target surrogate: the dataset does not retain the teacher's
full three-outcome distribution. This is a conventional objective repair,
not a new RL algorithm or a claim to reproduce all KataGo losses.

## Frozen-feature intervention

`head-repair-001.json` compares the two losses from the same trained head,
with the backbone frozen and identical sampled positions and AdamW settings.
Eight complete games supply training features and eight supply validation.
Main/draft weights stay 0.75/0.25; each update samples 256 positions; LR is
0.0003. The prespecified observations are updates 0, 1, 4, 16, 64, 128, 256.

| Head updates | MSE loss: validation MSE | CE loss: validation MSE |
| --- | ---: | ---: |
| 0 | 0.614269 | 0.614269 |
| 4 | 0.577326 | 0.691720 |
| 16 | 0.580741 | 0.349321 |
| 64 | 0.350288 | 0.340045 |
| 128 | 0.337310 | 0.337758 |
| 256 | 0.349596 | 0.360557 |

CE escapes earlier. Both eventually reach similar error, and the validation
error rises at the end while training error falls. Do not use the final head
as a selected model or claim an asymptotic CE advantage from this tiny sample.
The result shows that the frozen representation can already support negative
values, making the objective/head optimization a concrete repair target.

Both probe arms use CPU arithmetic. An initial overly strict CPU/TPU equality
check rejected the probe before updates. Independent float64 arithmetic agreed
with the CPU head within 0.00000282 on the inspected rank; default TPU dot
arithmetic differed by up to 0.01562 in logits and 0.002662 in signed value.
The rerun records the complete-sample numerical difference and bounds it by
0.05 logits / 0.01 signed value. These numerical differences are much smaller
than the saturation effect; no exact CPU/TPU equivalence is claimed.

The earlier [scalar value-logit experiment](../value_logit_distillation/README.md)
was negative, including real KataGo games. Its scalar tanh head, small 9×9
model, terminal targets and teacher lineage differ. That negative result is
preserved; it neither establishes nor refutes this measured three-logit trap.

## Full-model repair screen

`ce-prefix-plan-002.json` registers nine updates from the original initialization,
using exactly the original 128-game draws, D4 transformations, dataset and
108-update LR schedule, including the 40-update warmup. Only the value objective
and added probability diagnostics change. Model parameters and deployment
encoder/decoder FLOPs are unchanged. The original control already completed
the same nine updates; it is not rerun or relabeled as a concurrent control.
The full validation and training-probe populations remain fixed.

Four CPU qualifications pass: saturated-gradient correction and expectation
consistency, NaN padding/empty masks, distributed normalization by live position
count, and full-backbone gradient flow with the new objective actually wired in.
The immutable recipe is `strong19_value_ce`; the original recipe is unchanged.

The controller owns execution, complete prefix audit, initialization/population
comparison and verified checkpoint replication. The 108-step configuration
does not authorize this controller to run past update 9. Full training,
independent seeds, a matching stable-objective CNN control and trained KataGo
matches remain separate research stages.

The prefix is now complete and audited. Both runs consumed exactly 496,462
live training positions, with identical initial parameters and sampling/LR
clocks. Full validation covers 60,284 positions; the fixed training probe
covers 53,743. See `CE_PREFIX_RESULTS_001.json`.

| Update 9 metric | Original MSE objective | CE objective |
| --- | ---: | ---: |
| Validation value MSE | 0.921595 | 0.624168 |
| Equal-family value MSE | 0.914327 | 0.634436 |
| Validation policy KL | 2.116735 | 2.081378 |
| Equal-family policy KL | 2.155808 | 2.120941 |
| Training-probe value MSE | 0.925374 | 0.614099 |

The observed reductions are 32.27% in position-weighted value MSE, 30.61%
in equal-family value MSE, and 1.67% in policy KL. CE's mean third-outcome
probability is 0.000005407, with zero validation positions above 0.99.
Mean predicted value is still +0.35444, so this early model is still biased.
The first few updates oscillated between win and loss; CE maintained restoring
gradients and then improved both the training probe and validation. This is
evidence for repairing the saturation mechanism, not final calibration or
long-horizon superiority. Learning time was 704.96 versus 704.81 seconds;
the objective repair does not add model parameters or inference FLOPs.

`ce-continuation-plan-001.json` separately registers continuation from the
audited update-9 checkpoint to the original update-108 endpoint. It preserves
the optimizer, sampler, full validation cadence and source snapshot. The
continuation auditor checks complete accumulated state and both metric segments;
it also passes the existing exact 2+2 real-19×19 CPU recovery experiment.
Full-horizon results are pending. This decision follows the observed prefix
results; it is not a retrospectively declared preregistered success criterion.

## Reproduction artifacts

- `clone-001.json`, `diagnostic-preparation-001.json`: immutable source/checkpoint
  lineage and the fixed diagnostic selection.
- `diagnostic-analysis-001.json`: all-rank input/output checks, probability and
  gradient statistics, and training/inference differences.
- `probe_repair.py`, `head-repair-001.json`: paired frozen-feature test.
- `ce-clone-001.json`, `ce-qualification-001.json`: repair source and CPU checks.
- `ce-prefix-plan-002.json`, `ce-prefix-001/`: bounded full-model run and audit.

All diagnostic checkpoint reads are read-only. Existing final pilot checkpoints
remain preserved. RAM and disk floors are checked before accelerator dispatch.
