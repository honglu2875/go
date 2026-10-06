# Autonomous dense LR and batch follow-ups

Wait for the registered 6e-4 / 1e-3 / 1.5e-3 screen to close and pass every
audit. Use no test labels. Each subsequent run gets an immutable plan and
source snapshot, with two disk copies of its registration before launch.
Stop on failed execution, checkpoint audit, source verification, backup or
resource guard. Do not automatically retry failed learners or start another
pod job while an attempt remains open. Maximum campaign duration is 30 hours
from the adaptive controller's start, including its wait for the initial grid.

At most two further 128-update LR probes are allowed. Use the initial grid's
selection gates: at least 0.5% improvement in both endpoint policy KL measures,
no regression in their last-three means, at most 5% worse tail value MSE and
no sustained-overfit flag. If the selected improving rate is at an edge,
extend by 1.6 upward or 0.625 downward. Otherwise test the geometric midpoint
toward the best stable alternative, using the immediately adjacent sampled
rate in that direction. Bound peaks to [2.5e-4, 4e-3], and do not insert a point
within 8% of an existing point. A second probe is allowed only if the first
became the selected incumbent. Record every input metric and decision.

The screened winner is provisional. If it differs from 1e-3, confirm it at
256 updates against the existing 256-update control, applying the same gates
at 224/240/256. Use a fresh deterministic run and require its entire first
128-update validation prefix to reproduce the screened curve within numerical
tolerance. This deliberately costs an extra 128 updates, avoiding changes to
the qualified fresh-run transport or source/topology resume contract during
a learning-rate study. Do not change its 512-update schedule. If it fails
confirmation, retain 1e-3. This settles a rate for this bounded study, not a
claim of a globally optimal LR or a production promotion.

Then compare physical global batches of 64, 128 and 256 complete games at
the settled rate. Reuse the audited 128-game reference. The other arms split
or merge the same canonical logical-rank game/D4 stream, with 7,001,181 total
position exposures: 256, 128 and 64 optimizer updates respectively. Merged
short/long draws are padded to the larger bucket. Report the extra padding
and measured learning time. Keep evaluation batch grouping at 32 games per
host, and evaluate at the same canonical game-exposure milestones. Scale
warmup, cosine horizon and evaluation intervals inversely with batch size.
Keep peak/end LR, AdamW betas, per-update decay, clipping, model, initialization,
data and objective fixed. Changing batch size changes update count and hence
optimizer moment/decay dynamics; this is a practical fixed-LR batch comparison.
It is not an independent optimizer retuning or an MFU measurement.

CPU checks must independently reconstruct all canonical draws and D4 choices,
verify unchanged model/update code, complete-game padding, schedule clocks and
saved replay cursors. Before batch learning, qualify both physical batch sizes
on both 512- and 768-position shapes using two repeated fresh-state updates
per case. Require finite accepted updates on every rank, identical replicated
metrics and compiled donated memory below 31 GiB per device. A failed
qualification stops the batch phase; never substitute gradient accumulation
and report it as a larger physical batch.

Report a batch quality candidate using the same validation gates, and a
throughput candidate only when learning time improves by at least 5%, both
policy endpoints/tails regress by at most 1%, tail value regresses by at most
5% and no sustained overfit is observed. Preserve all curves and report the
tradeoffs. Do not alter production defaults automatically. One seed and a
short training prefix do not establish convergence or playing strength.

Preserve every pre-existing checkpoint, including both initial LR endpoints.
No retirement, unlink, overwrite or deletion of checkpoint payloads is allowed.
New follow-up trials save guarded temporary RAM state using the established
checkpoint-stage transport and receive two additional verified read-only RAM
copies. This is explicitly volatile provisional state, not a durable backup.
All rank metadata, configurations, source, metrics and audits stay on disk peers.

Before the batch phase, copy a selected new LR endpoint to two checksum-verified
fsynced disk bundles containing every rank state. Likewise copy any selected
batch endpoint before closing the campaign. Publication only adds copies and
never removes the original RAM payload or any prior disk checkpoint. The
existing 1e-3/reference and initial-grid disk states remain untouched.

Admit the entire worst-case campaign before launching: five new RAM endpoints
plus a 64-GiB shared-memory reserve on the owner and replica peers; two selected
full-state disk bundles per promotion peer plus 2 GiB free on every disk. This
uses a smaller explicitly reserved peer-disk floor than the initial grid; the
initial grid must finish before any new model is allocated or promoted. Source
and result copies also retain that 2-GiB floor. Stop if resource guards fail.

This non-destructive design supersedes the unlaunched retirement-based
registration in ../strong19_dense_tuning. Automatic approval review rejected
that earlier launch specifically because it removed disk payloads. None of
those retirement actions were performed; this replacement has no retirement
operator or call path.
