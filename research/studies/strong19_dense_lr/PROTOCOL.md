# Current-corpus dense learning-rate screen

Earlier dense-transformer LR experiments used 9×9 architectures and data.
The first four-point sweep favored 3e-4 on a 36-token CNN-encoder model. A
later one-board-token model benefited from 6e-4 versus 3e-4. The larger-data
9×9 follow-up rejected 1.5e-3 versus 1e-3 at its full 4,096-update horizon,
despite some early gains. Those experiments do not tune the current 19×19
architecture, corpus and repaired signed-value objective. The exact current
dense model has only the registered 1e-3 learning reference.

Run a one-dimensional grid of peak AdamW rates 6e-4, 1e-3 and 1.5e-3.
Reuse the audited 1e-3 dense-flat prefix. Run fresh 6e-4 first, then 1.5e-3,
sequentially on the full pod for exactly 128 updates each. These are the only
new learners authorized by this registration. Keep the architecture, initial
weights, seed 91312427, logical-rank game/D4 draws, global batch of 128 games,
betas 0.9/0.95, epsilon 1e-8, weight decay 0.01, clip norm 1, value coefficient
0.7 and auxiliary weight 0.25 unchanged. The 512-update cosine horizon and
40-update warmup remain; final scheduled LR is 0.3 times each peak. This
varies the scale of the complete LR schedule, including its decay updates.

Use the recovered immutable 19×19 dataset with manifest
8e1ab17423f083137367adec93d215d7dcea2440dfafa1c448f0cea834470855.
Every arm receives 7,001,181 position exposures. Evaluate the full 64,371-position
validation population and fixed 128-game training probe at 0 and every 16
updates. Keep test labels closed and generation stopped. Preserve all previous
MoE and CNN results; the newly tuned dense screen is reported separately.

Report both position-weighted and equal-opening-family policy KL, value MSE,
top-one agreement, every validation/probe curve, the registered last-three
means (96/112/128), learning time, gradient clipping and overfit observations.
A candidate passes this early screen only if both endpoint policy KL measures
improve by at least 0.5% over 1e-3, neither last-three policy mean regresses,
the last-three value MSE is at most 5% worse, and no sustained-overfit flag
occurs. If both pass, select the lower endpoint position KL, breaking ties
with family KL and then the lower peak rate. Otherwise retain 1e-3. Report
all metrics even when a candidate fails these gates.

This is an early-learning screen, not a convergence or playing-strength test.
The earlier 9×9 rate experiment is a specific warning that early gains can
reverse later. Do not promote a new production rate or claim a fair tuned
dense-versus-MoE advantage from this screen alone. Longer confirmation and an
equally budgeted MoE search remain separate decisions; no further arms are
selected automatically.

Clone qualified numerical code without modification. Bind its source lineage,
verify normalized LR schedules and a small analytic AdamW LR-scaling fixture,
and check each fresh learner's initial losses/gradients and scaled update
norms against the qualified dense first update. Independently audit all four
rank states, initial weights, RNG/data draws, complete parameters and moments,
fixed evaluation populations and the durable peer copy before advancing.

For this light screen, save a full durable optimizer/RNG checkpoint at the
128-update endpoint, with a verified disk peer. Earlier updates have logs and
validation records but no resumable midpoint; interruption may require
repeating at most this bounded arm. This changes storage cadence only, not
the science schedule. Reserve both final payloads plus metadata before the
sequence, preserve existing strong-model checkpoints, and keep the owner
2-GiB and peer 8-GiB disk floors. Stop on resource, execution, numerical,
audit or backup failure. Content deduplication may hard-link byte-identical
read-only snapshot files only after verifying their manifests; paths, bytes,
permissions and snapshot identities must remain unchanged.
