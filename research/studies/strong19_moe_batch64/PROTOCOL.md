# Temporal MoE under the tuned dense defaults

The user selected peak AdamW LR 1e-3 and 64 complete games per update as
defaults for new research. Reuse the audited dense batch-64 endpoint and run
two fresh temporal-only MoE arms sequentially. Both have four top-two experts
per temporal FFN, expert hidden width 1024, 317,001,492 total parameters, and
the unchanged dense visual encoder. Active cached-decoding matrix FLOPs,
including the encoder and routers, differ from dense by about 0.000132%.
This analytical budget excludes dispatch, padding and activation costs;
actual learning time is reported separately.

1. `temporal`: retain the previous per-layer balance weight 0.004285714285714286
   and router z weight 0.0004285714285714286.
2. `balance_low`: divide only the balance weight by three. The hypothesis is
   that less uniformity pressure allows useful specialization. This is not
   assumed beneficial; expert utilization, load variance and loss are observed.

Both arms use seed 91312427, identical initial parameters to each other, the
same canonical game/D4 draws as dense, 256 optimizer updates and exactly
7,001,181 position exposures. Shared dense tensors retain their initialization.
The original exposure-based cosine clock is preserved: 1024-update schedule,
80 warmup updates, end LR 3e-4. AdamW betas 0.9/0.95, epsilon 1e-8, decay 0.01,
global gradient clip 1, signed-target value CE weight 0.7 and first-pass
auxiliary weight 0.25 stay fixed. Evaluate the full validation population and
fixed train probe at update 0 and every 32 updates. Test labels stay closed.

Before learning, qualify full-size dense at the 512-position bucket and both
MoE arms at 512 and 768: two repeated updates per case, accepted finite
metrics, no dropped tokens, agreement across ranks and compiled allocation
below 31 GiB/device. The dense first update must reproduce the audited
reference. Each fresh learner's first update must reproduce its qualified
512-position case. Unchanged numerical sources inherit their existing model,
gradient, causal-decode and checkpoint tests; new source/replay/config checks
pin this composition before registration.

Primary comparisons are endpoint position/family policy KL and the last-three
validation means. A provisional winner must improve both endpoint KL measures
by at least 0.5%, not worsen tail policy KL, keep endpoint and tail value MSE
within 5%, have no sustained-overfit flag, retain every routed token and take
no more than 1.2x the dense recorded learning time. Report all outcomes even
when no arm qualifies. One seed and a partial schedule cannot establish
convergence, MFU, playing strength or a production promotion. Historical dense
timing is identified. Weaker balance is selected over standard balance only
if the same quality criteria also hold against standard balance.

The controller has a 14-hour campaign deadline, 6000 seconds for systems
qualification and 14600 seconds per learner. It launches no overlapping jobs,
checks occupancy before each allocation, stops after failed audits, rejected
updates, repeated monitor failures, 30 minutes without an update, or exhausted
resource reserves. There are no adaptive additional training runs.

Preserve all existing files and checkpoints. Two new endpoint states use RAM
with two verified extra RAM copies. Reserve both endpoints before launching,
maintain a 64 GiB RAM-filesystem floor and 2 GiB disk floor, and reserve one
full selected MoE state plus metadata on each of two disk peers. Preserve the
standard arm unless the weaker-balance arm passes the quality selection gate;
copy that MoE endpoint to both disk peers even if dense remains the winner.
Sources, configurations, metrics and audits have disk backups. Unselected new
array payloads remain volatile until separately archived. No deletion or
checkpoint retirement is part of this campaign.
