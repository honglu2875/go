Seed 1 is now complete. `seed1-through-4096-002.json` extends the read-only
diagnostic to the registered endpoint. The independent paired run audit is
`../strong9_scaling/seed1-contrast-001.json`; the second seed remains open.
Validation gaps are +2.73% position KL and +4.00% equal-family KL, while the
matching training-probe gaps are +5.32% and +5.12%. Neither model shows the
registered sustained-overfitting pattern, and neither clipped gradients in
the final 512 updates. This supports an optimization probe without establishing
that higher LR will help.

At the endpoint, position KL by ply range is:

| Plies | CNN | Transformer |
| --- | ---: | ---: |
| 0–15 | 0.11666012 | 0.12680662 |
| 16–63 | 0.19194162 | 0.19351995 |
| 64–127 | 0.03101349 | 0.03399992 |
| 128+ | 0.00480843 | 0.00980663 |

The late-game relative gap is large against a low control loss and only 14,495
validation positions. Earlier and later cohort findings are descriptive and
share the same games; they are not independent significance tests.

The original diagnostic operator rejected the transformer's final auxiliary
`draft_heldout` row when requiring identical log schemas. It published no result
for that invocation. `diagnose_v2.py` explicitly validates and records this
training-only auxiliary observation while comparing only matching main/probe
populations. `diagnostic-amendment-002.json` records the change. The numerical
recipe, raw logs and original diagnostic operator remain unchanged.

The earlier descriptive check uses matched seed-1 logs through update 2,816;
`seed1-through-2816-001.json` pins the exact prefixes and configurations. It
checks equal local game/D4 draws, learning rates, global position counts and
evaluation populations. Its observations below are preserved with their
original checkpoint scope; the final seed-1 audit has since passed.

At that boundary, validation KL is 0.15258014 for CNN and 0.15609777 for the
transformer (+2.31%). Equal-family KL is 0.10569930 versus 0.10939431 (+3.50%).
The corresponding training-probe gaps are +3.69% and +3.57%. Both probes and
validation continue to improve; this is not the sustained-overfitting pattern.
These percentages describe this checkpoint, not a final architecture ranking.

The last 512 updates contain no clipped gradients for either model. The
transformer's temporal group has small raw gradient norm but a median update /
parameter norm of 0.00213, comparable to the CNN trunk's 0.00240. All recorded
groups continue to update. Group definitions differ across architectures, and
AdamW update norms include weight decay; this evidence does not prove equal
optimization quality or that every individual layer learns equally. It does
argue against a simply frozen temporal trunk or persistent late clipping.

At update 2,816, the first 16 plies contribute about 74% of the aggregate
validation-KL gap. Plies 16–63 are almost tied. This attribution varies over
checkpoints: at 2,304 the transformer briefly led position-weighted KL while
remaining behind on equal-family KL. Do not infer a permanent opening defect
or change the data weights from one checkpoint. Long-history plies 128+ have
only 14,495 validation positions and a small absolute KL gap, despite a large
relative percentage.

The two models have seen exactly 34,114,038 positions through this boundary.
The transformer's measured learning time is 0.797× CNN's on that same prefix;
this excludes startup, validation, sampling and checkpoint time. It is not an
MFU measurement or a playing-strength gain. The registered endpoint comparison
still uses matching updates/exposures, with timing reported separately.

If the final paired result stays behind without overfitting, an LR-only probe
remains a reasonable next test. Early clipping is common, especially during
transformer warmup, so a higher peak must be tested rather than presumed better.
No rate or encoder change has been selected or queued from this diagnostic.

`seed1-time-001/result.json` adds a separate descriptive view using the audited
cumulative learning clock. At the transformer's final 5,922.69 learning seconds,
CNN's adjacent validation observations are update 3,072 at 5,565.82 seconds
(position/family KL 0.14665043/0.10270500) and update 3,328 at 6,034.59 seconds
(0.14245403/0.09919333). Both are above the transformer's endpoint
0.13349390/0.09345388. CNN first records losses at or below both of these
endpoint-derived thresholds at update 3,840, after 6,976.10 learning seconds;
its preceding validation at 3,584 remains above both. These are post-hoc
observations, not a prospective target or a certified crossing-time interval.

The step/time curves are exported in `seed1-time-001/curves.csv` and
`seed1-time-001/curves-review-001.png`. There is no loss interpolation or extrapolation.
Learning clocks exclude setup, sampling, validation and checkpointing;
individual evaluation wall timestamps were not recorded. The transformer has
seen more examples at this common learning-time boundary and its cosine
schedule is further along. A fixed-wall-budget comparison with a prospectively
matched schedule would be a separate experiment. This result leaves the
registered equal-update endpoint failure intact and awaits seed-2 replication.
