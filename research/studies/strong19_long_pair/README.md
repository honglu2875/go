# Longer fixed-data 19×19 CNN/transformer pair

The user selected AdamW for both models to compare the architectures under a
common optimizer. Both use the repaired signed-target CE value objective. Each
model starts from its original seed and trains for 512 accepted updates, using
exactly 27,217,367 live position exposures from the existing frozen cohort.
This is 24.94 training-population passes, versus 5.30 in the 108-update pilot.
The enlarged background corpus does not enter this comparison.

| Setting | Both arms |
| --- | --- |
| Training population | 2,634 games / 1,091,214 positions |
| Full validation | 146 games / 60,284 positions |
| Fixed training probe | 128 games / 53,743 positions |
| Batch | 128 complete games globally, with identical bucket and D4 draws |
| Optimizer | AdamW, betas 0.9/0.95, epsilon 1e-8, decay 0.01, clip norm 1 |
| Schedule | 40-update warmup to 0.001, cosine decay to 0.0003 at update 512 |
| Value objective | CE against `[(1+y)/2,(1-y)/2,0]`, coefficient 0.7 |
| Validation | Before training and every 16 updates |
| Checkpoints | Complete parameters, optimizer, RNG and diagnostics at 256 and 512 |

Architectures, parameter counts and complete decoding FLOPs retain the pilot
settings: CNN 233,220,870 parameters; transformer 232,011,540. CNN helper weight
0.8 and transformer shared first-pass auxiliary weight 0.25 remain their
baseline-specific choices. Temporal history, contextual spatial readout,
four spatial attention blocks and encoder scale 0.01 remain intact. Bounded
training chunks remain 16 CNN frames and 8 transformer frames.

Only the horizon, evaluation/checkpoint cadence and CNN value objective change
from the preceding repaired comparison. This is a new schedule from scratch;
the old short-run checkpoints are preserved. Raw KL is compared within this
dataset. The teacher's signed expectation is available, not its full WDL labels.

## Execution and interpretation

The controller first qualifies the full-size CNN CE path for four updates,
then runs the CNN and transformer sequentially across the full pod. The
transformer bounded numerical implementation is the completed CE repair.
Six CPU loss/gradient checks passed, and both small complete models passed
exact uninterrupted-versus-2+2 restart, including optimizer arrays, sampler
states and every non-timing update/diagnostic. The first loss-test invocation
had a test-only parameter-prefix assertion error; its failed receipt remains.

The RAM-checkpoint validator previously permitted only a final checkpoint.
The cloned recipe now admits at most two equally spaced RAM checkpoints.
The existing transport and numerical update implementations are unchanged;
the new cadence validation has its own qualification and source binding.

`registration-002.json` pins the scientific settings, source snapshots,
independent 512-step sample replay, all operators and prerequisite evidence.
Every completed arm receives a complete all-rank checkpoint/state/sample audit
and a verified final peer copy before the next stage can start. The controller
stops on execution, audit, retention or monitoring errors.

Primary comparisons are final position-weighted and equal-opening-family
policy KL. Report both full curves, the last-three-observation means, value
MSE and measured learning time. The registered three-observation overfit rule
flags validation rising while the fixed training probe falls (0.5% margin).
It is diagnostic: the controller preserves the common fixed endpoint and
does not select an unretained best checkpoint or extend either arm. This
bounded longer study deliberately tests whether learning continues or overfits;
any subsequent horizon change requires a new paired registration.

This is one paired seed, with test targets closed. Supervised loss does not
establish strength, RL improvement or MFU. Real trained KataGo matches and
independent-seed confirmation remain later work.

Expected duration from observed pilot learning times is approximately 18–20
hours for both arms including ordinary overhead, plus the short qualification.
Live progress and overfit flags are recorded in `sequence-002/*-current.json`
and `sequence-002/events.jsonl`; final results are written automatically to
`sequence-002/result.json` and `sequence-002/RESULTS.md`. The first registration
stopped at resource preflight before any TPU attempt. A second old transport
cache was then verified against its two sealed replicas and released. The
retry preserves every scientific setting and retains the failed preflight.
An independent completion reviewer writes `RESULTS_001.json`, `RESULTS.md`,
the full curve CSV and PNG/PDF figures after the complete paired audit passes.
It reports endpoint and last-three means and retains all overfit observations;
generated figures still require human/agent visual review.

## Storage

The six completed larger9 checkpoints retain exact root-sealed arrays on two
peer hosts, with owner metadata and restore locators preserved. Releasing only
their owner caches recovered 16,675,351,740 bytes. Redundant CE resume transport
caches on three peers were released after verifying owner and distinct peer
copies. No datasets or current19 checkpoint owners were removed. Runtime
deduplication preserves every pathname and every unique payload.

Each long run reserves two owner checkpoint payloads and a final peer payload,
plus background-producer growth above the 64 GiB RAM-filesystem floor. The
96 GiB available-memory floor and the scientific pilot's 1 GiB disk floor
remain. Checkpoints are volatile RAM copies; durable promotion remains separate.
