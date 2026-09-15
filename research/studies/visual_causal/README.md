# Causal visual Go: evidence and reproducible recipes

A 233,137,152-parameter pure-JAX observation/action transformer is implemented,
trained in bounded pilots, and integrated with Rust search and real KataGo
matches. Whole-history D4 augmentation helped the weak-data pilot. The first
online continuation did not improve the fresh KataGo panel. No strong-engine,
transformer-over-CNN, hardware MFU or faster-than-KataGo learning claim is established.

The authorized research window is 2026-09-12 01:39:55–09:39:55 UTC. Its complete
reservation cost is 128 chip-hours on the observed multi-chip pod, independently of
how much time is spent inside launched experiments. The allocation ledger also
covers work preceding this window. [Work contract](WORK_WINDOW.md),
[consolidated report](REPORT.md), artifact index (external or omitted experiment artifact),
[next research design](NEXT.md), [operational status](../../../ops/STATUS.md).

## Implementation

The causal decoder has 18 layers, width 1,024, 16 query heads, four KV heads and
an MLP width of 3,328. It uses RoPE, RMSNorm, SwiGLU, continuous 3x3 visual patches,
and interleaved observation/action inputs. Every earlier observation and action
is retained; future target actions cannot enter their own predictions. A 9x9
position costs 11 tokens; 19x19 costs 51. Parameters/residuals are FP32 and the
qualified matmuls/KV are BF16 with Highest precision. There is no Flax dependency.

Expert MCTS policy and observed opponent behavior have separate objectives and
heads. Value, score and ownership outputs also exist; score/ownership losses were
not active in these pilots. Layer-3/6 exits receive stopped deep-policy targets.
The behavior corpus is still limited; actual opponent-level calibration is not
established. The rule engine supports variable board sizes; this learning study
uses 9x9. A near-parameter-matched CNN screen completed 128 updates per arm; strong 19x19 training remains open. The CNN uses an explicit eight-observation window, so this screen does not isolate history from architecture.

Scientific model/loss/optimizer code is owned by cloneable recipes. Shared rules,
artifact handling, history conversion, data loading and checkpoint primitives
live in `packages/gozero`. Root `ops/` handles the production-facing pod executor,
source/native/data/candidate staging, supervision and cost accounting. Cargo and
uv locks, content-addressed snapshots and complete rank checkpoint groups pin
runs. Failed attempts remain recorded.

## Main measured results

| Comparison | Result | Interpretation |
| --- | --- | --- |
| Supplied H8 block versus sequential scoring, batch 128 | 2.19x | Inputs already supplied; excludes drafting, rules, acceptance and repair |
| Native suffix observation encoding | 1.6501x | Scorer gate passed; 48,323 leaves and 289,938 heads bitwise equal |
| Dynamic-block slot refill | 1.5995x generation segment | Exactness failed: 230 of 1,024 action tapes changed |
| Fixed one-position-block slot refill | 1.5118x generation segment | Gate passed: all 1,024 traces, seven target arrays and game records exact |
| Default versus Highest backward precision | 1.139x | Below 1.20x gate; retain Highest |
| Tensor partitioning, TP2/4 versus TP1 | Slower | Retain local replicated inference |
| Original early-exit speculative packets | 0.39–0.49x including CPU audit | Negative independent-root screen |
| D4-checkpoint speculative packets | 0.57–0.61x including CPU audit | Longer accepted traces; still below gate |
| Complete root-anchored decoding loop, H2/H4 | 0.726x / 0.724x | Negative after charging cache rollback and replacement inference |
| Complete deferred-repair decoding loop, H2/H4 | 0.596x / 0.539x | Negative with pending work and final deep cache drain timed |

The fixed-block pair completed 99,118 identical moves in each arm: critical
generation segment 716.919s blocked versus 474.205s refill. This is one paired
timing run, including native search, inference and checkpointing; it establishes
neither hardware MFU nor playing-strength improvement. The deferred-repair audit
checked 24,224 committed moves and 639 terminal paths; its timing gate failed.
Timing scopes differ and ratios must not be multiplied.

Kernel profiles (external or omitted experiment artifact),
suffix audit (external or omitted experiment artifact), dynamic refill audit (external or omitted experiment artifact),
fixed-block refill audit (external or omitted experiment artifact),
precision (external or omitted experiment artifact), tensor partitioning (external or omitted experiment artifact),
[original packets](speculation_result.json), [D4 packets](speculation_d4_result.json),
complete loop (external or omitted experiment artifact), deferred repair (external or omitted experiment artifact).

Raw XPlane traces are preserved. Compiler-leaf accounting estimated about 12.1%
of recorded peak for block scoring versus 5.12% for sequential scoring on the
profiled core, with masking and compiler-accounting limitations. This is not a
hardware MXU counter or end-to-end MFU measurement. Chrome instruction exports
were truncated and contained no counter samples.
[Accounting and limitations](xplane_kernel_accounting.json).

## Learning and external evaluation

The first two arms each made 1,024 updates from identical initial weights using
weak native-teacher and archived observed-move data. Early-exit auxiliary loss reduced
shallow/deep policy TV by 17.94%, but both models overfit. The separate D4 run used
the same complete episode draws and replayable augmentation randomness. Relative
to the unaugmented early-exit model, validation behavior CE, expert KL and value
MSE fell 39.33%, 31.31% and 25.05%. This is one seed, not an architecture comparison.
Learning audit (external or omitted experiment artifact), D4 audit (external or omitted experiment artifact).

The fresh D4 KataGo panel used 64 games per anchor and model, with paired colors
and opening positions. Candidate search used 16 simulations and KataGo one visit;
these are explicit budgets, not equal compute. Counts below are wins/losses/caps.
The `s` field in a KataGo network name counts training samples, not optimizer
updates; keep the full checkpoint IDs when interpreting the historical ladder.

| Historical/current KataGo anchor | Unaugmented reference | D4 model |
| --- | ---: | ---: |
| `kata1-b6c96-s938496-d1208807` | 56 / 7 / 1 | 64 / 0 / 0 |
| `kata1-b6c96-s24455424-d3879081` | 0 / 62 / 2 | 3 / 53 / 8 |
| `kata1-tf3-b11c768-s11500M-d6163M` | 0 / 28 / 36 | 0 / 62 / 2 |

All 48,002 boards and 337 completed scores were audited. Caps remain unresolved,
and no Elo or general strength interval is inferred. The earlier initial panel
had 259 completed losses and 125 caps over 384 scheduled games; it remains
retained. D4 panel (external or omitted experiment artifact), initial panel (external or omitted experiment artifact).

The D4 generator then produced 1,024 self-play games, with 1,015 terminal games,
nine caps and 99,250 moves. All boards, legal masks and outcome targets passed
native replay; an independent KataGo sample reproduced 5,978 boards and 32
completed scores. The merger retained 213 complete generated games as collection
holdouts, kept historical held-out/behavior data fixed, and recorded its sparse
bucket fallback. No KataGo weights, logits or games entered training targets.

Both continuations restored the same parameters, both Adam moments, absolute
step, host samplers and augmentation streams, then made 256 additional updates.
The control retained old expert data; the other arm used eligible fresh expert
self-play. Training-state and exposure audits passed. The fixed fresh panel had
64 games per anchor and arm:

| KataGo anchor | Old-data control | Fresh-self-play continuation |
| --- | ---: | ---: |
| `kata1-b6c96-s10014464-d2201128` | 49 / 15 / 0 | 45 / 18 / 1 |
| `kata1-b6c96-s24455424-d3879081` | 5 / 50 / 9 | 2 / 56 / 6 |

The continuation did not establish a strength improvement. Raw GTP, SGFs, boards
and completed scores passed independent audit. The collected match attempts
have failed scientific completion status because of caps; this is distinct from
an infrastructure crash. Training audit (external or omitted experiment artifact),
external panel (external or omitted experiment artifact). A separately registered post-hoc
collection diagnostic compares all three checkpoints on 213 games / 19,557
positions that were ineligible for training. Its outcome cannot change this panel.

The completed reuse audit reconstructed every episode draw. Expert exposures per
available position were 1.86 for old data and 18.66 for the fresh-self-play arm.
The former drew 7,730 distinct expert game IDs; the latter drew 842. Behavior
draws were identical. This is a substantial data-reuse difference and a motivation
for a controlled replay-ratio experiment, not proof of why strength declined.
Exact draw/reuse audit (external or omitted experiment artifact).

## Qualifications and remaining work

CPU and multi-host tests cover causal masking, gradients, role-weighted losses,
board conversion, captures/passes, cache branches and GTP integration. Controlled
checkpoint-boundary continuation reproduced all 114 small-fixture parameter/Adam
arrays and complete scientific state. Generation and full-Adam fork recovery also
passed on all configured hosts. These visual qualifications do not establish host-loss
or mid-update kill recovery. Fork recovery (external or omitted experiment artifact),
generation recovery (external or omitted experiment artifact),
suffix service (external or omitted experiment artifact).

The stable-block refill comparison and complete deferred-repair loop are closed,
with their CPU prerequisites, TPU attempts and independent audits retained.
[Refill registration](stable_refill_registration.json),
[deferred registration](deferred_registration.json). A second fixed-block timing
pair and the collection-holdout model diagnostic remain unrun. The dynamic-block
failure and both speculative-loop negative results stay immutable.

The new [CNN baseline recipe](../../recipes/visual_baseline/README.md) compares
232.39M convolutional and 233.14M transformer models with 128 matched updates.
CPU causality/gradient/search tests and a small multi-host TPU learning test passed.
Both full training runs and their checkpoint/draw audit passed. The separately
frozen KataGo panel and its independent raw-transcript audit also completed. The held-out test and measured costs are:

| 128-update endpoint | CNN | Transformer |
| --- | ---: | ---: |
| Expert KL | 1.20966 | 1.97943 |
| Behavior cross-entropy | 2.04707 | 2.92923 |
| Value MSE | 0.66170 | 0.97724 |
| Critical-rank learning seconds | 294.39 | 133.59 |
| Full attempt seconds | 580.90 | 544.29 |

The CNN fits this corpus better at the same update count, while taking 2.20x the
learning time. Full attempt cost includes compilation and checkpointing and
should not be used to infer steady-state compute efficiency. These are one-seed,
short-horizon observations with different history and inductive biases. All
1,024 rank-update draws and both complete Adam checkpoints were audited.
Learning comparison and costs (external or omitted experiment artifact).
The fresh paired KataGo panel strongly favors the CNN in this short screen.
Counts are wins/losses/unresolved caps, 64 scheduled games per checkpoint/arm:

| KataGo checkpoint | CNN | Transformer |
| --- | ---: | ---: |
| `kata1-b6c96-s938496-d1208807` | 64 / 0 / 0 | 25 / 17 / 22 |
| `kata1-b6c96-s10014464-d2201128` | 54 / 3 / 7 | 2 / 34 / 28 |

All 37,480 recorded boards, raw GTP transcripts, SGFs and 199 completed scores
passed independent audit. The 57 caps remain unresolved; no Elo is assigned.
The observed score bounds still favor the CNN even under the most favorable
assignment of capped games to the transformer. These are sample outcome bounds,
not a population confidence interval or equal-compute result.
Full panel (external or omitted experiment artifact), service qualification (external or omitted experiment artifact).

[Training registration](cnn_baseline_registration.json),
qualification (external or omitted experiment artifact), match design (external or omitted experiment artifact).

External checkpoint durability, a long-running generation/learner coordinator,
continuous production recovery, an equal-compute architecture study, replicated learning gains
and strong 19x19 play remain open. Read [NEXT.md](NEXT.md) before choosing the next
training campaign. More decode work or better device occupancy alone does not
justify scaling.
