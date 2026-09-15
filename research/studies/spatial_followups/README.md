This four-intervention study is complete. The strongest research model is
the transformer with context readout, four encoder attention blocks and
shared first-pass auxiliary supervision. Its final KL is 0.397010 / 0.401142
in two seeds, versus the CNN reference’s 0.420361 in its matching first seed.
All training, qualification, paired-draw and checkpoint audits passed.

Read the complete comparison and learning curves (external or omitted experiment artifact).
Completion record (external or omitted experiment artifact) pins the immutable evidence, current
checkpoint locations and resource ledger. This page remains a working index.

| Intervention | Status | Result so far |
|---|---|---|
| CNN main policy only | Complete, audited | KL 0.440776; the CNN with its original training helper achieved 0.420361 under the same optimizer and draws. |
| Context-conditioned spatial readout | Complete, replicated | KL 0.453701 and 0.445765; paired controls 0.513489 and 0.506703. Mean KL reduction 11.83%; both latency screens passed. |
| Four within-board encoder attention blocks | Complete; quality gain replicated under explicit adaptive follow-up | KL 0.403794 / 0.412231; paired gains 11.00% / 7.52%. Both original 15% latency screens narrowly failed. Selected as an auxiliary research parent with that cost retained. |
| Shared first-pass auxiliary supervision | Complete; full-policy gain replicated | KL 0.397010 / 0.401142; paired gains 1.68% / 2.69%, or 2.19% of the mean. Learning times 25.63 / 25.91 minutes. Both full-policy screens passed; both draft-utility screens failed on speed. |

Each learning run uses 1,024 AdamW updates, 128 whole games globally, width
768, and peak/end learning rates 0.001/0.0003. Within each seed, independent
audits reconstruct the actual game and D4 augmentation draws. Models stay
within 1% of the CNN's parameter and complete cached-decoding arithmetic
budgets, including the encoder. Full-shape TPU qualification covers all
three sequence buckets before learning. The test split remains closed.

The corpus contains 9,466 training games / 836,486 positions and 1,170
validation games / 102,339 positions. Labels come from the fixed weak native
MCTS teacher, not a KataGo expert checkpoint. These experiments measure
fixed-data learnability and neural execution; they do not establish Go
strength, RL sample efficiency, end-to-end speculative throughput or MFU.

The current context model also offers a useful learning-time trade-off:
seed 1 reaches KL 0.453701 after 20.28 learning minutes; the helper CNN is
at 0.459147 after 24.17 minutes and reaches 0.420361 after 31.98 minutes.
These are logged milestones, without interpolating crossings. Compilation,
evaluation, checkpoint writing and study engineering are excluded from the
learning clock. The LR schedule is in update units, so equal-time points
have different update counts and LR phases.

The context readout adds only 98,304 parameters. Its query comes from the
temporal model's representation and scores the current board's uncompressed
spatial features. The result supports this readout design; it does not
isolate the contribution of historical information from the current board's
global representation. The CNN control removes its helper objective and
the helper's batch-statistics path together, so it also does not separate
those effects or establish that its unchanged LR remains optimal.

The auxiliary experiment evaluates both the trained draft and the paired
parent's unsupervised first-pass draft on the complete validation population.
The draft sees full-pass history and a one-pass current board. Its cache
cannot replace a full-history cache without verification or recomputation.
Teacher KL, probability overlap, greedy agreement, full/draft latency and
extra training work are reported separately. The prospective contract has
distinct full-policy and draft-utility replication screens; a draft-only
gain does not promote a weaker full model as an architecture improvement.

Useful records:

- [Ordered registration](registration_001.json)
- Final completion record (external or omitted experiment artifact) and last running checkpoint (external or omitted experiment artifact)
- CNN control audit completion (external or omitted experiment artifact)
- Both context seeds (external or omitted experiment artifact)
- Attention parent selection (external or omitted experiment artifact) and [execution registration](attention_registration_001.json)
- Explicit adaptive attention replication (external or omitted experiment artifact) and [its pinned execution](attention_quality_replication_registration_001.json)
- Auxiliary loss, history and selection contract (external or omitted experiment artifact)
- [Exact transformer architecture and parameter shapes](ARCHITECTURE.md)
- Selected auxiliary parent (external or omitted experiment artifact) and complete arithmetic comparison (external or omitted experiment artifact)
- [CPU arithmetic comparison](prospective_training_arithmetic_001.json)
- Audited CNN/context learning-time milestones (external or omitted experiment artifact)
- Final auxiliary replication (external or omitted experiment artifact)
- Final resource ledger (external or omitted experiment artifact)
- Completed storage maintenance (external or omitted experiment artifact)

Storage maintenance reclaimed 6,317,989,888 allocated bytes across the pod
by sharing identical read-only evaluation fixtures and clearing disposable
peer uv caches. Installed runtime bytes were checked before and after.
All checkpoints and existing replica copies were retained. The initial
maintenance failure is preserved; the recovery leaves unqualified
historical peer source copies untouched. Subsequent single-owner checkpoint
relocations retain complete file hashes, ordinary-reader checks and explicit
persistent locations. Storage remains within the supplied pod.

The final draft KL is 0.411360 / 0.414780, with probability overlap
0.9393 / 0.9377 against each model’s full policy. Draft/full latency ratios
are 0.805 / 0.752, missing the registered maximum of 0.65. The draft uses
full-pass history; these are teacher-forced single-step measurements.

This round closed 11 TPU attempts, all passed, totaling 66.428983 attempt
chip-hours. The ledger includes engineering and idle allocation time
separately. No training attempt remains open and no further run is queued.
The final report records 14 related full checkpoints, including controls
and qualifications. The CNN main-only checkpoint is retained as exact
persistent byte parts on hosts 2 and 3 after a successful ordinary-reader
roundtrip; other relocated checkpoints have explicit complete-replica
locations. All storage remains within the supplied pod.
