# Authorized research window — 2026-09-12

The user delegated eight hours at approximately 01:39:55 UTC, ending around
09:39:55 UTC. Continue the existing strong-engine objective, with the new causal
visual model and speculative rollout route as the primary direction. The user
explicitly permits changes to architecture, rollout and training, including
early-exit draft models, to improve useful TPU work and CPU utilization.

Current deliverable: a 233,137,152-parameter pure-JAX observation/action decoder,
causal visual soft tokens, separate expert/behavior heads, full-history and cached
block scoring, functional optimization and reproducible qualifications. No model
training, new native rollout integration or speed improvement has yet occurred.
Eight development tests passed before this registration; the frozen tests and
qualification entrypoint must be run separately.

Initial sequence:

1. Freeze and qualify CPU causality, GQA, patch ordering/padding, variable-board
   outputs, objective routing, optimizer updates, ragged cache equivalence and
   invalid cache requests. Qualify the bounded model harness on CPU.
2. Qualify TPU Splash forward/backward and cached XLA scoring at small width on
   all configured hosts. Then instantiate the full 233M configuration and check finite
   predictions/gradients, numerical tolerances, compiled memory and actual memory
   statistics. Maximum initial TPU attempts: four, each at most 900 seconds;
   failed attempts remain recorded. No large untrained checkpoint export.
3. Register supplied-path block versus sequential scoring at realistic histories,
   horizons 1/4/8/16 and explicit per-host/per-device batch counts, including the
   user's batch-128 hypothesis. Do not equate supplied-input timing with complete
   rollout latency or MFU. Retain prefix padding, KV traffic and rejected-work
   accounting. Record what profiling actually measures.
4. Develop an early-exit draft intervention with deep-target distillation and
   real board generation/verification. Separate policy sampling correctness from
   MCTS decisions and separate confirmed observation history from hypothetical
   branches. Compare an unchanged sequential control under matched work.
5. Use remaining time for a bounded learning experiment only when the preceding
   execution/data/learning prerequisites pass. Pin teacher/pretraining cost if
   reused; use independent real KataGo evaluation and preserve inconclusive or
   negative results. A long run needs storage/checkpoint headroom and recovery.

The expert/behavior log-probability ratio is not automatically a value advantage.
Possible research uses include stopped-target policy correction, weighted
behavioral residual prediction and a separately learned Q/value baseline. Any
claimed advantage relationship must state its policy/temperature assumptions.

The eight-hour reservation itself corresponds to 128 chip-hours for the observed
multi-chip pod, assuming continuous allocation. Attempt windows are a subset of
that cost. There are about 4.1 GB free on the controller at the start; no full-model
training may silently rely on storing multi-GB checkpoints there. Existing
immutable artifacts must not be deleted to make a trial fit.
