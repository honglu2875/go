# Transformer research after the throughput pass

These are proposals, not changes to the current benchmark. The present user
instruction fixes architecture, objective, optimizer and data while optimizing
execution. The longer comparison supports a modest same-exposure loss advantage,
but the transformer costs 47.7% more learning time. Equal-time efficiency and
playing strength remain separate questions.

## Recommended order

1. **Diagnose what history contributes.** On a separately registered experiment,
   compare complete causal history against a short window and a board-only mask.
   First use an inference intervention as a sensitivity diagnostic; retrain a
   matched control before drawing a learning conclusion. The current target is
   a strong teacher's policy from detailed board features. Long history may
   provide less additional information for this target than for predicting a
   particular opponent. Preserve ko/rules and the provided historical feature
   planes; a history ablation must not silently change legal-state information.

2. **Improve the single board-token connector.** Replace the location-wise
   768→16 projection and flattening connector with a learned query attending to
   the 361 encoder features, followed by a small residual MLP. Keep exactly one
   width-768 temporal token and the successful spatial policy bypass. The
   hypothesis is that a content-dependent summary makes better use of that
   token; it is not a demonstrated Go improvement. Include spatial position
   information and preserve full/draft connector sharing. DeepMind's
   [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html) and
   [Perceiver IO](https://arxiv.org/abs/2107.14795) motivate learned latent queries.
   This proposes a connector change, not adoption of their complete architecture.

3. **Give history a behavior-prediction task.** In a later mixed-strength study,
   retain the teacher-policy head and add a distinct causal head supervised on
   the moves actually played. The actual-move target must not be relabeled as
   an optimal move, and the current/future action must be hidden from its own
   prediction. Evaluate behavior NLL/calibration and speculative branch
   acceptance separately from expert-policy KL. This returns to the original
   expert/behavior idea, where opponent style can make history useful. Use
   explicit mover/role masks from game provenance; no opponent identity leakage
   across held-out opening families. This is an objective change and therefore
   belongs outside the current execution-only pass.

## Other candidates and existing negative evidence

Parallel attention/MLP blocks and query/key normalization are plausible later
architecture ablations, motivated by Google's
[ViT-22B](https://research.google/blog/scaling-vision-transformers-to-22-billion-parameters/).
Their reported speed/stability gains do not establish a gain for this model.
Changing sequential residual dependencies changes the function and requires
retraining; it cannot be called an equivalent compiler optimization. QK
normalization should follow measured logit/gradient instability, not be added
solely because a larger model used it.

The prior small-data readout screens already tried rank 128, an additive
nonlinear readout and query refinement. All were worse than the bilinear rank-64
parent; see [their results](../readout_followups/RESULTS.md). Do not queue those
again without a new, explicit reason. The proposed learned input connector is
different from that already-tested output refinement.

KataGo's [auxiliary policy and value targets](https://arxiv.org/abs/1902.10565)
also motivate richer supervision. Our packed comparison data contains raw
teacher policies, signed values and played actions, not teacher ownership maps
or full WDL distributions. Ownership/score supervision needs a new audited data
release; it cannot be inferred from the present signed target.

Every architecture comparison should preserve width 768, approximately 232M
parameters, the user's encoder/temporal capacity balance, and explicitly counted
complete decoding FLOPs including the encoder. Match or report small residual
budget differences; do not hide extra encoder work behind parameter matching.
Review fixed held-out curves, late-checkpoint means and equal-time quality, then
confirm a winner with another seed before a longer learning commitment.
