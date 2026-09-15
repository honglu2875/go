# Gumbel AlphaZero search control

This complete clone of `score_utility` keeps the CNN, ownership/score heads,
optimizer, replay and self-play trainer. It changes native search: sampled
Gumbel perturbations select a root subset without replacement; sequential
halving allocates the exact simulation budget; completed action values define
a policy-improvement target. Interior action selection tracks that improved
policy deterministically. The played action is the surviving root recommendation,
which differs from sampling the training target.

The implementation follows [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)
and the [pinned Mctx reference](https://github.com/google-deepmind/mctx/tree/88f92056a420c2673bed282f5a0c00211f126e78).
Unvisited Q values use the network/visited-policy mixed estimate. The default
candidate uses value scale 0.1, visit offset 50, completed-Q range rescaling,
at most 16 considered root actions and training Gumbel scale 1. Evaluation
uses zero perturbation. Legal actions and exact terminal values come from Rust;
this is model-free Gumbel AlphaZero, with no learned environment dynamics.

PUCT and Gumbel both count completed simulations excluding root inference.
PUCT returns its visit target; Gumbel returns the softmax of prior logits plus
transformed completed Q values. True outcome/ownership labels and game scoring
are unchanged. Unused PUCT, FPU, Dirichlet and temperature fields must be zero
or absent in Gumbel mode, preventing silent ineffective sweep settings.

`eval/gumbel_reference.py` executes the authors' pinned JAX routines on masked
finite alternating-player trees and compares them with a frozen Rust probe.
The first audit passed all 320 cases, with exact actions and root visit counts,
maximum policy error 2.39e-7 and value error 1.20e-7. It includes odd budgets,
legal-action counts, zero/stochastic root perturbations and both Q-rescaling
modes. Mctx and Chex are validation-only dependencies outside the training
environment. The Apache-derived native module retains its source attribution
and license in `crates/go-search/MCTX_LICENSE.md`.

Actor RNGs own the root draws and enter the ordinary complete checkpoints.
Native integration, fresh CPU training/recovery and GTP qualification passed.
CPU continuation matched 60 arrays, all actors and 28 subsequent games. The
registered 16.8M-move-per-arm pilot is in `research/studies/gumbel_search/` and
uses score utility factor zero. Both arms use exact, compressed checkpoints.
The registered seed27 pilot won62 of64 direct games against its fresh PUCT
control, with complete KataGo board/score adjudication, meeting the exploratory
promising criterion. Neither arm recorded an absolute KataGo win. The result
does not establish strong Go or general sample efficiency; an independently
initialized seed28 replication is registered. See the study's README and
immutable results for budgets, identities and limitations.
