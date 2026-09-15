This is preparation for the joint 19×19 comparison, not a learning experiment or
architecture selection. The larger-9×9 registration and its running jobs remain
unchanged.

Qualify the complete 768→256 spatial / pooled 768→256 / three-logit value branch
against the pinned official KataGo PyTorch implementation, including all active
parameter gradients and gradients into its trunk input. The temporal candidate
can read the normalized causal latent through a 768→256→3 projection without
another encoder pass. Its integration is still pending.

The corpus stores one signed raw teacher value in [-1,1], from the player-to-move
perspective. It does not store the full three-way value distribution or all
KataGo ownership, score, seki and score-belief targets. Qualifying the value
branch does not reproduce these other objectives. The eventual joint loss and
optimizer need an explicit registration; the MSE objective in the arithmetic
fixture is a test, not a training decision.

Value evaluation must retain full raw totals and report position-, family-,
opponent- and phase-weighted MSE. Include zero and constant-mean reference errors
so that lopsided games cannot make a trivial prediction appear useful. Full
graph parameter/FLOP accounting, unequal-rank normalization, joint gradients,
cached inference, TPU memory and real search remain required integration gates.
