A complete clone of `ownership_aux` for a low-budget search control. The model,
optimizer, trainer and all replay logic remain local to this recipe. Ownership
supervision stays at coefficient 1.5 in both arms; only `actors.fpu_reduction`
changes. `null` uses zero for unvisited actions. A number uses the current node's
network value minus that amount, clipped below at -1. The candidate is 0.2.

At a losing root, zero FPU is optimistic relative to evaluated actions. With
only 16 simulations it can spend much of the budget on fresh actions, with
little information to distinguish them in the visit-policy target. This is a
mechanistic hypothesis to test, not an established explanation of the ownership
pilot's poor play or its move-limit truncations.

The underlying `go-search` already implements both FPU modes; this recipe makes
the choice explicit through native actors, external-game search, checkpoints
and evaluation. It is a fixed reduction from the node's network value. It does
not claim to reproduce KataGo's complete FPU/utility/search implementation or to
implement Gumbel AlphaZero. All other inherited limitations still apply.

`smoke.json` is a CPU integration check. The 8,192-turn candidate has now run
on the multi-host TPU pod, following CPU qualification and a registered protocol.
It reuses the prior ownership-candidate run as its zero-FPU control. That reuse
was declared before launching the candidate and is an exploratory follow-up,
not a fresh confirmatory comparison. Both arms used 4.19M real moves.

FPU produced 17.97% more eligible terminal rows and 45.03% fewer truncated
self-play games, with 8,092 versus 8,097 learner updates. It concentrated the
final replay's visit targets onto 2.62 versus 6.19 root actions on average.
These are changes in each model's own data distribution, not proof of better
play. The real KataGo screen produced six losses and two unscored move-limit
games. No strength or sample-efficiency gain was established. The complete
registered protocol, immutable analysis and limitations are in
`research/studies/low_visit_puct/`.
