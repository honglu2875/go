# Attention under Gumbel search

This complete recipe combines the qualified `attention_control` model with the
full trainer and native search settings from `gumbel_search`. Its trainer is a
recipe-owned copy, byte-identical to the Gumbel trainer at creation. Neither
recipe imports another recipe during execution. Models and optimization use
pure JAX without Flax.

The proposed 9x9 comparison uses a four-block, width-64 CNN (307,461 parameters)
and a six-block, width-64 attention model (311,425 parameters). The attention
model has a convolutional stem, pre-normalized global attention, D4-tied
relative position bias, and a four-times-width MLP. Both use the same policy,
value, ownership and score heads and losses. Search uses the same Gumbel
allocation, completed-Q targets and perturbation settings. The optimizer is
momentum SGD in both arms.

`smoke.json` and `smoke_cnn.json` qualify the full CPU pipeline. `pod_smoke.json`
qualifies multi-host training and recovery. `pilot_9x9_control.json` and
`pilot_9x9_candidate.json` specify equal real-move and search budgets; they do
not by themselves authorize or register a study. Parameter matching does not
match FLOPs, wall time, suitable learning rates, or inductive bias. A comparison
must report those limits and use frozen protocols and real KataGo evaluation.

The architecture pair has already passed independent attention arithmetic,
gradient, symmetry and sharding checks in its parent recipe. The combined
Gumbel/attention pipeline requires its own full trainer and recovery checks.
No architecture strength or sample-efficiency result is established here yet.
