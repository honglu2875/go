This cloneable pure-JAX recipe repairs the value objective of the fixed-data
causal transformer. It is cloned from the completed joint19 pilot; model
parameters, initialization, inference and policy supervision are unchanged.

Set `training.value_objective` to `signed_target_cross_entropy`. The qualified
path is the bounded causal transformer. The value target is
`[(1+y)/2, (1-y)/2, 0]`, where y is the stored signed teacher expectation.
This is a surrogate, not a recovered teacher WDL distribution. Main and draft
value losses retain their existing weights. Validation still reports signed
value MSE and now also records all three predicted probabilities.

The original three-logit MSE objective can saturate the third outcome and
suppress negative predictions. `qualify_value_ce.py` checks restoring gradients,
NaN padding, distributed position normalization and full-backbone gradient flow.
The model, optimizer and exact-continuation harness are locally owned; shared
transport and dataset code remain in the importable gozero library.

The included transformer CPU configuration is an execution fixture. Scientific
settings and results belong to ../../studies/strong19_value_debug/. The registered
TPU screen stops after nine updates of the original 108-update schedule. Only
its checkpoint can continue that exact immutable run. A full training or
playing-strength advantage requires further evidence. This recipe has not
qualified a CNN cross-entropy implementation.
