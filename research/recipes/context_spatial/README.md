This follow-up adds a history-conditioned correction to the successful one-token
spatial transformer. The frozen parent is
`fff3608e72827f0479a323cd22a0b672420cd2fcf890acdfd08bc933619d4bfe`.

For each point i, add dot(Wq h, Wk F_i)/sqrt(64) to the existing policy logit.
h is the normalized final temporal state; F_i is the final encoder feature.
Both projections are bias-free 768x64. Wq starts at zero and Wk uses an
independent seeded draw, preserving every common initial array and the
initial policy. Keep the existing local linear correction and pass logit.
The head executes in FP32, as does the existing policy readout. This adds
98,304 parameters and 8,071,296 dense FLOPs per 9x9 move, including both
projections and pointwise query/key products.

The encoder remains 24 shared width-768 convolutional blocks run twice;
the temporal transformer remains 18 layers at width 768. One observation
token and one action token represent each move. There is one policy target,
no behavior/value objective, and no batch statistics. All training settings,
fixed weak-teacher data, validation population, augmentation and optimizer
match the paired spatial control at LR 1e-3 and 1,024 updates.

`context_full_qualification.json` gates the full model/global batch on all
three TPU training buckets and complete trained-weight cache checks.
`context_1024.json` is seed 91312427; `context_seed2.json` is the registered
replication. Other inherited configs are reference material, not scheduled.
`qualify_shared.py` traces full encoder/connector/temporal/readout inference
and differentiated training including rematerialization. `test_shared.py`
checks causality, trained nonzero cached/full equivalence, guards, shared
pass gradients, rematerialization, zero-init equivalence, gradient activation,
spatial correspondence and unchanged pass behavior.

`compare_context.py` verifies identical actual data draws, common initial
arrays, all unaffected numerical source, dependency closure and initial
policy. A >=1% relative endpoint validation KL improvement with <=15%
initial-weight warm decode slowdown triggers the paired second seed against
the existing matching control. Both outcomes are retained. Only a replicated
gain can become the parent of the later attention/auxiliary experiments.
Results concern fixed-data learnability, not playing strength or MFU.
