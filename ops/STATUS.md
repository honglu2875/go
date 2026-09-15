The larger fixed-data 9×9 comparison is still in progress. Its first paired
seed favors CNN on same-update policy KL; the transformer uses less measured
learning time. The registered second seed must close before selecting the
next learning-rate-only intervention. See ../research/studies/strong9_scaling/
and ../research/studies/strong9_followups/.

Small complete joint policy/value models now pass CPU training, exact
fresh-process continuation, portable trained-parameter export, cached inference,
concurrent GTP/Rust search checks, and real KataGo interoperability. The separate
19×19 execution fixture and tiny trained weights do not establish learnability,
playing strength or full-size TPU performance. See
../research/studies/strong19_train/ and ../research/studies/strong19_eval/.

Full-size joint training and serving, the scientific 19×19 configuration,
selected-checkpoint strength evaluation and sustained production RL remain open.
Private live process, deployment and storage records are excluded from this
source publication. Configure a new deployment and freeze new source snapshots
before running these tools.
