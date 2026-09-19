# Longer common-objective CNN/transformer comparison

Cloned from the immutable transformer CE repair. Both bounded architectures
optimize signed-target cross-entropy against `[(1+y)/2,(1-y)/2,0]`. This preserves
the stored expectation but does not reconstruct the teacher's full WDL labels.
The CNN bounded loss and materialized reference now use CE. The transformer's
bounded numerical path, model parameters and deployment FLOPs are unchanged.
Scientific settings and evidence belong to `../../studies/strong19_long_pair/`.

This cloneable pure-JAX recipe trains a joint policy/value model from complete
fixed-corpus game histories. It owns its model, loss and optimizer code; common
dataset, snapshot, distributed execution and checkpoint transport live in the
importable `gozero` library.

`train.py` reads an explicit `fixed_joint_learning` configuration and establishes
the requested backend before importing JAX. `train_config.py` requires a positive
value-loss weight, memory path, optimizer and training purpose. The included CPU
configurations are qualification fixtures. They are not a proposed full-scale
training recipe or a reproduction of a published KataGo optimizer schedule.

Training uses the qualified joint objective and bounded encoder paths. Validation
uses `evaluation.py` to compute main policy and value outputs with bounded board
chunks; `host_metrics.py` normalizes aggregate metrics without scalar accelerator
calls. Complete game histories and masks retain their existing semantics.

Checkpoints include parameters, Adam moments and step, source/configuration
identity, all three sampling RNGs, exposure counters, validation and training
probe histories, and policy/value overfit diagnostics. Resume is source-strict.
The continuation tests and immutable snapshots are documented in
`../../studies/strong19_train/README.md`.

Execution-fixture manifests require `training.purpose = "qualification"`.
multi-host recovery stages the complete source-bound checkpoint closure before
distributed initialization, including RAM owner arrays and each rank's state.

Full-size TPU memory, timing and multi-process recovery still require execution
qualification. Register the data, optimizer, objective and comparison separately
before using this recipe for a scientific learning claim.
