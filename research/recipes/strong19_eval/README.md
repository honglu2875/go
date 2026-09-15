This cloneable recipe connects joint learner checkpoints to inference. Its
model and cached scorer files are exact copies of the qualified joint trainer;
`../../studies/strong19_eval/provenance-001.json` records their identities.

The importable `gozero.joint_artifacts` library exports only main FP32 policy
and value parameters, preserving their bytes and source/configuration lineage.
Optimizer state and CNN training-only helper weights remain in the complete
recovery checkpoint. Exported weights load independently of that checkpoint.

The current entry point qualifies export, full/cached policy and value outputs,
and actual Rust MCTS leaves using small trained 19×19 execution fixtures. This
is preparation for serving selected learned models, not a playing-strength
result or a full-size TPU inference qualification.
