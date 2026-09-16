This read-only diagnostic recipe is cloned from the completed joint19 pilot.
Its entry point accepts the frozen `joint_value_diagnostic` configuration in
../../studies/strong19_value_debug/diagnostic-config-001.json. It verifies the
retained checkpoint and dataset, broadcasts the exact parameter tensors, and
exports value logits plus normalized causal features for fixed complete game
histories. Both the ordinary inference and packed training paths are measured.

No optimizer updates occur and no test targets are read. The original model
arithmetic is preserved; the evaluator only exposes its existing latent vector.
The inherited training modules and CPU fixtures document source lineage, but
the diagnostic entry point does not accept learning configurations. The study
contains the all-rank analysis, paired CPU head-repair probe and separate
full-model objective repair. Use pinned snapshots to reproduce recorded runs.
