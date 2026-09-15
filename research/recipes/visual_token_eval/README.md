This clone evaluates completed fixed-policy checkpoints game by game. Its entry point runs no optimizer and never opens the test split.

`build_inputs.py` pins completed learning audits, original model code, schema, saved checkpoint lineage and expected validation metrics. The evaluator invokes each exact trained implementation: `katago.py` for the CNN, the pinned `linear_reference.py` for the linear visual transformer and `causal.py` for the convolutional visual transformer. It reads authenticated parameter arrays from the owning host and broadcasts them, checking that their bytes are unchanged and agree across hosts.

Evaluation uses the original complete validation games, bucketing, batch size and legal masking. It writes per-game cross entropy, target entropy, top-move agreement and phase sums. `audit_games.py` verifies the complete episode population and reproduces the original logged endpoint before computing paired differences. Its bootstrap resamples complete games, preserving position weights in the aggregate statistic. The resulting interval describes uncertainty from validation-game sampling, not training seeds, architecture selection, teacher distribution shift or Go playing strength.

CPU checks compare the per-game reduction to NumPy and the registered aggregate calculation, check game ordering under four-device sharding for all three exact model adapters, authenticate real saved parameter arrays, reject modified result hashes and test paired-bootstrap invariants. The TPU evaluation is bounded and runs only after the active learning job closes.

No additional policy targets, self-play, augmentation or auxiliary tasks are introduced. Original source, checkpoints and learning reports remain unchanged.
