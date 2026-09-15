This clone establishes the sequential study's KataGo CNN baseline. The model
and optimizer use pure JAX; PyTorch is isolated to official-reference checks.

The architecture is `b40c768nbt-fson-mish-rvglr-bnh`, reduced to current-policy
channel 0 and its training-only batch-normalized helper. The 40 nested blocks,
768-channel trunk, 384-channel bottlenecks, Mish, global pooling, fixed
scaling, RepVGG initialization and center-gradient multiplier are preserved.
Main/helper losses have weights 0.2/0.8 and share the same MCTS policy target.
No behavior, value, ownership or score task is trained. The helper is excluded
from inference; its running averages are unnecessary because it is never
used for evaluation. Its training mean/variance use all live board points
across the global batch, with gradients through the statistics. Microbatch
boundaries only limit trunk activation storage.

There are 232,431,872 trainable parameters and 232,134,784 inference-active
parameters. Counts include the board stem, global projection and policy
heads. The later causal transformer must use width 768, batch-independent
normalization, and include its entire board encoder in both parameter and
cached-decoding FLOP budgets. See `research/studies/visual_katago/DESIGN.md`
for the sequential comparison and matching contract.

`baseline_1024.json` fixes 1,024 AdamW updates from scratch, 128 expert
sequences per global batch, D4 augmentation, an episode-count-weighted
history bucket distribution and the original whole-game splits. Evaluate
validation KL every 128 updates. The test split stays closed during recipe
selection. AdamW is a common controlled optimizer for this ablation, not
KataGo's complete production optimizer/schedule. Weight decay is selected
by parameter role, never by the rank of a stacked array. Group gradient,
parameter and update norms are recorded.

`build_features.py` links a small worker to pinned KataGo objects.
`prepare_features.py` reproduces V7 inputs from existing game moves and
checks every pre-action board and legal mask against the earlier Rust
cache. `export_reference.py` exports official CPU outputs and gradients;
`qualify_reference.py` compares them to JAX. `test_policy.py` checks the
global distributed objective, optimizer rejection and semantic transforms.
`train_policy.py` retains the earlier replicated checkpoint/recovery
protocol. All executions use frozen source/config snapshots.

The prior homemade ResNet/transformer comparison remains in
`research/recipes/visual_baseline` and its immutable study artifacts. Its
results are not measurements of this new baseline.
