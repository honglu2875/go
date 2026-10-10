G8 at peak LR 0.0015 became the default for new fixed-data 19x19 research on
October 10, 2026, following the user's explicit selection. The original sweep
did not automatically change defaults. Its completion reports retain their
historical selection status.

The default is recorded in [strong19.json](../../defaults/strong19.json), with
the complete configuration in
[g8_default.json](../../recipes/strong19_fuzzy_topk/g8_default.json). This file
copies the audited G8/LR 0.0015 configuration exactly. The previous dense
defaults are archived, and registered experiment configurations remain unchanged.

The selected model has 298,257,732 parameters. The unchanged 768-wide spatial
encoder feeds one board token into 18 causal temporal layers. Each temporal FFN
scores 5,464 features partitioned into 683 groups of 8, selects each group's
largest positive feature, and combines the selected dictionary rows. AdamW uses
peak LR 0.0015, terminal LR 0.00045, and 64 complete games per update.

At the common 256-update / 7,001,181-position endpoint, G8 improves policy KL
from 0.728001 to 0.686237 relative to dense, with 5.01% more recorded learning
time and 1.11% higher value MSE. At the same LR 0.001, G8 also improves policy
KL and value MSE. See the [complete sweep results](RESULTS.md) and
[comparison with conventional MoE](MOE_COMPARISON.md).

The decision selects a working research baseline. The runs use one seed and a
partial learning-rate schedule. Playing strength, longer-horizon stability and
inference-speed improvements remain to be evaluated. Logical active FLOPs are
matched; the choicewise kernel still issues dense matrix work. Future runs must
freeze fresh source/configuration snapshots and register their overrides.
