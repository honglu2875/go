# Temporal fuzzy TopK sweep

The [protocol](PROTOCOL.md) records the prospective grouping and LR decisions.
All seven conditional trials completed. The [results](RESULTS.md),
[learning curves](curves.csv), and [numerical summary](scientific-summary.json)
contain the fixed-data measurements, selection gates, active/issued FLOP
budgets and system checks. The [MoE comparison](MOE_COMPARISON.md) uses the
same corpus, validation population and game/D4 exposure stream.

G8/LR 0.0015 was subsequently [selected as the research default](PROMOTION.md).
The [cloneable recipe](../../recipes/strong19_fuzzy_topk/README.md) includes
the complete trainer, configurations and six CPU qualification tests. It uses
the importable `gozero.fuzzy_topk` implementation. `budget.py` computes model
parameters and encoder-inclusive active/issued matrix FLOPs.

Private registration, launch, host and checkpoint receipts are excluded.
The original campaign validated every 32 updates and audited each endpoint
before selecting the next trial. To repeat the procedure, obtain the fixed
dataset, configure your environment, clone the recipe, and freeze fresh
source/configuration snapshots with the repository tools. Public source hashes
are recorded in the repository's `PUBLIC_SOURCE_MANIFEST.json`.

The 256-update runs use one seed and a prefix of the 1,024-update schedule.
Longer paired-seed confirmation and actual Go evaluation remain outstanding.
