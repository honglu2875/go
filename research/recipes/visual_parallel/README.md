# Local tensor-parallel visual inference

The TP1/DP4, TP2/DP2 and TP4/DP1 scorer owns its JAX mesh and real parameter/KV partitions. Attention/MLP expansion weights are column-sharded and outputs row-sharded. Packed KV projection weights remain replicated. All layouts use the same XLA attention and trained model bytes.

CPU and full multi-host TPU qualification passed. At 233M and global128, one-step medians were 7.641 / 8.084 / 8.552 ms; longer blocks also slowed. The 1.20x adoption gate failed. Read `research/studies/visual_causal/tensor_parallel_result.json`; no native serving adoption occurred.

Freeze each resolved configuration before execution. Existing snapshots and failed attempts remain immutable. Clone this complete recipe for a scientific intervention; shared environment and artifact helpers are imported from `gozero`.
