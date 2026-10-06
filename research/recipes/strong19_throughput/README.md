# Execution-only transformer throughput recipe

Clone of `strong19_long_ce`. Parameters, inference, full/draft objective, AdamW
and registered learning schedule are unchanged. `training.skip_padding` selects
an optional encoder fast path that skips only entirely padded frame chunks.
Its default is false. The original recipe and immutable experiment are retained.

`qualify_padding.py` checks the new path on CPU; `benchmark.py` performs the
bounded full-size paired runtime comparison specified by its frozen config.
The benchmark entry point is `train.py`, using kind `joint19_throughput_benchmark`.
It writes small timing/provenance reports and no persistent model arrays.
The normal `fixed_joint_learning` entry remains available for later qualified
learning, but the benchmark does not launch or register a new learning study.

See `../../studies/strong19_throughput/README.md` for gates and interpretation.
