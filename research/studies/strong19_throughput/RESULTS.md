# Fixed-model transformer throughput optimization

**Selected for subsequent research runs:** skip encoder chunks containing only padded frames. The 232,011,540-parameter architecture, full histories, losses, optimizer and sampled examples are unchanged.

| Sequence length | Original update | Optimized update | Less time | Compiled peak, original → optimized |
|---|---:|---:|---:|---:|
| 512 | 70.341 s | 62.124 s | 11.68% | 31.32 → 26.42 GB |
| 768 | 105.646 s | 83.010 s | 21.43% | 32.16 → 27.21 GB |

Timings are medians of two varied batches per length, taking the slowest synchronized host for each update. Compiled memory is a compiler estimate. Compilation and host audit work are outside timed updates.

## Numerical evidence

The initial strict coordinate-wise TPU screen **failed** and remains rejected. A separately registered diagnostic showed that its discrepancy appears when introducing conditional execution even with every board still computed. The same compiled program with padding enabled versus skipped matched exactly.

The final qualification carried state through four different real batches spanning both lengths. All parameters, both AdamW moments and scalar metrics matched the conditional reference exactly at every update on every rank. Independent comparisons with the original compiler passed the prospectively registered group/metric bounds. CPU checks also cover full gradients, FP32/BF16 and empty/partial shards.

Maximum relative L2 across semantic groups: parameters 7.09e-06, first moments 0.000635, second moments 0.000471. Minimum first-moment cosine: 0.9999997982. Maximum absolute parameter difference from the old compiler: 7.49e-05. This is not bitwise reproduction of the historical compiler.

The memory-summary overflow discovered in the first harness was corrected in subsequent drivers using ceiling-KiB int32 collectives. Historical failed receipts and original per-host measurements are retained.

## Reuse and limits

Use the cloned recipe with [selected-training-config-001.json](selected-training-config-001.json); its only training configuration change is `training.skip_padding: true`. The old configurations retain their default dense path. Freeze this configuration as a new run before training. Historical source-bound checkpoints require an audited migration or initialization step. No long training job was started by this optimization pass.

Normal validation and checkpoint audits are still required in the next long run. These short systems tests establish neither long-run learning quality nor playing strength or achieved MFU. See [implementation details](IMPLEMENTATION.md), [the rejected screen](SCREEN_001.md), [the diagnostic](DIAGNOSTIC_001.md), [the prospective protocol](VARIED_PROTOCOL_001.md) and [the final audit](VARIED_001.md).

## Separate architecture ideas

Next, test a learned attention pool producing one board token while retaining the spatial policy readout. Also diagnose the contribution of history, then consider a distinct head for actual opponent behavior. These are unscheduled model/objective changes, described with primary sources in [FUTURE_IDEAS.md](FUTURE_IDEAS.md).

The three bounded attempts used 15.97 reserved chip-hours; this is allocation accounting, not utilization.
