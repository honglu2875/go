# Varied-batch padding qualification

Outcome: **selected for research**.

Four varied real batches, same model/objective/optimizer. Exact shared-executable padding controls. Bounded original-compiler drift; no bitwise historical-compiler, long-run quality, strength or MFU claim.

| Runtime | Length | Median step | Live positions/s | Compiled peak |
|---|---:|---:|---:|---:|
| original_dense | 512 | 70.341 s | 732.0 | 31.32 GB |
| original_dense | 768 | 105.646 s | 663.0 | 32.16 GB |
| conditional_dense | 512 | 71.609 s | 719.0 | 26.42 GB |
| conditional_dense | 768 | 107.521 s | 651.4 | 27.21 GB |
| conditional_skip | 512 | 62.124 s | 828.8 | 26.42 GB |
| conditional_skip | 768 | 83.010 s | 843.7 | 27.21 GB |

Exact padding controls: **True**. Original-compiler group/metric gates: **True**.

The initial coordinate-wise screen remains rejected. Its diagnostic showed the discrepancy before any padding was skipped. This separate qualification preregistered fresh runtime batches and explicit group/metric bounds; see [the protocol](VARIED_PROTOCOL_001.md), [the failed screen](SCREEN_001.md) and [the diagnostic](DIAGNOSTIC_001.md).

Two samples per length are a short systems test. Compilation and host state-comparison costs are excluded from step latency. No new long learning run was scheduled, and no historical results or checkpoint arrays were altered.
