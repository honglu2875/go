# Padding screen 001: rejected

The CPU equivalence suite passed exactly, but both full-size buckets failed the prospective per-coordinate state tolerance. This implementation is not enabled. Original failed receipts and thresholds are unchanged.

| Bucket | Dense step | Skip-padding step | Less time | Compiled memory, dense → skip |
|---|---:|---:|---:|---:|
| 512 | 70.322 s | 61.815 s | 12.10% | 31.32 → 26.42 GB |
| 768 | 105.624 s | 83.012 s | 21.41% | 32.16 → 27.21 GB |

Two timed updates per bucket use the maximum host latency at each update. This is preliminary systems evidence on repeated batches, not a learning, strength or MFU result.

After three updates, 78 / 68 state coordinates exceeded the registered tolerance at lengths 512 / 768. Maximum errors were 6.45e-5 / 5.56e-5; aggregate relative L2 errors were 3.26e-6 / 2.71e-6. All scalar metric checks passed. Aggregate smallness does not override the failed gate.

Raw int64 byte counts were truncated by JAX x64-disabled process_allgather. All four per-host memory records are intact and individually below the gate, but the collective gate is invalid. The diagnostic will gather int32 ceiling KiB.

Follow-up: separately register a dynamic-mask diagnostic comparing original dense execution with conditional dense execution, then the exact same conditional executable with padding skipped. Hash initial states and compare complete parameters and AdamW moments after each of two updates, with per-leaf and semantic-group reports. This diagnoses the cause; it does not change screen 001 acceptance.
