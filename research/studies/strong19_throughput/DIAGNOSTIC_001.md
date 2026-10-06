# Conditional padding diagnostic

Padding skipping is exactly invariant within the shared conditional executable on both updates. Compare original-vs-conditional groups to diagnose compiler lowering; screen 001 remains rejected.

All variants use the same fixed real 512-frame batch, model, optimizer and initial state. The two conditional variants reuse one compiled executable, with only the runtime encoder mask changed. Complete parameters, both AdamW moments and every metric are compared after each update on all ranks.

| Path | First update | Second update | Compiled peak |
|---|---:|---:|---:|
| original_dense | 70.361 s | 70.376 s | 31.32 GB |
| conditional_dense | 71.623 s | 72.128 s | 26.42 GB |
| conditional_skip | 61.814 s | 62.725 s | 26.42 GB |

This diagnoses numerical behavior; it does not qualify the optimization for production or change the failed first screen. The memory collective now uses int32 ceiling KiB to avoid int64 truncation.
