Carrying the KV arrays through the transformer layer loop reduced warm complete neural decode latency by about 27%, with identical arithmetic and the same cached/full prediction error.

| Host | Original median | Carried-cache median | Speedup |
|---|---:|---:|---:|
| 0 | 82.058 ms | 59.639 ms | 1.376× |
| 1 | 82.002 ms | 59.472 ms | 1.379× |
| 2 | 81.992 ms | 59.524 ms | 1.377× |
| 3 | 81.989 ms | 59.486 ms | 1.378× |

The registered test used original/carried/carried/original order, ten warm repetitions per case, the same initial 234.4M-parameter model and real game prefixes. The complete graph processes the pending action, current visual tokens and policy readout. Batch size is 128, prior history 128 moves, allocated cache capacity 512 moves, and executed attention extent 4,901 tokens. Cache outputs are observed and donated. CPU feature generation, transfers and pointer reset are outside the timed interval.

The original implementation placed full layer caches in scan inputs and stacked updated caches in scan outputs. The variant carries the arrays through the loop, updates only new slots and reads the active attention extent. It preserves the external cache layout. Compiler temporary allocation fell from 11,628,711,424 to 11,260,237,824 bytes; substantial temporary memory remains. The observed speedup cannot be attributed solely to that small allocation difference without a hardware trace.

all configured hosts passed the arithmetic, HLO identity and numerical audit. Maximum policy total variation against full-sequence inference was 0.0012534, below the registered 0.005 tolerance. The runtime clone also passes independent CPU causality/cache tests, including rejected malformed pointers. This is a qualified implementation improvement, not an end-to-end rollout speedup or MFU measurement.

The learning comparison retains the original runtime so an encoder intervention does not also change its measurement implementation. The carried-cache implementation is available separately for later integration.

[Registered protocol](runtime_cache_registration.json) · All host measurements and audit (external or omitted experiment artifact) · [Implementation](../../recipes/visual_token_runtime/causal.py)
