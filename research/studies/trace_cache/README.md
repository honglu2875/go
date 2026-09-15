# Dense retained KV: correct, slower on TPU

The complete `trace_cache` recipe passed CPU and multi-host TPU event/state qualification, including real captures, legality corrections, terminal/cap resets and inactive games. Its registered repeated TPU speed criterion failed. Dense retained KV was slower than the corresponding full-prefix mode in every repetition, despite clear gains in the small CPU check. This result remains an implementation/control for further research; it is not promoted.

| Cached mode versus the same full-prefix mode | Four TPU speed ratios | Median |
| --- | --- | ---: |
| Sequential | 0.8292, 0.8384, 0.8608, 0.8413 | 0.8399× |
| Learned behavior k=4 | 0.8492, 0.8525, 0.8400, 0.8494 | 0.8493× |
| Known-policy joint k=1 | 0.8536, 0.8580, 0.8809, 0.8660 | 0.8620× |

The primary threshold required cached joint to reach a median ratio of at least 1.20 and every repetition at least 1.05. Four orders put each cached mode before its corresponding control twice and after it twice. All six modes consumed 1,024 accepted moves/game across 64 games/host, on all configured hosts. Every per-host event stream and final native state matched sequential execution before timing and after each repetition: 6,291,456 timed moves, 196,608 qualification moves and 4,096 separately profiled moves. The first timing stream extends the earlier short qualification stream; three additional streams and all orders were fixed before launch.

The original full-prefix joint mode remained faster than sequential, at 1.994–2.044× in this study. Cached joint was 1.734–1.761× original sequential and 2.041–2.093× cached sequential, while delivering lower absolute throughput than full-prefix joint. Rates were about 94,530–98,698 moves/second for full-prefix joint and 83,275–84,251 for cached joint. A gain over an artificially slower cached control cannot justify replacing the faster full-prefix implementation.

The graph retains KV on device, gathers the native-selected final policy branch, removes hypothetical entries at/beyond the actual root, appends the actual last move and continues paired decoding. Observed counts incorporate only accepted moves. A legality-corrected final move overwrites its speculative counterpart. Inactive game carries remain unchanged. The host coordinator binds reuse to stable game rows, canonical tapes, episodes, model/context identity and ticket generations, and independently checks the selected branch prefix. Go transitions and legal resolution remain in Rust.

Five frozen CPU tests passed in 24.72 seconds. They include nonempty capture-history reconstruction in a fresh process: actual native replay, model reconstruction, exact actions/noise/state inputs and float32 logits within the declared tolerance. This is narrower than distributed checkpoint/resume of ongoing cached self-play. Cancellation and model changes invalidate the coordinator; a nonempty invalid cache requires explicit reconstruction. The trained 9×9 CPU qualification then passed all 12 segments. Its short timing observed 4.11× cached sequential and 1.99× cached joint versus their own full-prefix controls; these were qualification observations, not a CPU performance promotion.

All timing includes required history/commit guards, scheduling, transfers and a final drain of retained device work. Immutable zero carries are allocated with compilation/warmup. Final inspection and event/archive construction are outside segment timing and inside total attempt cost. Independent auditing reconstructs accepted work, cache controls/certificates, inactive/reset handling, events and byte counters, and checks model elements, source/native identities, raw hashes, launch count and budget.

The cost evidence explains why fewer FLOPs did not establish a TPU gain. Guard/drain regions accounted for 24.4% of cached sequential loop time, 11.0% of cached behavior and 16.0% of cached joint. On host 0 the compiler estimated sequential FLOPs falling from 12.48G to 0.228G per executable call, while estimated bytes accessed only fell from 1.000G to 0.973G. For behavior and joint, estimated bytes increased. Dense retained output tensors occupy 172,532,736 bytes/host for k=1 and 690,004,992 for k=4; these stay on device and are not fetched by the host. The estimates and logical sizes are not measured HBM traffic.

The separate host-0 cached-behavior trace captured 46 dispatches on each of eight TPU trace tracks for four local chips. XLA-module intervals occupy 47.76–48.43% of its marked loop, including device computation, copies, barriers and waits. No device counter events were captured. Named collective operations were not found in the retained HLO text, which does not rule out runtime or chip-internal communication. This single-mode profile does not establish MXU utilization, MFU or a controlled utilization comparison with the previous study.

Immutable evidence:

| Stage | Source / attempt | Result |
| --- | --- | --- |
| Native/cache CPU tests | `05e61c7c…` / `runs/qualification/trace-cache-05e61c7c` | Five tests passed, including a fresh child process; log SHA256 `e60413a468bbdafdb27bc8fb5d836b03d6b6b6ff36eb540084aeece5786f1032` |
| Trained-model CPU qualification | `05e61c7c…` / `runs/trace-cache-cpu-05e61c7c` | cpu_result.json (external or omitted experiment artifact), SHA256 `de8da994805bb8a80a108115c72d011363f2c6c3a0bf81c7140aaa7ceca9c79b` |
| multi-host qualification | `c6a57a68…` / `pod-20260911T184532Z-8534090a` | tpu_qualification_result.json (external or omitted experiment artifact), SHA256 `c7efa87739401e3cce8d5b36122b3e0df42b423c2f2279d67dd6369f0c94c541` |
| Repeated timing | `389ed514…` / `pod-20260911T185318Z-6b64484c` | timing_result.json (external or omitted experiment artifact), SHA256 `cb60613c88edf2f4fae06dee6b9f34aae5b253d9c5d0ee4b06ed42717b39a93c` |
| Profile/cost analysis | `36243101…` / same raw timing artifacts | profile_result.json (external or omitted experiment artifact), SHA256 `c20c31f72eb27c674007ca8bf4f35d351b0823bd2b97c2cd64aea1a63b600872` |

Full source/native/config identities are in [cpu_spec.json](cpu_spec.json), [tpu_qualification_spec.json](tpu_qualification_spec.json) and [timing_spec.json](timing_spec.json). The timing registration SHA256 is `68dd967d50ae868e2e7a90b997be49f31fc1a2fd5326dd0a747b79e0d190529f`. Research protocols are read from the workspace and explicitly pinned because source snapshots exclude `research/studies`.

The TPU qualification took 53.8723 seconds / 0.239432 attempt chip-hours, and repeated timing took 218.5689 seconds / 0.971417, totaling 1.210850. These costs are included within the observed reservation ledger, which also includes engineering and idle allocation time. No training or new KataGo games occurred. The fixed student's failed real-KataGo criterion is unchanged. [Next work](NEXT.md) separates compact cache layout, Rust metadata ownership, MCTS integration and the pass/endgame learning diagnostic.
