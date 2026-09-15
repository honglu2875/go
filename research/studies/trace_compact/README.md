# Compact KV: faster wide forecasting, primary screen failed

The compact cache passed exact execution checks on all four TPU hosts. It improved learned-opponent k=4 forecasting, but did not beat the faster full-prefix joint path. The registered primary speed criterion failed. Retain the k=4 implementation and the full-prefix joint control; do not promote this as a general rollout or training speedup.

| Compact mode | Four ratios versus its full-prefix control | Median versus full-prefix | Median versus dense cache |
| --- | --- | ---: | ---: |
| Sequential | 0.7660, 0.7851, 0.7756, 0.7719 | 0.7737× | 0.9220× |
| Learned behavior k=4 | 1.3325, 1.3685, 1.3653, 1.3241 | 1.3489× | 1.5893× |
| Known-policy joint k=1 | 0.9698, 1.0238, 0.9594, 0.9738 | 0.9718× | 1.1421× |

The primary threshold required compact joint versus full-prefix joint to have a median ratio of at least 1.20 and every repetition at least 1.05. The observed median was 0.9718. Compact joint was 2.57–2.58× compact sequential, but that comparison uses a slower sequential implementation. Compact behavior remained only 0.873–0.927× full-prefix sequential and 0.428–0.454× full-prefix joint. Full-prefix joint remained about 1.98–2.08× full-prefix sequential.

Nine contemporaneous modes compare full-prefix, dense retained and compact retained execution for sequential, learned-behavior k=4 and known-policy joint k=1. Each consumes 1,024 accepted moves/game across 64 games/host in each repetition. Four orders put every compact mode before and after its corresponding dense and full-prefix controls twice. All 9,437,184 timed moves, every per-game event stream and all final native states matched. The independent audit also reconstructs cache controls, branch certificates, inactive/reset handling and logical byte/work counters. Another 294,912 moves qualified the timing attempt before measurement, and 4,096 were separately profiled. The first timing stream extends the earlier short qualification stream; the next three and all orders were fixed before launch.

The complete `trace_compact` recipe keeps model parameters, `model.py`, the full-prefix decoder and Python metadata coordinator unchanged. `dense_decode.py` is byte-identical to the preceding dense-cache implementation. The compact carry stores committed history once per game, short branch-local append deltas and observed-move counts. Native resolution selects the valid branch prefix. Only deltas strictly before the final accepted action are committed; the actual final action is appended separately, including when legality correction changed it. Inactive game carries remain unchanged; episode resets clear history. No buffers are donated.

Attention reads shared committed KV and branch deltas through a joint masked softmax and separate value reductions. That changes floating-point reduction order, so actual-hardware exact-event checks are required. Six frozen CPU tests passed in 28.55 seconds, covering captures, heterogeneous partial acceptance, corrected actions, inactive/terminal/cap resets, identity/generation checks and fresh-process reconstruction by actual native replay. A separate test compares attention states and KV against full-prefix reconstruction for heterogeneous histories through the last usable token, with finite junk in masked slots. Logit/KV tolerance is 2e-5; sampled actions, noise and board inputs are checked exactly. Cold reconstruction is narrower than distributed checkpoint/recovery of live cached rollouts.

The trained-model CPU qualification passed all 18 segments and 4,608 accepted moves in 33.97 seconds. Its short compact/dense ratios were 1.24 sequential, 2.11 behavior and 1.26 joint, without CPU speed promotion. The separate multi-host TPU qualification passed 589,824 accepted moves and final states. Its short compact/full-prefix ratios were 0.795 sequential, 1.413 behavior and 0.828 joint; the subsequent repeated screen was registered with those observations explicitly retained.

All measured loops include required Python prefix/commit guards, native frame generation, scheduling, dispatch, transfers, native resolution and final retained-work drain. Immutable zero carries are allocated during compilation/warmup. Final inspection, event reconstruction and archive writes are outside loop timing and inside total attempt cost. Model/native/source/configuration identities, compiled artifacts, raw hashes and attempt budgets are verified independently.

| Retained layout / mode, 64 games per host | Logical device carry bytes | Host-0 compiler estimated bytes accessed per call |
| --- | ---: | ---: |
| Dense sequential | 172,532,736 | 972,569,600 |
| Compact sequential | 86,811,648 | 1,725,252,608 |
| Dense joint k=1 | 172,532,736 | 1,798,261,248 |
| Compact joint k=1 | 88,384,512 | 2,047,680,512 |
| Dense behavior k=4 | 690,004,992 | 4,852,317,184 |
| Compact behavior k=4 | 94,675,968 | 2,097,872,896 |

The k=4 carry is 13.7% of its dense counterpart. Smaller logical storage did not uniformly reduce compiler-estimated memory traffic: sequential and joint estimates increased. Compact behavior retained roughly 1.05G estimated FLOPs/call versus dense behavior 1.08G. These estimates are not measured HBM traffic or normalized physical-chip FLOPs. The profile does not isolate the source of extra narrow-branch memory work; inspect lowering/layout before attributing it to a particular operation.

Unprofiled compact loops spent 16.4% of behavior time, 18.8% of joint time and 23.9% of sequential time in cache guard/drain regions. Compact behavior averaged 1.56–1.61 active host CPU cores, versus 1.08–1.12 for dense behavior. Moving those guards into Rust is a separate intervention; subtracting timer regions cannot predict its speed.

The separate host-0 compact-behavior profile captured 46 dispatches on each of eight TPU trace tracks for four local chips. XLA-module intervals occupy 23.23–23.85% of its 0.536-second marked loop. These intervals include computation, copies, barriers and waits. No device counter events were captured. This single-mode shorter profile does not establish MXU utilization, MFU, compute-bound execution or a controlled utilization gain over another layout.

Immutable evidence:

| Stage | Source / attempt | Audit SHA256 |
| --- | --- | --- |
| Six CPU/native tests | `baf5e420…` / `runs/qualification/trace-compact-baf5e420` | `d3f285aa5290a51abfff4ed7e72cbe25e2355527f6c8059f1b3950512e7040dd` |
| Trained-model CPU qualification | `baf5e420…` / `runs/trace-compact-cpu-baf5e420` | a142d0ecd7dd555724cace353bdec7b1b7ff78719bdcca1d4f075fd0984679e7 (external or omitted experiment artifact) |
| multi-host qualification | `51f450f6…` / `pod-20260911T192313Z-7406dd36` | 98c1dec0159959a28eee8c1b39d6869b0095dfa133693829619ec64a940c5eb8 (external or omitted experiment artifact) |
| Four repeated timings | `79094bb3…` / `pod-20260911T192608Z-8daffabd` | cb1a85ac3ea9c8e1cdb74dea2e2a7267db21409c629684fe749e3f2ed18da8a2 (external or omitted experiment artifact) |
| Profile/cost analysis | `79094bb3…` / same raw attempt | d5c581a972d58eee75f8598fdc9c8fc040e7cf2f50a1f39e5a877f8308a72e6e (external or omitted experiment artifact) |

The complete source/native/configuration identities are pinned in [cpu_spec.json](cpu_spec.json), [tpu_qualification_spec.json](tpu_qualification_spec.json) and [timing_spec.json](timing_spec.json). The timing registration SHA256 is `a0870204e1f186cf5c9498631c72de8e88418255f9fc97834813757fd05391b8`. Study files are excluded from source snapshots and explicitly hashed by the analyzers. The six-test log SHA256 is `a34f79c079e4e4a06498e47ed5418096a7397c60a56aaf33d1840ed6abd886a8`.

TPU qualification took 73.1168 seconds / 0.324963 attempt chip-hours; repeated timing and profiling took 311.9312 seconds / 1.386361, totaling 1.711325. These windows are part of the observed reservation ledger, which also includes engineering and idle allocation. No model was trained and no new KataGo games occurred. The fixed student's failed real-KataGo criterion is unchanged. The [next work](NEXT.md) separates the pass/value/search learning diagnostic, Rust metadata ownership, MCTS integration and production requirements.
