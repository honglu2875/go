# Paired decoding: 2.04× known-policy speedup, slower learned forecasts

The registered multi-host timing screen passed for known-policy joint continuation. Three counterbalanced repetitions measured 2.0119×, 2.0352× and 2.0644× complete packet-loop throughput versus one-step execution. Learned-opponent k=4 forecasting measured 0.6518×, 0.6471× and 0.6661× and was slower. Every compared per-host event stream and final native state matched exactly. This is fixed-model policy execution, with no MCTS, training, strength or MFU claim.

The execution control uses float32 activations and highest matmul precision, including the heads, with the same trained parameter elements and byte-identical model source as the earlier bfloat16 study. That study failed TPU action equivalence and remains [audited](../trace_throughput/README.md). The combined precision change does not isolate the responsible operation, and this result does not establish bfloat16 equivalence.

| Mode | Median speed ratio | Pod accepted moves/second across repetitions | Accepted moves/game/dispatch, pooled |
| --- | ---: | ---: | ---: |
| One-step sequential | 1.000× | 48,490–49,557 | 1.000 |
| Learned behavior, four samples/four plies | 0.652× | 31,588–32,299 | 1.599 |
| Known policies and actual draws, one sample/four plies | 2.035× | 99,343–100,101 | 3.230 |

The known-policy mode has both actual self-play policies and draws. The learned-behavior mode independently forecasts the opponent. Their information differs; the known-policy result does not demonstrate a speedup against an unknown or weaker opponent.

The native bounded batch consumes an exact accepted-move allowance per game, preserves inactive games, and checks model/context tickets, actual prefixes and the board behind every consumed policy row. Terminal/cap outcomes retain their meaning at work boundaries. Timing uses 64 games per host, 1,024 accepted moves/game/mode, three Latin-order repetitions and local-device inference on all configured hosts. Slowest-host elapsed time defines each pod rate. Timing covers 2,359,296 accepted moves across modes, with another 98,304 qualification moves and 4,096 separately profiled moves. The first timing stream extends the earlier short qualification stream; two additional streams and the counterbalanced order were fixed before launch. This is an exploratory systems screen.

Compilation and warmup precede measured loops. Required Python/native coordination, inference and materialization are inside timing; final-state inspection, event reconstruction and archive writes are outside the segment and inside total attempt cost. Every repetition rechecks events and final states against its own sequential reference. The primary criterion, registered before launch, required a joint median ratio at least 1.20 and every repetition at least 1.05. All three repetitions passed.

The unprofiled counters explain why dispatch count alone was misleading. Learned forecasting reduced dispatches to 62.6% of sequential, while performing 10.01× as many board-encoder rows and returning 7.88× as many logical tensor bytes. Its inference/materialization region consumed 80.7% of loop time. Known-policy continuation reduced dispatches to 31.0% with 1.24× the board encodes and output bytes. Only 18.6–19.6% of the fixed prefill token slots represented active history across these modes. Host processes averaged 1.08–1.13 active cores for learned forecasting, 1.41–1.44 for joint continuation, and 1.89–1.96 sequentially; these counts include runtime threads, Python and Rust.

A separate host-0 behavior-mode trace captured 46 dispatches and 4,096 accepted moves, with its action prefix independently matched to the timed reference. All eight TPU trace tracks recorded 46 XLA modules. Their module windows occupied 58.32–58.58% of the marked packet loop. These tracks correspond to four local dual-core chips. Module occupancy includes computation, internal copies, barriers and waits. No device counter events were captured, and this single-mode profile does not establish MXU utilization, MFU, a compute-bound workload, or a cross-mode utilization gain. The raw Chrome trace and XPlane protobuf remain under the attempt's rank-0 artifacts.

Evidence and immutable identities:

| Stage | Frozen source / attempt | Audit |
| --- | --- | --- |
| CPU qualification | `62080382…` / `runs/trace-precision-cpu-62080382` | cpu_result.json (external or omitted experiment artifact), SHA256 `be5ede50c49c9196e8dceabd85db3c3829c38ff3f0cbc5a739fe86c8686bfde8` |
| multi-host qualification | `eaf85f4d…` / `pod-20260911T180202Z-4020c3fb` | tpu_qualification_result.json (external or omitted experiment artifact), SHA256 `4e82b1c052f7207836e57355072c2c7bdad284af01a8fd38ceada37c6b21b074` |
| Repeated timing | `dca02187…` / `pod-20260911T180514Z-ed7b00c5` | timing_result.json (external or omitted experiment artifact), SHA256 `7552235fa540a887c7b8455f1a2ba78acac21ef1e314cdf76a481d2b38ad03b9` |
| Profile/cost analysis | `492e51cb…` / same timing artifacts | profile_result.json (external or omitted experiment artifact), SHA256 `d96605bb9844fa7ba0612659b999c0a99a5c27bb57e1204a9baaecf3fe5faff1` |

The full source identities, configuration/native hashes and prospective limits are in [cpu_spec.json](cpu_spec.json), [tpu_qualification_spec.json](tpu_qualification_spec.json) and [timing_spec.json](timing_spec.json). The timing protocol SHA256 is `184b2d4e788e5340bf82fe3be59589d0bc4efb452aaba932fd1da610c4145862`. Analyzers verify frozen source, native binary/receipt, trained parameter elements, raw event reconstruction, exact work, byte counters, final states, result/profile hashes, launch count and attempt budget. Research protocol files are read from the workspace and pinned explicitly because source snapshots exclude `research/studies`.

The float32 qualification took 34.5518 seconds / 0.153564 attempt chip-hours, and repeated timing took 98.6919 seconds / 0.438631. Including the retained bfloat16 failure, the three new pod attempts used 0.718238 chip-hours. This is a subset of the observed reservation ledger (external or omitted experiment artifact), which also includes engineering and idle time; it is not the total allocation cost.

The fixed 939,968-parameter 9×9 student still failed its real-KataGo strength criterion. This execution screen does not promote it to production. [The next independent intervention](NEXT.md) is retaining committed history KV on device, followed by numerical and matched-work validation. Native worker scaling, MCTS integration, pass/endgame learning diagnostics and production durability remain separate work.
