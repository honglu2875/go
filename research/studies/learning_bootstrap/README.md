The native self-play and plain-JAX learner now complete finite CPU and multi-host TPU training runs. These qualify the pipeline and recovery; they do not confirm hyperparameters, reproduce KataGo training, or demonstrate a stronger method. [The registered 9×9 specification](spec.json) and results with artifact hashes (external or omitted experiment artifact) retain the scope and evidence.

| Run | Completed games | Eligible positions | Global updates |
|---|---:|---:|---:|
| 3×3 CPU integration | 76 | 687 | 88 |
| 3×3, multi-host TPU integration | 1,024 | 10,463 | 87 |
| 9×9, multi-host TPU bootstrap | 1,398 | 209,315 | 418 |

The 9×9 configuration uses 512 concurrent games across the pod, 16 completed MCTS simulations per move plus a separate root evaluation, a four-block width-64 CNN with group normalization and bf16 convolution, terminal outcome targets, and explicit momentum SGD. There are 307,331 parameters. The workload advanced 262,144 real moves and evaluated 4,350,678 active neural positions. Thirty-six move-limit truncations exposed no training rows. Unfinished games remain in the complete checkpoint. The slowest host's training segment took 30.642 seconds, including checkpointing and game-record writes; the whole controller attempt took 55.192 seconds including staging, initialization, compilation and collection. This tiny-budget model is weak.

The 9×9 source snapshot is `7a7745037997ddad53162a92d2fbead438bc9a6235b240e9e973142772618b66`, native binary SHA-256 `55ef050573d12436a5be9d2f0ef2f54a6948eca5eb06318c4492862b4f3329e9`, and final weight SHA-256 `cdf6a6ece4f6cae578a1de5a8e5a404c07239597406682a97e5894f9cf0fa643`. Training artifacts are in `runs/pod-20260911T051928Z-a9a51f16`. The original failed TPU RNG-initialization attempt is also retained; global typed keys now originate from explicitly replicated host key data.

A fresh CPU process restored at turn 48 and reproduced all 51 saved arrays, every actor state, and all 36 subsequent game records through turn 96. A fresh multi-host TPU attempt restored at turn 256 and reproduced all 75 saved arrays per host, all actor states, and 851 subsequent complete/truncated game records through turn 512. The latter is `runs/pod-20260911T052235Z-f88ee7f3`. Timing counters are excluded from equality. This is exact checkpoint continuation on the observed fixed topology; it is not a fault-injection, changed-topology, remote-storage, or automatic recovery qualification.

The complete CNN loss and momentum updates also matched a one-device reference under four-way CPU data partitioning within numerical tolerance. Native worker-count changes preserved subsequent search requests and targets. Shared tests cover corrupted/incomplete checkpoints, stale neural replies, terminal-only targets, ownership of NumPy buffers after native shutdown, GTP deadlines and descendant cleanup.

Four paired-color games used official KataGo 1.18.1 Eigen/AVX2 and the official `kata9x9-b18c384nbt-20231025` checkpoint at `maxVisits=16`, against the candidate's 16 simulations plus root evaluation. The candidate lost all four. Each played board matched between engines. Three numeric score margins differed:

| Opening | Candidate color | Native/raw score | KataGo adjudicated score |
|---|---|---|---|
| Empty | Black | W+85.5 | W+88.5 |
| Empty | White | B+67.5 | B+73.5 |
| C3, G7 | Black | W+88.5 | W+88.5 |
| C3, G7 | White | B+60.5 | B+73.5 |

The original match attempt `runs/eval/bootstrap-3ab60224` remains marked failed because its harness required identical numeric scores. The separate frozen audit `runs/eval/bootstrap-3ab60224-score-audit.json` verifies the raw area scores from retained terminal boards and confirms that all winners agree. The bug was the harness's assumption that KataGo's pass-alive area adjudication is identical to direct terminal-board scoring. The audit did not change game records, training labels or original failures. [Evaluation documentation](../../../eval/README.md) and the actual-game fixture explain the distinction. General rules compatibility and larger held-out strength panels remain open.

The follow-on [local-inference study](../local_inference/README.md) isolates a systems improvement while requiring exact equality of all learned state and game data. It is separate from claims about sample efficiency or playing strength.
