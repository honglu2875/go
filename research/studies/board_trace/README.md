# State-validated paired execution

The registered CPU execution qualification passed in greedy and stochastic play. Every speculative mode reproduced its one-step sequential action prefix exactly, including terminal scores and episode resets. Both registered mechanism thresholds passed. This is a systems mechanism result using a fixed student that failed its real KataGo strength criterion; it does not promote that model or establish throughput.

| Mode | Greedy moves/game/dispatch | Stochastic moves/game/dispatch |
|---|---:|---:|
| Sequential | 1.0000 | 1.0000 |
| Initialized behavior forecasts, k=4 | 1.0469 | 1.1055 |
| Trained behavior forecasts, k=4 | 2.0879 | 1.6953 |
| Known policy, independent forecasts, k=4 | 3.3984 | 1.8086 |
| Known policies and actual self-play draws, k=1 | 3.3984 | 3.3262 |
| Hold root board, actual self-play draws, k=1 | 1.0156 | 1.0234 |

Each speculative mode runs four games for 128 packets with horizon four. A separate one-step baseline covers every resulting per-game prefix: 1,740 reference events in the greedy condition and 1,776 in the stochastic condition. Modes share histories; these counts must not be treated as independent samples. Greedy game IDs and restarted episodes can be identical. No confidence interval over independent opponents or training seeds is implied.

The learned-behavior condition must reach at least 1.1 moves/game/dispatch and the known-policy joint condition at least 1.25, in both sampling settings, alongside exact prefix equivalence. Those thresholds passed. The known-policy independent and joint controls coincide in greedy play because the own-policy temperature is zero. In stochastic play, knowing a probability distribution and knowing the actual self-play draw are materially different information conditions. Actual external opponents' future random draws are unavailable.

## Why board validation matters

The trained model combines a canonical action-history transformer with a separate per-position board encoder. The new decoder retains that exact trained model implementation and generates both views and all sampled branches inside one JAX scan. It proposes future boards with a cheap place-stone update; pass leaves the board unchanged. These proposals intentionally omit captures, suicide and superko.

Rust requires the model/context/episode ticket, preceding actions and the current prediction's board input to agree before consuming an own-policy row. The active player's expert logits and its counter-based actual draw select the move; opponent forecasts only determine whether another continuation is available. A legal proposed move is insufficient if its policy was computed from the wrong board. The capture test accepts the capturing move, then stops before a subsequent legal move because its model input still contains the captured stone.

In the learned-behavior run, state mismatches stopped 77/512 greedy and 54/512 stochastic game packets. In the known-policy joint run they stopped 156/512 and 140/512. Those stops are required to preserve the policy. Holding the original root state produced 504/512 and 508/512 state-mismatch stops, explaining its near-one-move acceptance. The hold control undergoes the same validation; it never silently consumes a stale-board policy.

The actual one-step and all speculative prefixes matched after these stops, including legality corrections, terminal scores and resets. No capped self-play episodes occurred in the recorded modes. The recipe performs no MCTS: expert parameters were distilled from earlier MCTS data, while observed-behavior parameters remain a separate forecast objective.

## Validation, identities and cost

Five Rust trace tests and all 19 Python native binding tests passed. They cover capture cutoffs, terminal/cap semantics, stale tickets, batch-wide atomic rejection and independently owned root arrays. Four JAX tests independently reconstruct every proposed input state, compare all reported logits with fresh causal prefixes, test pass/root masking, and verify shared actual draws across packet boundaries. The final analyzer reconstructs event streams and work counts from packet records and checks every loaded parameter element, model/native/config identity, compiled HLO hash and attempt log.

- [Registration](spec.json): `33c76122ecbd9dd9955665d18858b62bbe45c4e741e05ae49fb789a48152ceb6`.
- Final audit (external or omitted experiment artifact): `3e16e9f3f96fbdc664f8b6e46e69c2a41bc6f618337c23b75d3e64cf57ee60e8`.
- Actual execution records (external or omitted experiment artifact): `a5086edd3fd238638dff24254e29ec72eeaac7cfbb3486bf2a47b0b2e3faa5cd`.
- Greedy source `356a8035d8f3caaa7fcc8bbf516153e6f50bbe4de9d64dafc3bef8cf049248af`; output `runs/board-trace-greedy-356a8035`.
- Stochastic source `b8712ffe3d0a6b63adc703c7263c55d83481bba039e8cd47ac153888e446a42b`; output `runs/board-trace-stochastic-b8712ffe`.
- Analysis source `ac9cb05023595929bd57f85099e20fad063a8164b3b3762095404f9ca62631b4`.
- Fixed exact-board student: `c865322774cf333073f05413a41b645116adc788a916c6c40b6855a6b28b7e13`, 939,968 parameters. Its training and failed strength evidence remain in [board_state_distillation](../board_state_distillation/README.md).

The first controller used a read-only existing-artifact path helper for a new output directory and stopped before launching a condition. Its failed record (external or omitted experiment artifact), original operator (external or omitted experiment artifact) and correction (external or omitted experiment artifact) remain unchanged. The corrected operator used fresh output containment checks and launched each registered condition exactly once. The analyzer independently checks this attempt census; no model-execution retry occurred.

The greedy and stochastic conditions took 64.24 and 63.93 CPU wall seconds, including compilation and all controls. No TPU training was added. Instrumented modes have unequal real move counts, branch work and input/output payloads, so these times cannot establish a speedup. For example, one k=4, four-ply mode returns 7,438,336 logical array bytes over 128 packets, of which 1,327,104 bytes are proposed states; the known-policy joint k=1 mode returns 2,363,392 bytes. These are tensor payload counts, not measured hardware-link traffic. The native resolver remains serial and each packet prefills the full real history. The next study must match accepted work and measure TPU and CPU execution before making a throughput or MFU claim.
