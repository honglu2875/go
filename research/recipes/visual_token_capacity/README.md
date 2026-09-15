This recipe compares two-layer visual CNN encoders on the user-selected peak learning rates **1e-4, 3e-4, 6e-4**. It is cloned from the frozen conv64 transformer `24b261c87179b78e36d60050c14027030bb3a122d872fa703f886638fdc0f848`.

Each encoder uses 3×3 SAME convolutions, stride 1, channels 22→C→C, per-point RMSNorm and SiLU, followed by the original overlapping 2×2 patch projection. There are 36 visual tokens per 9×9 board. Decoder width remains 768 with 34 causal layers, 12 query / 4 KV heads and one tied policy head. There is no batch normalization.

| Encoder channels | SwiGLU hidden width | Total parameters | Encoder parameters |
|---:|---:|---:|---:|
| 64 | 2304 | 234,588,416 | 290,816 |
| 128 | 2296 | 234,281,728 | 610,816 |
| 256 | 2288 | 234,516,224 | 1,472,000 |

The small feed-forward width adjustment keeps total parameter count and complete neural decoding arithmetic within the registered 1% tolerance of the CNN at 9×9, global batch 128 and 128 past moves. These are joint encoder-capacity / compensated-decoder comparisons; actual TPU padding, latency and memory must be measured separately. Other histories, 19×19 and full training FLOPs are reported separately and are not matched.

The 64-channel 1e-4 and 3e-4 cells reuse completed audited runs after verifying the unchanged 64-channel implementation. Train its missing 6e-4 control, then the 128-channel grid. Review that stage before launching the conditional 256-channel grid. Keep all three rates for each encoder and compare both common-rate endpoints and each encoder's best in the same grid. A boundary winner is evidence to review the range, not a globally optimal LR.

The dataset, complete histories, episode/D4 draws, seed, one policy loss, 1,024 updates, global batch of 128 sequences, 64-step warmup, and cosine end/peak ratio of 0.3 are unchanged. The cached runtime experiment stays separate. The current CNN LR sweep completes before any capacity TPU work.

`conv64_reference.py` is the exact frozen parent for numerical equivalence checks. `test_capacity.py` checks the configurable encoder against independent convolution and gradient references, causal/cache behavior and unchanged initialization. `qualify_budget.py` traces the complete decoder including its encoder; full-size TPU qualification remains required for each changed architecture before training. The JSON configurations are concrete candidates, not evidence that they have run or been queued.
