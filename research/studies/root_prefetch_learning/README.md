The wide root-prefetch learning candidate did not establish noninferiority and is not promoted. All 584 scheduled games were recorded; 141,819 boards and all 553 completed scores matched KataGo. No game had a process or integrity failure.

| Evaluation | Wins | Losses | Unassigned caps |
| --- | ---: | ---: | ---: |
| Direct versus the reused serial Gumbel CNN | 233 | 250 | 29 |
| Historical small KataGo anchor | 44 | 20 | 0 |
| Strong KataGo anchor | 0 | 6 | 2 |

The primary scheduled-score bounds are [0.45508, 0.51172]. The conservative 95% outer interval across 256 opening pairs is [0.37020, 0.59660]. Its lower bound does not exceed the registered 0.40 threshold; 483/512 completed games also misses the 95% completion requirement. This is a failed noninferiority screen, not proof of inferiority. Caps were never assigned draws.

| Whole-training diagnostic | Serial control | Wide candidate |
| --- | ---: | ---: |
| Completed self-play games | 123,941 | 86,478 |
| Capped self-play games | 479 | 14,948 |
| Terminal replay rows produced | 16,589,675 | 11,905,917 |
| Actual neural evaluations | 280,537,570 | 280,330,112 |
| Padded neural slots | 285,212,672 | 480,808,704 |
| Host inference fetches | 2,228,224 | 1,790,238 |
| Slowest training segment, seconds | 1283.382 | 1297.204 |
| Largest host inference total, seconds | 676.844 | 625.003 |
| Largest native total, seconds | 326.165 | 353.999 |
| Largest learner total, seconds | 236.059 | 247.321 |

Both runs advanced 16,777,216 real moves and used 32,695 updates / 33,479,680 global learner-row exposures. The candidate consumed all 227,124,216 prefetched predictions, so unused cache predictions were zero, while padded neural work grew substantially. The reused control ran earlier; these timings are diagnostic and cannot establish a repeated systems speedup. Smaller isolated inference latency did not translate into lower whole-training time in this run.

Training source: `1282c598ac47d926303af6c96535a015f405968d3f159da4c6d4d7c0b589d611`. Attempt: `pod-20260911T134557Z-319fa735`. Model: `c06a6fac8a13da82126eea10e0b64d5e82177274d1a079562f4d7a99644d067b`. New training cost: 5.895689 recorded attempt chip-hours. New evaluation wall time: 1,878.829 seconds, within the registered 5,400-second limit.

The read-only protocol is `spec.json`; result is `result.json`. Evaluation source: `62bc7ba2167c20a08a046c7a1ca5472ec0e6dd2e1fd7a16a8cc05ca66b179bdd`; analysis source: `5d3eb17f634b86774c5a3257895ff9b38c8d36ba1023287c227238f460845b82`. Raw games, SGFs, transcripts and counts remain under `runs/eval/root-prefetch-learning-d7d45556`. Earlier analyzer errors are retained in `runs/analysis`; they did not rerun or alter games.

The eight primary direct specifications were individually hashed before training. The secondary KataGo parent panels and weight identities were registered, but child configuration hashes were omitted. Their full inputs were frozen before evaluation and are now audited in `evaluation_input_audit.json`; this post-execution receipt does not retroactively repair registration. Future studies should register and verify the complete input closure with `gozero.evaluation_inputs`.

Keep the serial Gumbel CNN as the control. Any further wide execution study needs independent-seed learning evidence, consistent per-game completion, and matched total-work timing. There is no production promotion or superiority claim over current KataGo.
