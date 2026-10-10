This report describes the sweep at completion. G8 was subsequently [selected as the research default](PROMOTION.md).

The seven temporal fuzzy TopK trials completed on 2026-10-08 in about 21 hours. All endpoint audits passed. G8 at peak LR 0.0015 is the selected candidate under the prospectively registered selection rule. Dense remains the default.

Each trial used the same canonical game and augmentation sequence, 256 updates, 64 complete games per update, and 7,001,181 position exposures. The fixed validation population contains 152 games / 64,371 positions. Training uses 2,906 games / 1,233,366 unique positions. Test data remained closed.

| Architecture | Peak LR | Parameters | Endpoint policy KL ↓ | Last-three KL ↓ | Value MSE ↓ | Teacher top-choice agreement ↑ | Learning hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense SwiGLU | 0.001 | 232.0M | 0.728001 | 0.774999 | 0.121956 | 48.67% | 2.437 |
| G1 ReLU control | 0.001 | 232.1M | 0.729072 | 0.779350 | 0.120766 | 50.57% | 2.887 |
| G2 | 0.001 | 260.4M | 0.730063 | 0.766682 | 0.146423 | 51.22% | 2.784 |
| G4 | 0.001 | 283.0M | 0.737636 | 0.773753 | 0.141097 | 48.47% | 2.646 |
| G8 | 0.001 | 298.3M | 0.706977 | 0.744699 | 0.116157 | 49.47% | 2.560 |
| G8 | 0.0006 | 298.3M | 0.746353 | 0.781527 | 0.124758 | 49.39% | 2.565 |
| G8 (selected) | 0.0015 | 298.3M | 0.686237 | 0.720701 | 0.123311 | 51.96% | 2.559 |
| G8 | 0.00225 | 298.3M | 0.685562 | 0.729530 | 0.134341 | 51.49% | 2.561 |

G8 at LR 0.0015 reduces endpoint policy KL by 5.74% and the last-three validation mean by 7.01% relative to dense. Teacher top-choice agreement increases by 3.29 percentage points. Endpoint value MSE is 1.11% higher; its last-three mean is 1.63% lower. Recorded learning time is 5.01% higher. This candidate passes all registered provisional quality and health gates.

The same-LR G8 comparison is also positive: LR 0.001 gives 2.89% lower endpoint policy KL and 4.75% lower value MSE than dense. G1 ReLU alone essentially ties dense endpoint KL; G2/G4 do not beat the dense endpoint and fail the value health criterion. These outcomes support testing the grouped feature mechanism further, subject to repeated seeds.

LR 0.00225 gives a marginally lower endpoint policy KL (0.685562), but its value MSE is 10.16% above dense and its last-three policy KL is worse than LR 0.0015. It fails the registered health gate, so the selection remains 0.0015. No arm triggers the registered sustained-overfit detector. This short screen does not establish that overfitting is absent at longer horizons.

G8 keeps the spatial encoder and readout unchanged and replaces all 18 temporal SwiGLU FFNs with grouped ReLU feature selection. Width is 768; each FFN has 683 groups of 8 features (5,464 candidates), selecting at most one positive feature per group. Total parameters rise from 232.0M to 298.3M. Logical active FFN decoding matrix FLOPs differ from dense by +0.0488%; encoder-inclusive cached-move matrix FLOPs are essentially matched.

The implementation still issues masked dense down-projection products. Its temporal FFN issued matrix FLOPs are 1.779 times dense; it does not yet exploit the logical sparsity with a sparse kernel. Synthetic cached decoding at batch 128/history 512 measured 27.11 ms for G8 versus 22.03 ms for dense. These repeated non-donated cache timings are systems checks, not an optimized deployment benchmark or an MFU measurement.

At update 224, the selected candidate reaches KL 0.719560 after 8,072 recorded learning seconds, compared with dense KL 0.728001 after 8,772 seconds at update 256. This is an encouraging observed quality/time point. Dense timing is historical, and the models have different total capacity and selected peak LR; a contemporary paired comparison is needed before claiming a reproducible training-speed gain.

These are single-seed, validation-selected 256-update prefixes of a 1,024-update schedule, not fully annealed convergence, CNN comparisons, or playing-strength tests. The reference here is our dense transformer, not KataGo CNN. Longer matched-data runs and paired seeds should come next, with dense and G1 controls at LR 0.0015 to separate learning-rate and activation effects. Continue to report value quality and wall time, and profile a truly sparse implementation separately.

All seven endpoint and replay audits passed. A final backup-verification timeout was recovered after training; the selected full state was independently reverified on durable storage. Private execution and storage receipts are excluded from this source publication.

Detailed fixed-population curves are in `curves.csv`; the public numerical evidence, selection criteria, system measurements and qualification summary are in `scientific-summary.json`.
