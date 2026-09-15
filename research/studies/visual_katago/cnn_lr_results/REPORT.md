Both models completed the same four-rate AdamW sweep. The CNN's lowest final validation KL is **0.420361 at 1e-03**; the CNN-encoder transformer's is **0.540367 at 3e-04**.

| Peak LR | CNN KL ↓ | Transformer KL ↓ | CNN top-move agreement | Transformer top-move agreement |
|---|---:|---:|---:|---:|
| 3e-05 | 0.661511 | 0.996168 | 57.92% | 49.14% |
| 1e-04 | 0.513569 | 0.640906 | 62.83% | 58.84% |
| 3e-04 | 0.439092 | 0.540367 | 65.52% | 61.93% |
| 1e-03 | 0.420361 | 0.678066 | 66.66% | 57.49% |

Each cell completed 1,024 accepted updates. Independent audits verified checkpoint arrays, identical initial weights within each architecture, identical episode/D4 draws across both architectures, and the same complete validation population. Each run saw 11,469,333 position exposures from 836,486 training positions; validation contains 102,339 positions from 1,170 games. The test split remains closed.

Only peak LR and its proportional endpoint changed within each architecture. Warmup remains 64 updates, cosine ends at 30% of peak, and the remaining AdamW settings are unchanged. Each architecture reuses its existing 1e-4 endpoint. The CNN retains its same-target training helper; the transformer has one policy head.

Our shared AdamW training recipe is an adaptation, not KataGo's official production optimizer schedule. This comparison tunes both models over the same LR grid before further encoder-capacity changes. Individually selected endpoints include validation-selection uncertainty and do not establish a global optimum, convergence, Go playing strength or RL improvement.

CNN learning curves (external or omitted experiment artifact) · Selected endpoint learning curves (external or omitted experiment artifact) · Selected phase errors (external or omitted experiment artifact) · CNN analysis (external or omitted experiment artifact)
