The original C128 model develops a stronger current-board gradient path during training on this example. At128moves, its share of the sum of frame gradient norms rises from0.805% at initialization to25.090% after1024updates. This limits the inference that the initialization probe alone can support.

| History | Original at initialization | Original after1024updates |
|---:|---:|---:|
| 16 | 5.944% | 34.138% |
| 64 | 1.983% | 33.688% |
| 128 | 0.805% | 25.090% |

One fixed training example at the same endpoint, using the audited final control checkpoint. These are input-gradient norms, not attention probabilities or a causal attribution. The shorter contexts are truncations of the full history and can change the input distribution. This is a mechanism diagnostic, not a held-out learnability comparison. The actual readout intervention must be judged by its completed validation curve.

Figure (external or omitted experiment artifact) · Trained diagnostic (external or omitted experiment artifact) · Initialization diagnostic (external or omitted experiment artifact)
