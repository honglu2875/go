On this example, the original C128 model develops a stronger current-board gradient path during training. At 128 moves, its share of the sum of frame gradient norms rises from 0.805% at initialization to 25.090% after 1,024 updates. This limits what the initialization probe alone can establish.

| History | At initialization | After 1,024 updates |
|---:|---:|---:|
| 16 moves | 5.944% | 34.138% |
| 64 moves | 1.983% | 33.688% |
| 128 moves | 0.805% | 25.090% |

One fixed training example at the same endpoint, using the audited final control checkpoint. Shares use the sum of per-frame L2 gradient norms. They are not attention probabilities or a causal attribution. The shorter contexts truncate the original history and can change the input distribution. This is a mechanism diagnostic, not a held-out learnability comparison. Judge the actual readout intervention by its completed validation curve.

Figure (external or omitted experiment artifact) · Trained diagnostic (external or omitted experiment artifact) · Initialization diagnostic (external or omitted experiment artifact)

This publication corrects labels and clipping in the first figure; the diagnostic data are unchanged.
