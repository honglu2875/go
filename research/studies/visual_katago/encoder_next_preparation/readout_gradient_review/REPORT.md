The original constant readout must obtain board information through attention. Adding the mean of the current board visual tokens creates a direct current-observation path without additional parameters.

| History | Original current-board gradient share | Pooled readout share |
|---:|---:|---:|
| 16 moves | 5.94% | 44.54% |
| 64 moves | 1.98% | 48.37% |
| 128 moves | 0.80% | 42.41% |

These are fractions of the sum of per-frame L2 gradient norms, not fractions of total squared gradient energy. The probe uses one fixed training episode at one initialization, with all context windows ending at the same move. Absolute gradient norms and cross-entropy are retained in the JSON. It is a mechanism probe; training, validation improvement and Go strength remain untested.

Figure (external or omitted experiment artifact) · Raw diagnostic (external or omitted experiment artifact)
