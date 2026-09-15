This clone owns a fixed-input inference probe and the unchanged learned CNN
implementation. It does not perform training or board transitions.

Each host reads a pinned committed Gumbel CNN checkpoint, samples real replay
positions without replacement, and runs identical parameters and inputs in
three modes: serial small forwards with individual host fetches, the same small
forwards queued before one fetch, and one wide forward/fetch. Actor-major row
ordering matches the root-prefetch recipe. Timings include transfers and host
output assembly; compilation, warmup and numerical analysis are recorded
separately. The three timing orders are fixed in the configuration.

Raw input/output arrays, checkpoint identities, compiled HLO, per-mode
dispatch/fetch counts, and CPU/wall times are retained. Legal policy divergence,
top-action disagreements and value errors characterize wide-batch numerical
changes. Queuing the same executable must preserve every prediction exactly.
Wide output finiteness is required, but changed values do not imply rejection.
Actual learning and MCTS effects require separate studies. This probe has no
padding or native work and cannot establish rollout speedup or production MFU.
