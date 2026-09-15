The first useful systems result came before the transformer learning screen.
The qualified 128x128 Splash tiles were slow on long histories. Replacing
only their tile dimensions with 512x512 passed independent output/gradient
checks and improved attention forward/backward timing by 5.3–6.5x on every
host. Full-model update timing improved from 7.33 to 2.02 seconds at 128
moves, 27.17 to 5.48 at 256, and 59.63 to 11.13 at 384. All sequence buckets
fit HBM, and the model, data, objective and decoding arithmetic were held
fixed. The larger tiles execute some extra padding work, which is included
in the differentiated arithmetic artifact.

Matching at 128 prior moves is not matching at every history. The complete
9x9 transformer decode costs 17.46, 22.76, 37.24 and 56.54 GFLOP of dense
multiply-add work at 0, 32, 128 and 256 prior moves. The CNN costs 37.35
GFLOP throughout. The training corpus has a position-weighted mean of
54.28 prior moves, where the transformer's average hypothetical cached
move cost is 26.11 GFLOP. The 19x19 figures are reported as unmatched.

The differentiated training graph expands all static loops. It counts
convolutions and matrix products, including encoder/head/helper and
rematerialization. The pinned Splash source has two forward, three dQ and
four dKV matrix products per active causal square tile; backward score
recomputation and full diagonal tiles are counted. Optimizer/scalar work,
communication, physical MXU layout and memory transfers are excluded.
For the fixed 1,024-update draw schedule, CNN matrix work is 3.579 EFLOP
and transformer work 2.793 EFLOP with the registered rematerialization.
Without layer rematerialization, the hypothetical figures are 2.168 and
2.264 EFLOP. Those hypothetical programs were traced, not executed at this
batch size. These are arithmetic estimates, not measured MFU or hardware
counters. See training_arithmetic.json and training_matrix_budget.json.

Despite matched reference decoding arithmetic, the first cached transformer
implementation took about 82 ms per warm batch versus about 16 ms for the
CNN. Its compiled temporary allocation was about 11.6 GB per device in
addition to the KV argument. A separate runtime clone carries KV buffers
through the layer loop instead of producing stacked full-cache outputs.
It passes CPU equivalence/rejection tests and exactly preserves the
reference arithmetic. A frozen ABBA TPU probe is prepared for after the
active learning run. No latency improvement is claimed yet.

The first full-size profile failed before training because its Splash
reference was outside an explicit shard_map. The corrected profile and
both full-size tile configurations passed. All closed attempts, including
that failure, remain in the allocation ledger.

Older fully replicated checkpoints were retained and restore-tested before
redundant copies were removed. Newer single-owner checkpoints needed fresh
replicas first; the initial retention attempt correctly stopped without
removal when they were absent. Two persistent replicas were then copied,
verified, restored for qualification, and retained. These are pod-local
replicas, not external backup durability. Maintenance receipts are in
runs/maintenance/.
