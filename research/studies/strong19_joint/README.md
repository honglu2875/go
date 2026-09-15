This is a preparation prototype for the later 19×19 joint policy/value study.
It does not change the running larger-9×9 registration or select its winner.
`provenance-001.json` records the copied source identities; the model is a
separate clone in `research/recipes/strong19_joint/`.
`source-bundle-001.json` seals the tested recipe, value-head reference operators,
shared library and dependency locks in `source-001.tar` (389,120 bytes). Every
archive member was read back and verified. This is a source archive, not a
hermetic runtime or a registered learning snapshot.

The CNN computes its existing trunk once and feeds its final spatial features
to policy and the qualified KataGo value branch. Its existing training helper
gets a separate value branch alongside the separate helper policy. The
transformer reads policy and value from the same normalized causal latent;
its full and first-pass outputs share both heads. No value readout invokes an
extra board encoder. Existing backbone and policy initialization draws are
preserved.

The joint objective takes an explicit policy/value coefficient. Test fixtures
use 0 and 0.7 to check isolation and gradients; these are not proposed or
registered training hyperparameters. Existing policy auxiliary weights are
retained in the prototype (CNN 0.8, transformer 0.25) and also applied to the
corresponding value fixture. Main inference discards training-only CNN helpers.

`cpu-qualification-001.json` passed seven tests in 170.59 seconds. These cover
all backbone gradients, unequal counts across four simulated devices,
future-action isolation, and cached main/first-pass policy and value equivalence.
The prototype rejects stale-cache value results
and returns zero on inactive cache lanes, avoiding a plausible value caused
only by the head's bias.

`budget-qualification-001.json` traced the full width-768 models without
allocating their parameters or activations. At batch 128 and 128 prior moves:

| Quantity | KataGo CNN | Temporal candidate |
| --- | ---: | ---: |
| All trainable parameters | 233,220,870 | 232,011,540 |
| Deployed parameters | 232,529,283 | 232,011,540 |
| Complete decode multiply-add GFLOPs per move | 166.586066432 | 167.576019584 |

All three registered budget comparisons remain within 1%: trainable parameters
−0.519%, multiply-add work +0.594%, unit-cost floating operations +0.669% for
the transformer. This includes its full two-pass encoder, policy and value
readouts. Training-only CNN helper parameters are counted, while their compute
is excluded from the deployed decoder. Longer history extents through 1,535
prior moves are also recorded; the 1% match refers to the declared 128-move
reference, not every history length.

The differentiated graphs were traced at local batch 8 with 128, 512 and 1,536
board positions. The largest individual abstract array at the longest extent
is 13,627,293,696 bytes for CNN and 2,839,019,520 bytes for the compact
transformer path. These are individual graph shapes, not peak HBM. In
particular, the CNN may need bounded helper-statistic computation before the
longest histories are practical. No full-size TPU memory result is implied.

The separate `cnn_chunks.py` path now computes centered moments in bounded
chunks, merges them across the complete normalization population, then
recomputes chunks to emit readouts. `cnn-chunks-cpu-001.json` passed five tests
in 74.73 seconds, including every coupled backbone/head gradient, two empty
simulated shards, padding and a float64 reference for centered moments. The
moment reduction reorders float32 arithmetic; agreement is within declared
tolerances, not bit identity.

`cnn-chunks-memory-001.json` traced chunk sizes 8, 16 and 32 at each registered
history extent. At 1,536 positions, the largest individual abstract array falls
from 13.63 GB to 0.39–0.46 GB. The cost is one extra trunk pass: forward matrix
work is 1.997× and differentiated matrix work 1.614× the materialized reference.
This is a candidate for fitting long histories, not a measured speedup. Its
actual peak memory, optimizer equivalence and latency on TPU are still open;
it has not been adopted for training. Deployed parameters and inference remain
the qualified joint model above.

The subsequent `../joint_inference/` study passed native V7/Rust identity checks
and joint cached/full-history comparisons through real Rust MCTS leaves with
small CPU models. It adds an inference runner without changing these qualified
model functions. Complete-history identity and branch rewinds are tested; full
model TPU serving and actual playing strength remain open.

Required work before training: full-size TPU memory/optimizer qualification
and immutable data, objective and optimizer registration. Long-history CNN
helper activations need measured memory and latency qualification; the
transformer's compact grid optimization does not solve that separate
requirement. The corpus supplies signed raw value, not all of KataGo's
original auxiliary targets or three-way probabilities. Real KataGo strength
checks follow trained checkpoint export and serving qualification.
