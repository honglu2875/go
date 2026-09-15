The combined source optimizer now passes complete joint policy/value
training, exact fresh-process recovery and native search inference on the
small real-19×19 execution fixture. The [recipe](../../recipes/strong19_source_muon/README.md)
combines standard Muon/AuxAdam, Lookahead, source LR/decay/clipping equations,
print-batch norm snapshots and explicit epoch/subepoch boundaries.

The [runtime check](runtime-cpu-001.json) exercises 76 events against the
pinned KataGo training source's statements, including the cadence change
after 200 million samples, and rejects five inconsistent states. The
separate kernel, schedule and Lookahead reference qualifications remain
linked through the source bundle.

The [neural recovery check](harness-result-001.json) passed in 869.16 seconds.
Eleven uninterrupted updates and a fresh six-plus-five run consumed the
same 19,404 position exposures. All 275 checkpoint arrays (34,071 elements),
host schedule state, RNGs, counters, update metrics apart from timings,
and validation/probe histories match exactly. The final checkpoint manifests
are identical. The six-update boundary retains distinct fast/slow weights
and counter two, immediately before the next synchronization.

Lookahead synchronized at updates 3, 7 and 10. Pre-update norm snapshots
were taken at 3, 6 and 10; epoch flushes occurred at 7 and 11. Each applied
learning rate, decay and clipping cap was replayed from logged source
inputs, independently of the saved final runtime state. All final fast
parameters equal their slow counterparts after the epoch flush.

The [trained-inference check](inference-cpu-001/result.json) passed in 11.51
seconds. All 54 deployed parameter arrays match their training-checkpoint
bytes. Forty-eight policy/value comparisons and two actual Rust searches
passed; maximum cached/full policy disagreement was 6.15e-8. Seventeen
training-only helper arrays remain in the complete training checkpoint.
The existing serving implementation required no optimizer-specific change.

These results qualify execution and continuation of a 9,794-parameter CNN.
They do not establish learning quality, playing strength, full-size TPU
memory use or historical KataGo hyperparameter equivalence. The fixture
uses variable whole-game batches, an explicit 256-position schedule
reference, a synthetic initial sample offset, Lookahead period three and
short epochs. Scientific 19×19 batch/sample conventions and hyperparameters
must still be registered before enabling learning mode.

The registered larger-9×9 learning-rate experiment keeps its original
source, schedule and accelerator allocation. This integration ran only on
auxiliary CPU capacity.
