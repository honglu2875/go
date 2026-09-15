This recipe integrates the reference-qualified standard Muon/AuxAdam kernel
with the unchanged joint policy/value CNN and its bounded training path.
Complete state includes Muon momentum, auxiliary Adam moments, optimizer step,
all sampler RNGs, counters, validation/probe history and per-rank state.

The initial configuration is an execution qualification with explicit fixed
per-group LR/decay values and a clipping cap in global summed-gradient units.
The joint objective is position averaged, so the bridge multiplies gradients by
the global live-position count before RepVGG scaling and clipping. This is not
a scientific learning-rate choice or a reproduction of published training.

Source-derived schedule updates, running norm observations, lookahead averaging
and their complete continuation state remain to be integrated before registering
the final KataGo-informed 19×19 learning comparison. The initial fixture permits
only `purpose=qualification` and leaves the running 9×9 comparison unchanged.
