This isolated pure-JAX recipe combines the qualified joint CNN loss with
standard KataGo Muon/auxiliary Adam, Lookahead, source LR/decay/clipping equations,
print-batch norm snapshots and explicit epoch/subepoch boundaries. It retains
complete fast/slow weights, moments, dynamic scalars, host schedule inputs and
all sampler/diagnostic state for fresh-process continuation.

The actual globally live position count converts mean loss gradients to sums
and advances sample progress. The schedule reference batch is separately explicit.
The initial CPU fixture uses a 256-position reference with variable complete
histories, an offset sample clock and short epoch/print/Lookahead periods to
exercise transitions. These are execution fixtures, not scientific choices or
claims to reproduce published fixed-batch training. Learning mode remains gated
until the 19x19 population, batch convention and hyperparameters are registered.

The original source prints norms before the fast update, overwrites its latest
snapshot at print cadence, refreshes settings at epoch entry and every five
batches through 200M samples (50 afterward), resets Lookahead clocks at subepoch
entry and copies slow to fast weights at epoch end. Epoch copies run outside
the donated executable to preserve independent fast/slow storage. Dynamic group
settings transfer together as one small vector. The numerical CNN and joint
policy/value objective are unchanged from the qualified parent.
