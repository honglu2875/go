Qualify the slow-weight and clock behavior surrounding the source-qualified
Muon optimizer before integrating the final 19×19 training configuration.
This does not change the running 9×9 comparison or choose any learning rate.

Execute the exact initialization, subepoch reset, post-optimizer averaging and
epoch-flush AST statements from the pinned KataGo trainer with actual Torch
parameters. Compare a pure-JAX port across disabled Lookahead, k=1/3/6,
alpha=0.25/0.3/0.5, irregular subepoch lengths and epoch boundaries. The source
resets the counter at subepoch entry without flushing the fast weights there;
the fast-to-slow alignment occurs after the complete subepoch loop.

Use predetermined fast-optimizer increments, with no neural learning or
optimizer selection. Require parameter agreement within 2e-6 relative or 2e-7
absolute and exact clocks/synchronization flags. Separately require bit-exact
uninterrupted versus fresh-process continuation through a mid-cycle checkpoint,
with every fast/slow parameter and counter retained. Exercise donated execution
to reveal aliasing problems before full model integration. Reject malformed
configuration and checkpoint states. A failure must be retained and diagnosed;
do not loosen numerical bounds to pass it.
