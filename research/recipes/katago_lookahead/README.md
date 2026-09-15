This pure-JAX component implements the pinned KataGo trainer's slow-parameter
averaging, synchronization counter and explicit subepoch/epoch boundaries.
The configuration is either a positive period with alpha in (0,1), or both
fields set to None. The upstream CLI resolves alpha=1 to disabled Lookahead.

Call `after_step` after the fast optimizer. The caller owns its optimizer
moments and accepted-update decision. `begin_subepoch` resets only the counter.
Call `end_epoch` at the host epoch boundary, outside the donated training
executable: it copies the slow values into independent fast buffers. Returning
the same buffers for both trees breaks the next donated update.

The separate checkpoint codec is an execution fixture for fast/slow arrays and
the counter. Integrating these with Muon moments, source schedule/norm cadence,
sampler state and full-model recovery remains required. This component alone
does not reproduce a published training configuration.
