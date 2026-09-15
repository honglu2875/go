Before a later scalar experiment inherits an accepted encoder change, check
the complete two-seed outcome and the exact candidate audit for each seed.
One passing seed is not an accepted parent. Retain the original controls.

Inspection of the current operators corrects an earlier planning assumption:
`register_scalar.py` already takes each parent from the review's `parent_audits`,
and `scalar_source.py` accepts any pair whose frozen sources are identical and
whose configurations differ only in the declared scalar. Neither helper has
a hard-coded parent snapshot list. No registration rewrite is needed merely
to clone a newer parent.

The remaining restriction is in `launch_scalar.inspect`: it allows the
initializer scale to differ from the full-shape qualification only when the
current mechanism is `encoder_scale`. A later `lr_floor` trial inheriting a
changed encoder scale would fail that qualified-model comparison. It needs
an explicit qualification transfer through the already verified
initialization-only evidence, while still rejecting every other model/source
change. Do not bypass the comparison by relabelling the intervention, replacing
the qualification's recorded model, or silently resetting the encoder scale.

Any such operator change must use a new version after the active continuation
finishes, with the accepted paired parent and its initialization qualification
pinned explicitly. The operators currently owned by the encoder continuation
remain unchanged. No following rate, parent or intervention is selected here.
