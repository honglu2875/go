# Current-corpus CNN and temporal-expert comparison

This clone retains the qualified temporal-MoE learner and the existing BN-free
nested CNN path. It changes only the systems-qualification dispatch to honor
each architecture's existing padding option; mathematical training files are
unchanged. Sparse layers still use the original qualified grouped kernel.

The study in `research/studies/strong19_moe_comparison` freezes independent
CNN and temporal-only expert configurations on the current recovered 19x19
corpus. Existing dense-transformer records provide the matched reference.
The old-corpus CNN losses remain separate. The shared 64-update comparison
also includes the completed all-expert MoE; the registered new horizon is 128.
