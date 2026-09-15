The larger fixed-data 9×9 comparison has completed its first paired seed and
both CNN runs. The second transformer seed is in progress. In seed 1, CNN has
lower same-update validation KL while the transformer uses less measured
learning time. The paired conclusion and next LR-only intervention remain open.

Standard Muon plus auxiliary Adam now has a qualified functional JAX port.
Its BF16 stage-rounding correction, exact optimizer checkpoint continuation and
source-derived LR/decay/clipping equations are documented under research/studies/.
The existing 9×9 runs retain their frozen AdamW code. These numerical checks do
not reproduce a published checkpoint's undocumented training schedule.

See PUBLICATION.md for the scope of this sanitized source backup. Datasets,
weights, private execution logs and deployment inventories are not included.
