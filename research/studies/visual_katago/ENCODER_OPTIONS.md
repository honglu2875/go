The active learning run changes only the board encoder: two 3×3 convolutions with 64 channels, per-point RMSNorm and SiLU, followed by the original 36 overlapping 2×2 patch projections. The full 1,024-update endpoint must be reviewed before another architecture is selected. This note describes options, not additional launched experiments.

There is prior evidence for testing this direction. *Early Convolutions Help Transformers See Better* found that replacing a ViT patchifying stem with stacked small convolutions improved optimization behavior and image classification accuracy under its tested schedules. That motivates an encoder ablation; it does not establish the outcome for Go, our small board patches, or our batch-norm-free implementation. [Paper](https://arxiv.org/abs/2106.14881)

*CoAtNet* studies staged combinations of convolution and attention across image data scales. Its results support considering how computation is divided between spatial processing and attention, rather than treating a linear image tokenizer as the only transformer input. Its image classification results are not measurements on this fixed Go corpus. [Paper](https://arxiv.org/abs/2106.04803)

A U-Net uses a contracting path for context and an expanding path for localization. [Original paper](https://arxiv.org/abs/1505.04597) For this Go model, one possible adaptation is a 9→5→3 spatial pyramid, an upsampling path with skip connections, and learned token projection from the resulting features. RMSNorm would act independently at each location; no batch statistics or future game observations would enter the encoder. This would be a new model requiring its own qualification, not a drop-in claim of equivalence to the original U-Net.

My design inference is that a plain CNN pyramid is the more direct first option when the output is a small token grid. A full U-Net decoder earns its extra work only if restoring fine spatial detail improves the policy representation. A U-Net becomes especially natural if a later model also uses spatial policy or ownership outputs; those auxiliary tasks are outside the current single-policy ablation.

Token count is a separate architectural choice. The earlier width-1,024 transformer used nine non-overlapping 3×3 patches; the current width-768 reference uses 36 overlapping 2×2 patches. Those earlier and current studies differ in more than tokenization, so their learning results cannot identify the best patch count.

The following untrained analytical examples keep the present 34-layer width-768 trunk and two-layer convolutional stem. They include dense encoder, trunk, attention and head arithmetic at 9×9 and 128 prior moves. The smaller token grids use hypothetical 3×3 projection footprints; their exact sampling layouts are not implemented or numerically qualified.

| Visual tokens | Complete dense GFLOPs per move | Allocated KV cache, batch 128 / capacity 512 |
|---:|---:|---:|
| 9 | 6.793 | 23.375 GiB |
| 16 | 12.809 | 38.250 GiB |
| 25 | 22.484 | 57.375 GiB |
| 36, current qualified model | 37.256 | 80.750 GiB |

These are not equal-FLOP comparisons or latency predictions. Reducing tokens saves both historical attention work and cache storage, while changing the information bottleneck. If the goal remains matching both the CNN's parameter count and decoding budget, the saved arithmetic must be assigned to a concrete encoder/trunk design and counted again.

An analytical search within the present fixed-width/fixed-head family illustrates the constraint: with the same tiny stem, 25 tokens can approach both budgets only by making the trunk much deeper and its feed-forward layers narrower. Nine or 16 tokens had no matched candidate within the searched 1–96 layers and feed-forward widths 8–16,384. This is a restriction of that family, not an impossibility result for richer CNN/U-Net encoders, shared-weight refinement, different heads or other architectures. It also does not validate those deep/narrow candidates; scalar work, compiler padding, memory and learning remain untested.

Analytical configurations and scope (external or omitted experiment artifact) · [Current registered encoder experiment](conv_registration.json) · [CNN versus linear-transformer result](LINEAR_REVIEW.md)
