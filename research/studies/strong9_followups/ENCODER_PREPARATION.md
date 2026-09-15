Both completed larger9 transformer seeds learned encoder residual scales well
away from their initial value of 1e-6. The independently audited checkpoint
observation is [encoder-scales-001.json](encoder-scales-001.json).

| Residual branch | Seed 1 median magnitude | Seed 2 median magnitude |
| --- | ---: | ---: |
| Convolutional block | 0.04682 | 0.04884 |
| Spatial attention | 0.03052 | 0.02997 |
| Spatial attention MLP | 0.03841 | 0.03760 |

Approximately half the scales are negative. These are learned parameter
statistics, not measurements of the branches' activation contribution or
evidence that an initialization equal to their final magnitude would be best.
They rule out an encoder whose residual scales simply stayed near zero.

An initialization-only probe is straightforward: the frozen numerical source
uses `encoder_layer_scale` only to initialize three parameter tensors. Its
shape, forward path, causal history, spatial readout, auxiliary objective and
optimizer settings can all remain fixed. A larger initial magnitude might
improve early optimization, but that hypothesis still needs a learning run.

[encoder-scale-preparation-001.json](encoder-scale-preparation-001.json) checks
the original 1e-6 and numerical fixtures 1e-3, 1e-2 and 1e-1. All retain exactly
231,181,121 parameters and 37.428121728 billion multiply-add FLOPs per cached
move at batch 128 and 128 past moves, including the complete encoder. The
small float32 fixtures have identical forward/backward StableHLO, finite losses
and gradients, and bit-identical initialization outside the three scale arrays.
These synthetic initialization losses are not learnability results.

After reviewing the [completed negative LR1.5 result](lr15-results-001.md),
the [scale1e-2 review](scale1e-2-review-001.json) selected 0.01. This moderate
initialization is below the observed learned median magnitudes; that motivates
a probe without asserting an optimum. Its immutable registration preserves the
original LR, complete horizon, data draws and decision threshold. Seed 1 has
launched, with a paired second seed conditional on the registered improvement.
Other encoder changes need their own parameter, decoding-cost and numerical
qualification.
