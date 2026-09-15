The pure-JAX CNN value readout passed an independent comparison with the pinned
official KataGo implementation. This qualifies the readout, not a joint learner.

The branch is a bias-free 1×1 convolution from 768 to 256 channels, learned
channel bias and Mish, the three KataGo value pooling statistics, a 768→256
affine layer and Mish, then a 256→3 affine layer. It has 394,499 parameters.
Value is `softmax(logits)[win] - softmax(logits)[loss]`, from the player-to-move
perspective. The full three-way teacher distribution is not present in our data.

`reference-001/manifest.json` pins the official source, initialized tensors,
five differently masked boards and all active/input gradients. The latest
`cnn-reference-qualification-002.json` passed all 18 checks: maximum logit
error was 4.77e-7 and maximum active parameter-gradient error was 1.49e-7.
Its head source includes the subsequent metric correction below. PyTorch was
used only in the existing independent reference environment; the model path
uses JAX and NumPy.

`metric-qualification-001.json` passed six tests in 6.30 seconds. Independent
NumPy populations agree with position-, opening-family-, opponent- and
move-phase-weighted totals, including fractional family mass. Padding containing
NaNs contributes neither error nor gradients. Four simulated CPU devices,
including two empty shards, agree with the unsplit global objective and every
replicated value-head gradient. Fractional mass must be divided by its actual
positive sum; clamping the denominator to one was caught and corrected before
integration.

The zero-predictor error and fitted-constant error are diagnostics. The latter
uses the evaluated population's own mean and is therefore an optimistic
in-population reference, not a predictor fitted on the training split.

The proposed temporal readout is 768→256→3 with Mish and 197,635 parameters,
reading the same normalized causal latent as the policy. Integration proceeds
in [the separate joint prototype](../strong19_joint/README.md). The joint loss
coefficient, optimizer and learning horizon still require a frozen registration,
followed by full-size TPU and inference qualification.
