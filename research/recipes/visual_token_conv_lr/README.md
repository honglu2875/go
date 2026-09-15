This clone sweeps only the peak learning rate of the qualified convolutional-encoder transformer. Its model and optimizer implementation are byte-identical to the 1e-4 parent.

The registered candidates are 3e-4, 1e-3 and 3e-5, in that order. The completed 1e-4 run is the reference. Each candidate starts from the same seed and uses all 1,024 updates, the same games and D4 draws, 64 warmup updates, the original cosine horizon and an ending rate of 30% of the peak. Every other optimizer setting remains fixed. Validation occurs at the same 128-update intervals; the test split remains closed.

Choose among successfully completed endpoints by validation KL, and retain the full curves, time and stability diagnostics. Divergence is a result, not a reason to change that arm's optimizer mid-run. This one-seed sweep selects a candidate within the tested range and schedule; it does not establish a globally optimal rate or uncertainty across training seeds.
