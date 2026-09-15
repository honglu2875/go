Prepare the optimizer needed by a later KataGo-informed joint 19×19 baseline,
while the registered larger-9×9 comparison retains the TPU. This is a source and
numerical preparation study, not a learning-rate selection or learning run.

The pinned source is KataGo revision 92ee95c0a4b25fec214da00951ab69e97e207729.
The published b40 metadata establishes Muon, and the author describes an early
SGD/later Muon history. Its numerical schedule, batch size and command-line
overrides remain unverified. Current source defaults cannot establish those
historical settings.

Port standard Muon with Nesterov momentum 0.95, five quintic Newton–Schulz
iterations in BF16, match-RMS scaling and auxiliary Adam (0.95, 0.995), epsilon
1e-6. Keep learning rate and weight decay explicit per semantic parameter group.
Compare against the actual pinned PyTorch implementation. The reference scalar
path avoids optional compiled/foreach batching; the upstream documentation
already distinguishes that arithmetic from bitwise equivalence of fused paths.
Retain the upstream licenses. The production numerical implementation is JAX;
PyTorch stays confined to exporting reference evidence.

Convolutions must orthogonalize output channels against input/spatial channels
after converting HWIO to OIHW. Leading stacked layers are independent matrices,
not an additional tensor axis to flatten into a larger coupled matrix. Normalized
vectors remain auxiliary Adam parameters even when stacked arrays have rank two.
Verify CNN grouping against the actual official model's complete registration,
including input, policy/value, helper and normalization parameters.

Test matrix orientations, rectangular and convolutional shapes, independent
stacked layers, zero gradients, multiple updates, all optimizer-state arrays,
nonfinite rejection and checkpoint-style round-trip/fresh continuation. Require
float32 momentum/Adam outputs within atol 2e-6 / rtol 2e-5. BF16 Muon updates must
have relative RMS error below 3%, cosine above 0.999 and norm ratio within 3% of
the scalar reference; record maximum absolute errors as well. These tolerances
account for BF16 reduction/compiled arithmetic, not optimizer hyperparameter
changes. Retain any failure and investigate before changing a bound.

Do not insert this optimizer into the frozen 9×9 comparison. Integration with
the joint learner, semantic LR/decay schedule, complete recovery and measured
full-size TPU overhead are later requirements before any Muon learning claim.
The next 9×9 intervention remains LR-only after both registered seeds close.
