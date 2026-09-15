The functional JAX standard Muon plus auxiliary Adam port passed its CPU
reference qualification. It is preparation for the later joint 19×19 learner;
the registered 9×9 runs continue to use their original AdamW implementation.

The reference is the actual scalar implementation in pinned KataGo revision
`92ee95c0a4b25fec214da00951ab69e97e207729`, with its licenses retained. This
establishes the optimizer equations and parameter groups, not the unreported
learning-rate schedule of a published checkpoint.

The port uses Nesterov momentum 0.95, five BF16 quintic Newton–Schulz steps,
match-RMS scaling, and auxiliary Adam with betas (0.95, 0.995), epsilon 1e-6.
Rates and decays are explicit inputs. Convolution weights convert from HWIO to
independent output-by-input/spatial matrices; stacked layers remain separate.
Normalization vectors use auxiliary Adam even when their stacked rank is two.

The first full qualification failed the predeclared 3% relative-RMS bound.
Two arithmetic details were investigated rather than changing that bound:

- Torch multiplies BF16 tensors by Python coefficients using FP32 intermediate
  arithmetic before rounding. Implicit JAX scalar promotion pre-rounded those
  coefficients. Explicit FP32 multiplication matches the reference probes.
- JIT compilation could bypass intermediate BF16 rounding before subsequent
  polynomial matrix products. The uncompiled stages matched the reference;
  explicit optimization barriers restored those rounding boundaries. A separate
  isolation diagnostic showed that changing momentum or division equations was
  unnecessary.

The failed run, earlier sources and intermediate diagnostics remain retained.
The corrected frozen snapshot is
`890e0a320861bd0acda457287ac25c3c1b1cc8cacb82083d6b728d3ad32b51f3`.
Its [qualification result](cpu-result-002/result.json) records:

- 16 Muon comparisons over four updates: the BF16 directions matched exactly.
- 20 auxiliary Adam comparisons: maximum update error 8.35e-7; maximum master
  parameter error across all leaves was 5.97e-8.
- All 111 ported policy/value CNN parameter objects matched their official
  semantic groups. The reference export also retains the official model's
  complete 143-object registration, including outputs absent from this port.
- Zero-gradient behavior, atomic nonfinite rejection and malformed-checkpoint
  rejection passed. A fresh process resumed two updates and reproduced all
  24 checkpoint arrays and the exact manifest of the uninterrupted four updates.

These are small CPU numerical checks with synthetic gradients. Neural-loss
integration, source-informed group schedules/decay, learner recovery and
full-size TPU optimizer overhead remain to be qualified. In particular,
rounding barriers have not yet been timed on TPU.
