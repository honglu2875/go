The compact path preserves the selected model's parameter tree, cached decoder,
and 0.75 full-pass / 0.25 first-pass policy objective. It changes training
execution: independent boards are encoded in bounded chunks, and each retained
768-channel grid is projected immediately into rank-64 readout keys and local
logits. The ongoing frozen larger-9×9 comparison does not use this path.

`cpu-tests-003.json` records six passing CPU tests, including nonzero heads,
every-parameter gradients, partial chunks, unequal live counts, causal isolation,
and explicit exclusion of the current full-pass token from its first-pass output.
Both inner-rematerialization settings pass. These are small-model checks.

`19-shapes-001.json` and `19-shapes-002.json` trace the actual full-size model and
differentiated training graphs at 128, 512, 1,024, and 1,536 positions. At eight
sequences per device and 1,536 positions:

| Quantity | Reference | Compact, inner rematerialization retained | Compact, inner rematerialization disabled |
|---|---:|---:|---:|
| Retained encoder/readout outputs | 13,702,791,168 B | 2,382,200,832 B | 2,382,200,832 B |
| Largest individual abstract differentiated array | 136,272,936,960 B | 2,038,431,744 B | 2,839,019,520 B |
| Differentiated matrix work / reference | 1 | 1.248199 | 1.000216 |

All columns use width 768 and chunked paths use 32 frames per chunk. The first
compact version recomputed both the whole chunk and each internal encoder block.
Keeping only the outer recomputation boundary recovers almost all that extra
work, with larger temporary arrays inside each bounded chunk. The forward matrix
work is exactly unchanged for these divisible shapes. Padding work is explicit
for other chunk/frame combinations.

These byte counts describe abstract array shapes and retained outputs, **not
compiled peak HBM**. Multiple arrays may be live together, while XLA may fuse or
reuse others. Full-shape TPU numerical, peak-memory and latency checks are still
required; chunk sizes 8, 16 and 32 are prospective runtime choices. No MFU or
training-throughput improvement has been measured.

For policy-only 19×19 decoding, a 16-channel connector gives 231,813,905
parameters. Complete traced decoding, including the encoder, matches the CNN
within 1% at 128 past moves and batch 128: parameters −0.266%, matrix FLOPs
+0.680%, unit-cost floating operations +0.755%. Value heads and the final 19×19
comparison still need separate accounting and registration.
