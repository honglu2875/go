# Activation-only rematerialization follow-up

Prepared during the immutable first MoE learning screen. No scientific model,
objective, weights, RNG, data, or running snapshot is changed. This is an
execution experiment, not an additional architecture or learning result.

Abstract reverse-mode inspection of the current model identifies multiple
FP32 arrays of shape (4, 5, 5776, 1536), each about 710 MB, retained across
convolutional encoder blocks within a chunk. JAXpr sizes are not live HBM.
Recompute only GELU or SwiGLU nonlinear arithmetic during backward, retaining
its inputs and all expert matmul results required by their existing VJPs.
The option defaults off and is explicit in a separate cloned recipe/config.

First compare outputs, all gradients, optimizer updates and checkpoint
continuation on CPU, with original and recomputed paths and several chunk
sizes. Then count abstract scan residual extents. TPU compiled-memory and
real update timing must be measured before accepting any optimization. A
larger chunk may reduce repeated routing and improve matrix utilization, but
must remain below the 31 GiB guard and preserve complete-history mathematics.
Do not queue TPU work while the registered MoE learning screen holds the pod.

No TPU speedup, memory saving, learning improvement or promotion is claimed
from the abstract inspection or CPU qualification alone.
