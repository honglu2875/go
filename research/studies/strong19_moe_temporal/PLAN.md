# Dense encoder with temporal experts

This arm tests the location of sparse capacity while retaining the dense
transformer's encoder. The original all-expert screen is frozen and continues
unchanged. Neither this proposal nor its qualification starts a learning run.

The encoder is 24 shared blocks at width 768, applied twice: 20 convolutional
blocks and four spatial-attention blocks. Every encoder parameter and initial
draw remains identical to the dense reference. Only the 18 temporal SwiGLU
FFNs become four independent experts with top-two routing. Each expert uses
hidden width 1024, versus 2048 in a dense FFN. Attention, one-token board
connector, readouts, auxiliary objective and signed-value loss stay fixed.

The resulting parameter count is expected to be 317,001,492, including
118,161,424 encoder parameters. This increases total capacity at effectively
the same active matrix work. The encoder dominates cached-move arithmetic,
so placing extra parameters in the temporal stack may incur substantially
less runtime cost than routing every spatial FFN. This is a hypothesis to
measure, not an established speedup or learning improvement.

Preserve the original per-layer router coefficients: the loss averages over
active router layers, so the 18-layer arm uses balance `0.01 * 18 / 42` and
z-loss `0.001 * 18 / 42`. Empty encoder-statistic rows contribute nothing.
Router initialization, top-two normalization, dropless grouped kernels,
qualified tile `(256, 512, 512)`, precision and AdamW are unchanged. There is
no permutation-VJP optimization in this recipe.

Qualification checks include exact dense-parent initialization/loss/gradients,
unchanged dense encoder draws, full-model gradients, padding, causality,
cached decoding, checkpoint continuation, and four-device router reductions.
Before learning, run two fresh full-size updates at each sequence bucket and
the same dense control. Record actual memory and wall time; refuse allocations
above 31 GiB per device. The data split and exact sampling remain those of the
registered dense reference. No test set or playing-strength claim is involved.

A learning registration must freeze its source, configuration, qualifications,
schedule, storage budget and stopping point separately. Do not launch this arm
concurrently with another pod learner.
