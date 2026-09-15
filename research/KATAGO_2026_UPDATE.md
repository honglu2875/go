# KataGo reference refresh — 2026-09-11

KataGo's official index now includes transformer models introduced in August
2026. The smallest listed transformer has about 10.5 million parameters; the
largest has about 70.4 million. The documented blocks combine learned 2D RoPE,
RMSNorm, SwiGLU, and nested bottlenecks. The author reports favorable strength
per approximate inference cost, with backend-dependent performance. Those
ratings are not a controlled training-efficiency comparison with our system.
[Pinned architecture documentation](https://github.com/lightvector/KataGo/blob/9bb7efc39b2b432133c964ece6f8fb969441fdfc/docs/NetworkArchitectures.md).

Our simple attention recipe is an ablation control. A transformer alone is not
a novel departure from current KataGo. Later architecture comparisons should
include a compact version of these modern components, a well-tuned CNN, and
our causal opponent-modeling mechanism, with separate inference-cost and
learning-efficiency measurements. The pinned engine already supports model
format 17; actual new-weight/backend compatibility still needs qualification.

For early learning, four small historical checkpoints were pinned from the
[official index](https://katagotraining.org/networks/). The first calibration is
recorded in [the ladder study](studies/katago_ladder/README.md). They supplement
the existing strong 9x9 anchor. Beating a historical checkpoint at unequal
search visits must not be described as surpassing current KataGo or its
training process. Source pages, retrieval receipt and hashes are retained in
`.gozero/external/research/katago-20260911/`.
