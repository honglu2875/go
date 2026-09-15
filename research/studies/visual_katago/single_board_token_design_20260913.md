Proposed 2026-09-13 after the user requested one token for an entire board, a simple large encoder, and roughly 1:1 or 1:2 encoder-to-transformer parameters. This records a design and arithmetic, not a training registration or an implemented model.

The first candidate should use approximately 1:1 allocation. A residual CNN processes the board, a learned projection makes one width-768 observation embedding, and a standard causal transformer processes alternating observation and action embeddings. Predict the current action from the current observation's transformer output. All components receive gradients from the same supervised policy objective. There is no separate policy-readout token, additional loss, or return-conditioning input.

This representation has a direct precedent in [Decision Transformer, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf): its visual observations use a convolutional encoder and one state embedding, with actions predicted from state-token outputs. Its return-conditioned offline-RL objective differs from our fixed-teacher policy distillation.

The proposed encoder borrows residual channel mixing and LayerNorm from [ConvNeXt, FAIR/Berkeley 2022](https://arxiv.org/abs/2201.03545), using a 3x3 depthwise kernel for this Go adaptation. [CoAtNet, Google Research 2021](https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf) provides direct hybrid-model evidence for LayerNorm replacing BN and for studying convolution/transformer allocation. Reusing the encoder stack is our additional experimental choice, not a result established by either paper.

| Component | Proposed shape and operation |
|---|---|
| Input | Current 9x9x22 board features and 19 global features |
| Stem, once | 3x3 dense convolution, 22 to 768, stride 1, SAME; affine LayerNorm and GELU |
| Residual encoder block | 3x3 depthwise convolution over 768 channels, stride 1, SAME; affine LayerNorm; pointwise 768 to 3072; GELU; pointwise 3072 to 768; learned channel scaling and residual addition |
| Encoder refinement | Apply the same ordered block stack to its own output for two or three passes; parameters are shared between passes, without stopping gradients |
| Connector, once | Pointwise 768 to 64; flatten 9x9x64 to 5184; linear 5184 to 768; affine LayerNorm; add projected global features and a board-type embedding |
| Temporal transformer | Width 768, 12 query / 4 KV heads of dimension 64, SwiGLU width 2048, per-token RMSNorm and causal attention |
| Sequence | `[z_0, action_0, z_1, action_1, ...]`; output at z_t predicts action_t before that action is visible |
| Policy | One tied 768-to-82 action head including pass; existing legality masking and fixed-data targets |

Flattening gives different projection weights to different board locations; it does not make compression lossless. A bottleneck with 64 channels keeps this projection to about 4M parameters. The connector is charged to the encoder. This connector and policy vocabulary are specific to the 9x9 ablation; other board sizes need an explicit adapter contract even though the environment supports arbitrary sizes.

The analytical candidates are approximately matched to the current CNN's 232,431,872 parameters and 37.3483 billion dense multiply-add FLOPs per move at 128 prior moves:

| Allocation | Encoder parameters | Transformer parameters | Unique encoder blocks x passes | Transformer blocks | Total parameters | Dense GFLOPs/move |
|---|---:|---:|---:|---:|---:|---:|
| Approximately 1:1 | 117,780,544 | 113,273,856 | 24 x 2 | 18 | 231,118,912 | 37.2676 |
| Approximately 1:2 | 79,921,216 | 151,031,808 | 16 x 3 | 24 | 231,017,536 | 37.4281 |

There are also 64,512 parameters for the tied action embedding, action type and final policy norm. Both candidates execute 48 spatial residual blocks. Their dense arithmetic and parameter totals are within 1% of the reference under this ledger. Initializations, executable parameter-schema agreement, forward/backward checks, causal cached decoding, padding work, training rematerialization and TPU memory/latency remain to be qualified before registration.

The pass count matters: with just one encoder pass, these allocations require only about 18.9 or 12.9 GFLOPs per move. Those would be cheaper alternatives, not equal-FLOP comparisons. Changing from two to three shared passes also changes refinement and sharing, so the two proposed candidates compare complete allocations rather than isolating only a parameter ratio.

At fixed temporal depth and KV dimensions, alternating one board token and one action token reduces history/cache slots by 19 times relative to the current 36-patch plus readout plus action layout. This is a token-storage ratio, not a whole-model latency or MFU prediction. The large CNN is still evaluated for each new board. The key learning question is whether its single embedding retains enough location-specific information for accurate move prediction.

Run 1:1 first using the same fixed games, position exposures, one policy objective and full comparison horizon; compare a narrow learning-rate range before selecting it. Use the approximately 1:2 candidate only as the next allocation experiment. Persistent checkpoints must be written at each run boundary. The analytical results can be regenerated with [the standalone ledger](single_board_token_budget_20260913.py); machine-readable output (external or omitted experiment artifact) is retained alongside this draft.
