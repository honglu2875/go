The four-attention transformer and its auxiliary candidate have **231,181,121 parameters**, width **768**, and **one historical soft token per board**. Full cached decoding costs **37.428121728 GFLOPs per move** at board 9×9, global batch 128 and 128 past moves, including the encoder and pending action. Master parameters are FP32; the main computation uses BF16 with FP32 policy readout. There is no batch normalization.

| Stage | Exact structure |
|---|---|
| Input | 9×9×22 spatial planes and 19 global features. |
| Stem | Conv weight [3,3,22,768], bias [768], stride 1, SAME padding; affine LayerNorm and GELU. |
| Encoder layout | 24 unique blocks, executed twice with shared weights. Positions 6, 12, 18, 24 are attention; the other 20 are convolutional blocks. Resolution remains 9×9 throughout. |
| Each convolutional block | Depthwise 3×3 weight [3,3,1,768], groups 768, bias [768], stride 1/SAME; affine LayerNorm; linear [768,3072]+bias, GELU, linear [3072,768]+bias; residual LayerScale [768], initialized at 1e-6. |
| Each attention block | Bidirectional attention over 81 points within one board; affine LayerNorm; bias-free Q/K/V/O [768,768], 12 heads × 64, 2D RoPE with 32 dimensions each for row/column; residual LayerScale. Second affine LayerNorm, bias-free SwiGLU gate/up [768,1024], down [1024,768], second LayerScale. |
| Board connector | Pointwise linear [768,64]+bias, flatten 81×64 = 5184, linear [5184,768]+bias and affine LayerNorm; add global projection [19,768] and learned board-type vector [768]. |
| Temporal sequence | Board token, played-action token, board token, played-action token, …; causal. No future action is visible to a board prediction. |
| Temporal transformer | 18 layers; RMSNorm; Q [768,768], combined K/V [768,512],12 query heads and 4 KV heads, head dimension 64; output [768,768]; RMSNorm; SwiGLU gate/up [768,2048], down [2048,768]. RoPE base 10000; cached capacity 512 board positions. |
| Expert policy | RMSNorm; tied action embedding/readout [82,768] and separate action-type vector [768]. 82 actions include pass. |
| Spatial corrections | Local linear [768,1]+scalar bias; plus contextual dot product between Q [768,64] from temporal state and K [768,64] from each current spatial feature, divided by√64. These corrections affect board points; pass keeps its original logit. |

The connector does not turn the board into a discrete vocabulary item. It produces a learned continuous vector, and the policy readout also retains access to current uncompressed spatial features. The encoder attention uses 81 temporary spatial vectors; this does not increase the number of historical tokens retained by the temporal transformer.

For auxiliary training, the first encoder pass supplies a draft token. Training packs [draft board, full board, action] with an explicit mask, sharing all weights and the expert target. The loss is 0.75 full CE + 0.25 draft CE. Both draft and full predictions use full-pass historical context; no draft key enters history. Full inference retains the architecture above. The draft evaluates one encoder pass and all 18 temporal layers.

Parameter counts include each unique weight once, even though the encoder executes twice:

| Component | Parameters |
|---|---:|
| encoder and connector | 117,743,680 |
| temporal blocks | 113,273,856 |
| readouts and action type | 163,585 |

architecture_001.json (external or omitted experiment artifact) records every parameter path, dtype, stacked tensor shape and the pinned training report. Stacked leading dimensions 20/4/18 denote unique convolution, spatial-attention and temporal layers; repeated encoder passes add computation without new parameters. The pass-sharing and intermediate supervision require separate ablations from increasing unique depth.
