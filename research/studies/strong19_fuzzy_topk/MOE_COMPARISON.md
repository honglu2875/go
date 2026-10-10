G8 improves policy learning over both completed conventional temporal MoE arms,
including at the same peak learning rate. All rows use the same fixed corpus,
game/D4 replay, batch of 64 complete games, 256 updates and 7,001,181 position
exposures. Validation and training-probe population identities match.

| Architecture | Parameters | Peak LR | Policy KL (lower) | Value MSE (lower) | Learning hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense SwiGLU | 232,011,540 | 0.001 | 0.728001 | 0.121956 | 2.437 |
| Temporal MoE, standard balance | 317,001,492 | 0.001 | 0.743047 | 0.124549 | 2.444 |
| Temporal MoE, threefold weaker balance | 317,001,492 | 0.001 | 0.733949 | 0.120592 | 2.444 |
| G8 | 298,257,732 | 0.001 | 0.706977 | 0.116157 | 2.560 |
| G8, selected LR | 298,257,732 | 0.0015 | 0.686237 | 0.123311 | 2.559 |

Against the weaker-balance MoE, G8 at the same LR improves policy KL by 3.67%
and value MSE by 3.68%. The selected G8/LR 0.0015 improves policy KL by 6.50%,
with 2.25% higher value MSE. G8 has 5.91% fewer total parameters and takes
about 4.7% more recorded learning time. Last-three validation policy KL means
are 0.759474 for weaker-balance MoE, 0.744699 for G8/LR 0.001 and 0.720701 for
G8/LR 0.0015.

Conventional MoE uses four independent SwiGLU experts and routes each temporal
token to two experts of hidden width 1,024, with balance and router z losses.
G8 selects individual features within fixed groups and uses no independent
router or balancing objective. Both preserve the dense spatial encoder.
Logical active decoding matrix FLOPs are approximately matched; dispatch,
padding, extra issued matrix work and measured runtime differ.

These are one-seed schedule-prefix comparisons. G8 received additional LR
tuning; the MoE arms shown here used only LR 0.001. A matched MoE LR sweep and
longer paired-seed comparisons would strengthen the inference. The weaker-balance
MoE's late top-choice-agreement decline remains unexplained. No new playing-
strength result follows from this table.

Sources: [MoE protocol and results](../strong19_moe_batch64/RESULTS.md),
[G8 sweep results](RESULTS.md), and their fixed-population learning curves.
