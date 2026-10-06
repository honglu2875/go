# Observations from the registered screen

The original 64-update registration completed and passed all audits. These
diagnostic notes supplement [the endpoint comparison](balance010-001-RESULTS.md).
The registered source, horizon and scientific settings were not changed.

At update 64, sparse policy KL is 1.086170 versus dense 1.133431 (4.17%
lower), while value MSE is 0.195272 versus 0.169567 (15.16% higher). Top-one
accuracy is 0.400864 versus 0.419475. Sparse fixed-train-probe policy KL is
1.077530 and value MSE 0.161432. The policy advantage is smaller than at
update 48; selecting that earlier validation would exaggerate the result.
Recorded learning time is 9775.58 seconds versus 4353.77 seconds, a 2.245x
penalty. The historical dense update-144 validation fits within that sparse
learning-time budget and has policy KL 0.813972 and value MSE 0.115248. This
is an earlier timing reference, not a contemporaneous equal-wall-time trial.
The current all-expert implementation does not meet the efficiency goal.

Final average router load CV-squared is 0.393840, dead-expert fraction is
0.035714 for that update, and no tokens were dropped. Zero traffic in an
update does not imply an expert was never trained. The midpoint optimizer
inspection below distinguishes these questions.

At update 16 the sparse model's validation policy KL was 1.775396 and signed
value MSE 1.091200, versus 1.627531 and 0.302845 for the dense transformer on the
same exposure prefix. The fixed training probe was similarly poor for value
(1.100847), so this point does not suggest ordinary train/validation overfit.
Mean value prediction was -0.620914 on validation against mean target 0.001084.
Training-step value predictions and errors oscillate during warmup. The next
validation points are needed before judging convergence.

At update 32 the value calibration has recovered: validation policy KL is
1.471876, value MSE 0.280419 and top-one accuracy 0.367168. The dense prefix is
1.422605, 0.303448 and 0.302139 respectively. The fixed sparse training probe
has policy KL 1.467312 and value MSE 0.217647. These mixed metrics do not show
an overall win; the endpoint and substantially greater wall time still matter.

At update 48, sparse policy KL is 1.172795 versus dense 1.342085 (12.61%
lower), while value MSE is 0.257546 versus 0.218418 (17.92% higher). Top-one
accuracy is 0.349288 versus 0.352224. Sparse fixed-train-probe policy KL is
1.163407 and value MSE is 0.165689. Both its validation errors decreased from
update 32; the value train/validation gap widened. This is a policy-learning
benefit at matched exposure, not an overall time-efficiency result.

The median pre-clipping global gradient norm in updates 1–16 was 15.83 for
sparse versus 7.96 for dense; median clip multipliers were 0.0643 and 0.1262.
In updates 33–48 those norms were 1.65 and 2.14. These observations accompany
the early instability but do not identify a causal optimizer fix.

Hard expert usage is uneven even though softmax entropy is near log(4). This
is possible with almost tied logits: tiny probability differences can select
the same two experts repeatedly. The checkpoint diagnostic reports each
layer's optimizer moments; moment energy is explicitly not a token-load count.

The durable update-32 checkpoint passed integrity and schema checks in
`expert-state-032-001.json`. Every FFN expert has a nonzero accumulated second
moment. In the most concentrated spatial-attention layer, one expert holds
91.96% of its layer's second-moment mass. Across layers, the mean largest share
is 75.95% for spatial attention, 54.95% for convolutional FFNs, and 50.41% for
temporal FFNs. These are optimizer-gradient history shares, not traffic shares;
they show unequal learning contributions but do not establish why they differ.

Two hyperparameter conventions need attention before the next learning arm:

- Router initialization uses weight standard deviation 0.01 / sqrt(width).
  AdamW uses the same absolute learning rate as other weights. At width 768
  the router standard deviation is about 0.000361; relative router updates
  may therefore be large. This is a hypothesis, not an identified cause.
- Balance and z penalties are averaged over 42 distinct router layers.
  Thus balance coefficient 0.01 here is equivalent to about 0.000238 per
  layer in a summed-loss convention. Numeric coefficients cannot be copied
  between those conventions without conversion. Our assignment fraction also
  counts both selected experts, unlike a first-choice-only balancing loss.

For context, [Switch Transformers](https://arxiv.org/abs/2101.03961), section
2.4, uses standard deviation sqrt(s / fan_in) and studies reducing variance
scale s from 1 to 0.1. That is a different scale and a broader initialization
change than our small router-only initialization. [ST-MoE](https://arxiv.org/abs/2202.08906)
states that it follows the Switch initialization scheme. Neither observation
establishes an optimal initialization or LR for this Go model. If instability
persists, isolate router update scale and balancing strength in separate,
same-seed arms before changing the architecture or claiming a MoE result.

Execution follow-ups are separately cloned and CPU checked:
[activation memory](../strong19_moe_remat/README.md) and
[routing permutations/tiles](../strong19_moe_permute/README.md).
The latter changes no logical active matrix FLOPs. Larger chunk sizes remain
unqualified because BF16 first-moment drift exceeded the proposed conservative
limit; unchanged-chunk activation rematerialization was exact on CPU.

An architecture follow-up keeps the expensive spatial encoder dense and puts
experts only in the temporal transformer:
[temporal-only proposal](../strong19_moe_temporal/README.md). Its abstract
budget is 317,001,492 total parameters and active cached-move matrix FLOP ratio
1.00000132 relative to dense. This adds capacity with very little extra logical
arithmetic. The completed full-size systems check measured only 0.30–0.38%
update-time overhead at the two sequence buckets, with about 0.96 GiB extra
compiled memory. Its learning quality remains unmeasured; see the
[round report](ROUND_20261004.md) for the complete evidence and next arm.
