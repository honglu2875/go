Follow-up: [the value-head diagnosis](../strong19_value_debug/README.md) confirms third-outcome saturation in the retained transformer. The original results below remain unchanged. The audited nine-update objective repair lowered held-out value MSE by 32.27% and policy KL by 1.67%; its full continuation is running.

All three fixed 19×19 pilot arms completed 108 updates and passed the complete
all-rank, draw/population, checkpoint and retention audits. The sequence ended
at 03:34 UTC on 2026-09-16, after 6.14 hours. Each arm saw the same 5,787,025
training-position exposures; validation contains the same 60,284 positions.

| Arm | Position policy KL | Equal-family policy KL | Position value MSE | Equal-family value MSE | Learning minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN, source Muon/AuxAdam/Lookahead | 0.80164 | 0.82307 | 0.15782 | 0.16885 | 92.51 |
| CNN, AdamW | 0.98984 | 1.01418 | 0.15960 | 0.16739 | 92.24 |
| Transformer, AdamW | 0.96571 | 0.98898 | 0.54223 | 0.54356 | 136.26 |

Lower is better. Learning minutes exclude compilation, evaluation, sampling and
checkpoint work. The source optimizer improves CNN policy KL by 19.0% relative
to its AdamW control, with similar value error. The transformer improves policy
KL by 2.4% relative to AdamW CNN, but has 3.4 times its position value error and
takes 47.7% more learning time. It is not the strongest joint model in this pilot.

No arm triggered the registered sustained-overfit diagnostic. The transformer's
value loss stayed almost exactly at the zero-predictor baseline through updates
9–54 on both validation and the fixed training probe. It later improved, but
its final mean prediction is +0.408 against a target mean near +0.001. Its final
training-probe MSE is 0.547, versus validation 0.542. This points to poor value
learning rather than an ordinary widening generalization gap.

The next diagnostic should inspect value-logit probabilities and gradients.
The shared three-logit softmax is trained here only through its signed
win-minus-loss expectation using MSE. Saturation toward a neutral prediction is
one plausible optimization failure; the current aggregate logs do not establish
which logits saturated. Do not call this a proven encoder-capacity limitation.
A bounded head/objective ablation is preferable to scaling training unchanged.

These are single-seed supervised learning results. Trained KataGo matches have
not run. The registered queue has stopped for review, while background corpus
collection continues. Checkpoints retain two verified peer RAM copies and
restore locators. See PILOT_RESULTS_001.json and pilot-comparison-002.png.
