The baseline is ready for the registered learning experiment. This is a
policy-only adaptation of KataGo's nested-bottleneck CNN, initialized from
scratch. It is not a pretrained KataGo checkpoint or the earlier homemade
49-block ResNet.

| Check | Result |
|---|---|
| Pure JAX versus official PyTorch | 204 checks passed; largest absolute error 5.37e-7 |
| Outputs checked | Main policy and same-target training BN helper |
| Derivatives checked | Every active parameter gradient; helper statistics remain differentiable |
| Microbatch change | Same outputs and gradients; primary inference is batch independent |
| Four virtual CPU devices | Global loss, gradient and optimizer checks passed |
| Interrupted CPU learner | All 297 parameter/Adam arrays bitwise equal to uninterrupted run; sampler states equal |
| Native V7 input reconstruction | 1,047,681 expert pre-action boards and legal masks exactly matched Rust replay |
| multi-host small-model run | Four accepted updates; independently reconstructed 19,251 positions |
| multi-host 232M run | Four accepted updates over all three buckets; independently reconstructed 76,954 positions |
| Parameter accounting | 232,431,872 trainable; 232,134,784 used by inference |
| Complete neural decode | 37,348,255,744 dense multiply-add FLOPs per 9x9 move, encoder included |

The independent JAX graph counter expands static loop trip counts and agrees
exactly with the analytic dense arithmetic count. The full-size warm profile
used a global batch of 128; host 0 dispatch/completion measurements ranged
from 16.14 to 16.65 ms. These measurements exclude CPU rules/feature work,
input transfer and prefill. No MFU claim is made. The raw XLA cost estimate
undercounts repeated scanned blocks and is retained as a separate field;
it must not be substituted for the complete decode count.

Qualification source snapshots and logs remain under `runs/qualification`.
The authoritative gate identities and frozen 1,024-update configuration are
in `baseline_registration.json` (SHA256
`8f5fc01507af9aed2ed1a525384705e84e865fe4321bd9fb06b664377827bafd`).
The active baseline is `pod-20260912T111225Z-a8f7c3c1`, source
`625f75ca64b04da8b65e36e6ad8c83f42ee5674eb67637faabcf60efaf256997`.
Later operator snapshots do not change that running experiment.

The input-target disagreement diagnostic uses only training positions. Exact
V7 inputs have 750,061 distinct keys across 836,486 positions; 100,644
positions belong to repeated-input groups. Mean target disagreement is
0.0230944 KL under uniform training-position weighting. Keeping the entire
move prefix changes this finite-set quantity to 0.0229879. This measures
one source of ambiguity at repeated inputs; it does not measure noise at
unique inputs, establish a population loss floor, or show that history is
unhelpful for generalization. Raw results and operator identity are in
`training_target_disagreement.json` (SHA256
`835c00ab1a7b2308f1a5d4c762b515c0aad26c464612b705249f3f0e155ec1cc`).

Validation contains 102,339 positions, but only 97 occur after move 256.
The registered phase metrics therefore support much stronger conclusions
about opening and midgame learnability than very long histories. The test
split stays closed while the sequence of encoder ideas is selected.
