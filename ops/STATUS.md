The larger fixed-data 9x9 comparison is complete on both paired seeds.
The transformer did not meet the registered improvement criterion: mean final
policy KL was 3.82% higher per position and 4.36% higher with equal opening-family
weights. Its measured learning time was about 20% lower. Both validation and
fixed training probes continued improving without sustained overfit. These
results compare supervised learning under the registered settings; they do not
establish playing strength or RL efficiency.

The next intervention changes only the transformer learning rate: peak 1.5e-3,
end 4.5e-4, with the same architecture, initialization, data draws, 4,096-update
horizon and 64-update warmup. Its first seed is running. The registered
continuation records validation/probe curves and permits a second seed only
after both endpoint KL measures improve by at least 0.5%, neither final-three
mean regresses, and no sustained overfit appears. A new evidence review is
required before another rate or encoder. See research/studies/strong9_followups/.

The separate Lookahead component matches the pinned KataGo source on 812 tensor
comparisons and preserves complete state across fresh-process continuation.
The retained first failure identified aliased fast/slow storage at an epoch
boundary; the correction matches the source copy semantics. This remains a
small CPU component qualification. Integration with the joint Muon learner,
source norm/schedule cadence and full-size accelerator qualification remains
open before the scientific 19x19 policy/value and playing-strength comparison.

This source backup excludes credentials, private deployment inventories,
operational logs, datasets, model weights and binary evidence. Machine and
resource examples are anonymized; see PUBLICATION.md.
