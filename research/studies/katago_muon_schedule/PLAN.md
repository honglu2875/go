Prepare the settings needed around standard Muon for the later 19×19 baseline.
This study does not change the registered 9×9 AdamW runs or select a new LR.

Port the pinned KataGo source's fixscaleonenorm branch for explicit global
position batch size, cumulative samples, effective LR scale, LR/decay factors,
lookahead alpha, warmup switch and running norm/baseline ratios. Execute the
original nested functions extracted as AST nodes for independent expected
values. Cover every 250,000-sample warmup boundary through two million samples,
multiple batch sizes, disabled/enabled lookahead scaling and norm ratios.
Require relative error at most 2e-14 or absolute error at most 1e-15 for scalar
Python arithmetic. Reject invalid or omitted settings.

The source scales its local summed loss by DDP world size before averaged
gradient reduction. The resulting gradient is the global position sum. Our
joint objective is position averaged, so any integration must explicitly
convert those units before RepVGG scaling and source clipping. For standard
Muon, the source clipping base is 11,000, scaled by sqrt(global batch/256),
the explicit clipping factor and inverse sqrt(effective LR scale).

These scalar equations are only one part of the final baseline. The caller
must preserve/declare norm observations and running averages, update cadence,
lookahead slow parameters/counter, epoch flushing and complete checkpoint
continuation. The current published checkpoint's numerical LR/batch/overrides
remain unverified; source defaults do not establish historical settings.
