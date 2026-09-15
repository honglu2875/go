The larger 9×9 encoder-initialization intervention is complete. Scale 0.01
improved both seeds, with mean reductions of 1.94% in position-weighted KL and
1.15% in equal-family KL. The CNN remains ahead at equal updates.

Full-size joint 19×19 source-CNN execution and exact restart recovery passed.
The selected Transformer also passed actual updates on both history shapes,
complete checkpoint/optimizer/sampler audit and verified checkpoint replication.
The initial memory failures and smaller successful execution chunks are recorded.
These are execution results, separate from learning and playing strength.

The fixed-data 19×19 learning pilot has started with the source-optimizer CNN.
Its AdamW CNN control and selected transformer follow sequentially. Each uses
108 updates, the same 5,787,025 position exposures, full fixed validation and a
fixed training probe. No further horizon or seed is launched by this controller.
See research/studies/strong19_scaling/PILOT_DESIGN.md.

Background mixed-strength collection continues outside this fixed cohort.
Actual trained KataGo matches and long-horizon training remain unfinished.
Private deployment/retention receipts and live checkpoints are excluded from
this portable source backup; adapt local deployment bindings before execution.
