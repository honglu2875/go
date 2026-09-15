The [completed study](../../studies/visual_katago/shared_spatial_study_20260913.md)
found 3.7% and 3.9% endpoint KL reductions in two paired seeds at LR 1e-3.
Mean control/spatial KL is 0.530250/0.510096. The CNN remains the accuracy
reference. All four full runs and the full-shape qualification passed audit.

This recipe clones the completed `single_board_shared` study to test a direct
spatial path to the single policy distribution. The one-board-token connector,
24 shared encoder blocks executed twice, width 768, 18 temporal layers, and
fixed teacher data remain the control. There is no behavior or value objective.

For intersection i, add `dot(F[i], w) + b` to its existing transformer logit.
F is the final 9x9x768 encoder feature map already computed for tokenization.
One projection is shared over all intersections. The pass logit is unchanged.
The new weight [768,1] and scalar bias start at zero; all other initial arrays
and the initial policy match the control. The option `policy_spatial_bias=false`
removes both extra parameters and the correction for a paired control.

The new path adds 769 parameters and 124,416 multiply-add FLOPs per 9x9 move.
Trace the full encoder, connector, decoder, and readout before a TPU launch.
The total parameter and decoding budgets must stay within 1% of the frozen
CNN reference at global batch 128 and 128 past moves. Report other history
lengths, training/rematerialization arithmetic, and measured latency separately.

`test_shared.py` exercises trained nonzero spatial corrections in causal/full/
cached inference, inactive and stale cache guards, rematerialized gradients,
shared-pass gradients, zero-initialized policy equivalence, spatial gradient
flow, location correspondence, and pass behavior. `qualify_shared.py` traces
full-shape arithmetic on CPU. A separate four-update, all-bucket TPU run must
pass `audit_qualification.py` before 1024-update learning.

The learner also records an initial parameter digest excluding the added
projection and reindexing the original arrays. This lets the comparison audit
verify identical common initialization against historical controls. Those
metadata-only additions are marked in `train_policy.py`; sampling, losses,
optimizer, schedules, and checkpoint arithmetic are unchanged.

Experiments run sequentially. Select the control LR after the registered 1e-3
follow-up; then run the spatial correction at the selected LR. A relative
endpoint KL gain of at least 1%, with warm decode within 15% of its control,
triggers an independently initialized paired replication. Preserve every
outcome and report both seeds. The test split stays closed. Checkpoints are
persistent, and the study does not claim Go strength or RL improvement.

To iterate, clone the entire editable recipe from the workspace root:

```bash
.venv/bin/python -B -m gozero clone research/recipes/single_board_spatial my_spatial_followup
```

The `equal_lr_1e-03.json` and `equal_lr_1e-03_seed2.json` configurations contain
the two spatial arms. A paired control removes or disables
`model.policy_spatial_bias`; the strict contrast verifies the shared initial
arrays, optimizer settings, actual game/D4 draws and full validation population.
The inherited allocation/LR configurations are available for editing; their
presence is not evidence that each was trained in this round.

Freeze the edited clone and its selected configuration using
`python -m gozero snapshot`, then run that snapshot's `ops/pod_run.py` with
`--snapshot` and `--workspace-root`. The pod controller prepares the pinned
environment and executes the same frozen source across all configured hosts. Record
the concrete comparison, endpoint and budget before launching. A changed
numerical graph needs CPU tests, complete decode arithmetic from
`qualify_shared.py`, and a full-shape TPU qualification before a learning run.

`audit_learning.py` verifies a closed learning attempt.
`report_comparison.py` checks actual draws before producing a comparison;
`compare_spatial.py` verifies the minimal readout intervention;
`compare_replication.py` combines the two independent pairs without selecting
the better seed. `finish_replication.py` can run those analyses sequentially
after a registered candidate closes, with pinned inputs, bounded waits and
timeouts, separate logs and no automatic retry. It performs CPU analysis only.
Use a persistent controller lifetime for a detached analysis process; a child
of a short-lived sandbox namespace does not survive that namespace's exit.
