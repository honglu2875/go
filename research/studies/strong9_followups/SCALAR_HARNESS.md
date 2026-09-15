The next scalar experiment can reuse the completed larger9 controls and the
qualified training program. [scalar_source.py](scalar_source.py) permits one
configuration change: a larger initial encoder residual scale, or a lower
final learning rate. The complete 4,096-update horizon, warmup, every other
setting and all frozen source files stay unchanged.

Selection requires the preceding experiment's completed outcome, a written
review, and an immutable two-seed registration. After the negative LR1.5
closure, the [completed review](scale1e-2-review-001.json) selected encoder
initialization 0.01 with the original LR. The
[immutable registration](scale1e-2-registration-001.json) was inspected before
launch; seed 1 is running under its own conditional continuation. No LR-floor
experiment is selected or queued. The completed LR operators remain unchanged.

The [CPU qualification](scalar-cpu-001.json) reproduces the parent snapshot,
checks both temporary scalar configurations against all 1,310 source files,
rejects 37 confounded or malformed fixtures, and checks six replication
decisions. Temporary configurations and decision records were discarded.
The paired curve arithmetic and all-rank evidence reader are imported from
the unchanged, previously qualified LR comparator.

An LR-floor probe must retain identical observed initialization hashes.
An encoder-scale probe must have consistent but changed initialization hashes
across ranks. Its source/configuration construction changes only the three
encoder scale tensors; CPU preparation separately checks exact initialization
of the other arrays. We do not claim a separately measured common-parameter
TPU hash. Both interventions require identical parameter schemas, actual
per-rank game/D4 draws, position exposures and evaluation populations.

For an encoder-scale probe, the existing full-shape TPU qualification is reused
only if its model differs solely in `encoder_layer_scale`, every numerical
source matches, and the selected value passed initialization preparation.
That field is consumed by initialization, not by the forward/backward functions.
Complete decoder/encoder arithmetic and the small forward/backward HLO remain
unchanged. The candidate's real learning run still has all finite-update,
checkpoint, evaluation and resource guards.

The launch operator retains exclusive accelerator ownership and RAM reserves.
The finalizer retains a verified peer checkpoint, runs the frozen learning
audit, then produces the paired contrast. The continuation monitors and plots
each run, and launches the paired second seed only when both endpoint KL
metrics meet the registered gain, neither last-three mean regresses and no
sustained overfit is present. It stops after that intervention.

Real registration, launch and continuation inspection passed for scale1e-2.
Its eventual learning result still requires actual finalization and audit.
A CPU fixture does not qualify a future result in advance.
