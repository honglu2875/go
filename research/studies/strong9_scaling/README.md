This stage compares the selected transformer and the KataGo CNN on the finalized
strong-teacher 9×9 corpus. Architecture selection remains in the preceding
readout study; no larger-data learning result is available yet.

The source is `quintic/go9x9`, revision
`4f579b0a21c456f3bb568174d384056351124cd5`, release ID
`6a26ffa743f3c7be95d783102f6b3596e0ce69893c5a8e28dcca154dc54f05f4`.
The original NPZ files and the release mirror remain in RAM. Derived data also
belongs in RAM, with explicit capacity checks and immutable provenance.

Data preparation must use the fixed teacher's raw policy and player-to-move raw
value on every ply. Played actions are historical inputs, not policy targets.
Search labels are not substituted on teacher turns. Features are reconstructed
causally by the pinned KataGo V7 adapter and checked against every stored board
and legal mask. Lossless bit packing can store its binary spatial features;
global features and teacher targets retain their original float32 precision.

Before registering the training view, inspect length and opponent distributions,
opening-family concentration, and complete-trajectory duplicates. A structural
split audit may read held-out actions and metadata, but held-out test policy,
value and outcome targets are not part of selection or training. The regular
validation set is separate from a deterministic training-set probe. Record
opponent-stratified metrics as well as the overall position-weighted metrics.

Register the selected data view, seeds, horizon, validation schedule, and decision
rules before launching this comparison. Its numerical model code will be cloned
and frozen separately from the original small-data research recipes.

Preparation completed: 199,085 train/validation games, 18,789,200 positions,
12,179,056,995 bytes of lossless arrays. The manifest is
`prepared-manifest-001.json` (SHA256
`37244b2e743b3b7f80e0b14eccddfc2942aa4d81120c31aab10c4b3b4f733fc2`).
`staging-001/receipt.json` confirms every file on all configured hosts. The RAM path is
`/dev/shm/gozero-datasets/strong9-v7-0f78b62d`. Fixed-target/full-loader checks passed.
No duplicate or opening-family cap was applied; report equal-family and opponent
stratified validation alongside the original position-weighted result.

The prospective comparison is registered in `comparison-protocol-001.json`:
4,096 updates per arm, two paired seeds, full validation every 256 updates and a
fixed 4,096-game training probe. Position-weighted and equal-opening-family KL
are co-primary metrics. Both arms use the same whole-game bucket and D4 draws.
The game-count-weighted length sampler is stated explicitly; it is not exactly
uniform sampling of all corpus positions. Sustained overfit is diagnostic during
this fixed horizon and requires a separately registered paired follow-up.

Both possible transformer selections passed the larger-recipe CPU arithmetic
checks. The preceding small-data refinement decision determines which is used;
no larger-data result can affect that choice. CNN source/config snapshots are frozen. The full-shape qualification
`pod-20260915T034236Z-4377451c` completed four updates on all hosts; its checkpoint
replication and independent qualification audit precede the full learning run.

The qualification passed its independent audit. CNN seed 1 is running as
`pod-20260915T035557Z-c34403f3`, using the registered 4,096-update horizon.
`seed1-progress.png` and its CSV show the recorded validation and training-probe
curves; `monitor.py` records runtime estimates and overfit diagnostics without
changing the run. The remaining registered order is transformer qualification,
transformer seed 1, the paired contrast, CNN seed 2, and transformer seed 2.
