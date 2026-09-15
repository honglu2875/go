The joint policy/value training harness now passes CPU training, validation and
fresh-process continuation checks on complete real 19×19 games for both model
families. Checkpoint transport also passed on all configured hosts. This is
preparation for the later 19×19 study; it neither changes the running larger-9×9
comparison nor registers a new learning experiment.

The owned recipe is `../../recipes/strong19_train/`. Its model, bounded-gradient
paths and optimizer are copied from the qualified `../strong19_joint/` source.
`provenance-001.json` records those copies. `train_joint.py` derives from the
frozen larger-9×9 trainer; `trainer-clone-001.json` records that initial cloning
step, before the final integration changes. The qualification snapshots below
bind the final tested bytes, shared library and complete configuration.

| Evidence | Result and scope |
| --- | --- |
| `evaluation-cpu-001.json` | Five tests, 69.15 seconds: bounded main policy/value output equivalence on real 19×19 prefixes, chunk sizes 1/3/32, causal and cross-game isolation, empty lanes, family weighting and global reductions over four simulated CPU devices. |
| `evaluation-schema-001.json` | Six abstract traces at full width 768, local batch 8 and 128/512/1,536 positions. Bounded evaluation has the same logical matrix work as ordinary main inference. |
| `host-metrics-cpu-001.json` | Host-side scalar aggregation matches the qualified metric equations within 2.46e-7, including fractional family mass and empty populations. It requires no accelerator dispatch. |
| `harness-result-001.json` | Both small complete models pass four updates versus a fresh-process 2+2 continuation. Every parameter, optimizer array, sampling RNG and diagnostic history matches exactly. |
| `harness-real19-result-001.json` | Both small complete models pass the same continuation check on real 19×19 train/validation games, including 512/1,536-position runtime buckets. |
| `resume-transfer-result-002.json` | Complete synthetic RAM checkpoint closure stages to all configured hosts; actual checkpoint readers recover exact arrays and every rank state. |
| `pod-regression-001.json` | Eight existing controller/rank supervision, failure, cancellation and source-isolation tests pass after recovery integration. |

The CPU harness uses the actual fixed larger-9×9 corpus and whole-game loader
to exercise sampling, length buckets, augmentation and recovery. It consumes
the same 1,751 position exposures in each architecture. The separate output
tests use real 19×19 prefixes. Neither fixture is an architecture learnability
result. AdamW, value weight 0.7 and the small model sizes are numerical test
settings, not selected 19×19 hyperparameters.

The continuation test ran for 390.42 seconds and checked 214 checkpoint arrays
for CNN and 160 for transformer. It independently replayed all game, bucket and
D4 draws. Validation and training-probe histories at updates 0/2/4 and all
eight policy/value overfit observations survive restart. Full and resumed
checkpoint manifests are identical for each architecture.

The frozen CPU snapshots are:

- CNN: `c3ae429eb5efc4581520b4f19a2c20cc914e819bc894f27919f7994b2ff49e06`.
- Transformer: `5970693ea03f4e278c4f88999acaf8abff4d42ce134a87f50063a6086dff501e`.

The subsequent real-19×19 fixture contains 19 complete train/validation games
and 7,992 positions, selected deterministically from the ongoing collection
without reading test targets. `fixture-plan-001.json`
states the selection; its preparation receipt verifies native boards, legal
masks, V7 inputs and packed target read-back. It is explicitly an execution
fixture, not the full collection or a learnability population. The trainer
rejects a fixture when the configured purpose is learning.

The fixture occupies 20,557,793 bytes including its manifest and is verified
on all configured hosts (`fixture-staging-001/receipt.json`). The real-board CPU
continuation check took 1,186.11 seconds: both models consumed exactly 7,907
position exposures; all 214 CNN and 160 transformer checkpoint arrays, optimizer
step, sampling RNGs, counters, metrics and diagnostic histories match after
four uninterrupted versus fresh-process 2+2 updates. Both complete-game length
buckets were exercised without truncation. Its snapshots are:

- CNN: `f906ff4dced2bdc19b80a0dc55a36deb495692fffc37b14c051ce2d7c2265b74`.
- Transformer: `4c62fa3a71f1b0b5a64578ec8828e38142e36012adce4f45490c95007043b480`.

RAM checkpoint recovery now resolves the actual owner arrays and gathers all
four rank states before distributed initialization. It verifies source, data,
optimizer/schema identities, rank mapping and every payload hash, then stages
the complete closure with explicit CPU affinity and memory floors. Host 0 uses
the actual RAM checkpoint argument; other ranks use their logical metadata
paths and the staged owner payload. Shared code is `gozero.joint_resume`, wired
through `ops/stage_joint_checkpoint.py`, `pod_run.py` and `run_host.py`.

The first network staging test failed with rsync exit 23 while attempting to
preserve implied system-directory timestamps. Its result and logs remain as
case 001. A new frozen case disables implied-directory attributes and directory
timestamp updates while preserving file permissions. Case 002 passed in 7.99
seconds: all 22 closure files were verified, and each of the four hosts read all
four rank roles exactly. This tests transport with small synthetic arrays; it
does not qualify full-size TPU memory or distributed numerical continuation.

The TPU source pins in `real19-snapshots-001.json` precede the staging correction
and must not be launched. `real19-snapshots-002.json` binds the qualified
correction with the same numerical recipe: CNN snapshot `3d33b7b8e3d255de5acad1e63c81513f8b89421bc654d3a0e1989e4cdc62b194`
and transformer snapshot `62f1b2506cc1b8f22bdc7824a4bc5c2b549a0c3a2bf644f9bb049ce0e86e94a3`.
No 19×19 TPU job has been
launched; the larger-9×9 queue retains ownership.

Main validation chunks the board encoder and retains only the features needed
for the ordinary policy/value readout. The transformer still uses the complete
causal board/action sequence. Validation does not evaluate the training-only
auxiliary branch. At 1,536 positions, the largest individual abstract array
falls from 13.63 GB to 0.39 GB for CNN and from 76.87 GB to 1.14 GB for transformer.
These are graph shapes, not measured peak memory or speed. The bounded CNN
training path has a separate, previously documented extra trunk pass; this
evaluation result does not remove that training cost.

Before learning or strength claims, the remaining gates are full-size TPU
compilation, measured memory/latency, four-process continuation, frozen 19×19
data and objective/optimizer registration, trained-checkpoint serving, and
actual games against the pinned KataGo reference. The current larger-9×9 queue
retains exclusive ownership of the TPU until its registered sequence closes.

`source-bundle-001.json` now seals 161 source/evidence members in
`source-001.tar` (1,546,240 bytes), with every archive member read back and
verified. It includes the qualified trainer/library, parent numerical and
inference dependencies, locks, and the CPU snapshot manifests/configurations.
Datasets, checkpoint arrays, runtime wheels and compiler objects remain in
their separately identified stores.

`source-bundle-002.json` seals the subsequent real-board execution work in
`source-002.tar`: 214 members / 6,328,320 bytes, with complete archive read-back.
It includes the current qualified recipe/library/recovery sources, selected
CPU and multi-host evidence, corrected TPU preparation identities and the
preceding archive verbatim. The local bundle contains private operational
paths and must pass the publication sanitization workflow before public use.
