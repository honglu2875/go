# Prospective temporal dictionary sweep

## Fixed comparison

Use the existing packed 19x19 corpus and exact train/validation split: 2,906
training games (1,233,366 positions), 152 validation games (64,371 positions).
Reuse the audited dense batch-64 reference at LR 0.001. Requalify its first update
with the cloned recipe before borrowing historical quality results. Do not read
the test split. Data generation is out of scope.

Each fresh run trains 256 updates, 64 complete games per update, on precisely the
same canonical game/D4 stream (7,001,181 position exposures). Use AdamW with
betas 0.9/0.95, epsilon 1e-8, weight decay 0.01, gradient clipping 1.0. Use the
same 1,024-update cosine schedule and 80-update warmup; each LR variant sets
its terminal LR to 0.3 times peak. These runs compare the same schedule prefix,
not fully annealed convergence. Evaluate the complete validation population and
fixed 128-game training probe at 0 and every 32 updates.

## Sequential stages

1. Run G1 (dense ReLU control), G2, G4 and G8 at peak LR 0.001, in that order.
   Keep width 768, 18 temporal layers and the complete encoder/readout fixed.
   H/K choices are 3072/3072, 4096/2048, 4912/1228, and 5464/683 respectively.
   Report parameters, active/issued matrix FLOPs, illustrative 128-tile rounding,
   actual compiled memory, training time and cached decoding latency separately.
2. Select among G2/G4/G8 using the mean validation policy KL at updates
   192/224/256, breaking ties by endpoint KL then label. Prefer candidates
   without sustained overfit, with endpoint/tail value MSE no more than 5% above
   dense, and endpoint/tail top-choice agreement no more than 3 percentage
   points below dense. If none passes those health gates, choose the lowest-tail
   KL candidate for diagnosis and explicitly report the failed gates.
3. For that grouping, run fresh peak LR 0.0006 and 0.0015. Initial parameters,
   draws and loss coefficients remain fixed. If the selected rate is an edge,
   optionally test 0.0004 or 0.00225, respectively, if one full run plus audit
   and closure reserve fits the remaining 24-hour budget. No other adaptive
   changes to width, objectives, initialization, batch or data are permitted.

At most seven learning runs are launched. Expected duration is roughly 18–24
hours including qualification, depending on measured compile/evaluation costs.
All candidate configurations, including conditional rates, are frozen before
learning. Partial endpoints cannot compete with complete endpoints. A deadline
admission failure is recorded as skipped; partial runs are not reported as wins.

## Qualification and interpretation

CPU checks cover literal-kernel forward and gradients, deterministic ties,
nonpositive features, live-token statistics, unchanged non-FFN initialization,
bounded/reference gradients, causal full/cache parity and optimizer round trips.
The full pod then runs two fresh-state updates at both 512/768 buckets for each
new grouping, plus dense 512. Require finite accepted updates and compiled peak
memory at most 31 GiB per device. Measure cached decoding at global batch 128,
histories 128/512, five repeats after warmup. Synthetic timing is not strength.

LR variants share the already-qualified shapes. Their first update must match
the base-rate system check except for the linearly scaled learning rate and
per-group update norms. This follows the unchanged AdamW first-update equation
and is separately tested on CPU. Every full run still receives all-rank numerical,
replay, population, optimizer-state and checkpoint audits.

A provisional quality win requires endpoint policy KL and family KL at least
0.5% below dense, no worse tail KLs, the health gates above, and learning time at
most 1.2 times dense. Historical timings are contextual, not a contemporary
randomized speed comparison. Keep active FLOP claims distinct from issued work;
all ReLU controls include biases whereas the historical SwiGLU FFN has none.
No MFU, playing-strength, convergence or population-significance claim follows
from this sweep. Do not automatically promote the default.

## Storage and recovery

Use the existing protected RAM corpus. Do not alter any dataset or prior study's
checkpoint. Keep at most the current selected fuzzy endpoint plus the just-finished
new endpoint in owner RAM, with extra copies on two peers. After endpoint audit,
replication and small-evidence backup, delete only nonselected new trial
`arrays.npz` payloads, checking exact attempt, path, size and SHA-256 first.
Preserve source, snapshots, configs, metrics, manifests, optimizer/sampler metadata
and deletion receipts. This uses the user's explicit permission to discard
ablation checkpoints. Make one checksum-verified fsynced disk copy of the final
selected full state on a peer with enough disk capacity. RAM remains volatile.

Require at least 64 GiB free shared memory, 2 GiB disk and 24 GiB available RAM
during learning; admit runs only with additional endpoint and memory reserves.
Mirror source/registration and all small closure evidence to two disk peers.
Do not include private host inventories or credentials in public research notes.
