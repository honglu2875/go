The full-size source-optimizer execution fixture is prepared as snapshot
`49412bc1b88b99bd800d0081481836a8ceaa3a2133a1ff9b580110c95def9497`.
It uses the same 233,220,870-parameter joint CNN and real-19×19 fixture as the
previous AdamW preparation, with 128 complete games per global update and
history buckets of 512 and 1,536. The source-Muon/AuxAdam recipe is byte-identical
to the passed small neural recovery harness; shared data and recovery sources
match the qualified transfer implementation.

The [abstract check](full-size-trace-001.json) traces both complete distributed
updates, including the SUM-gradient collective, and preserves every model,
optimizer and scalar diagnostic shape. The persistent arrays are:

| Component | Bytes |
| --- | ---: |
| FP32 master parameters | 932,883,480 |
| First moments / Muon momentum | 932,883,480 |
| Auxiliary Adam second moments | 7,056,408 |
| Lookahead slow parameters and counter | 932,883,484 |
| Dynamic schedule vector and update counter | 56 |

Only the auxiliary Adam parameters need second moments. Complete optimizer
state is 1,872,823,428 bytes, in addition to the master parameters. At the
longest bucket, the input batch is 413,638,688 bytes per device. These counts
exclude activation lifetimes, compiler temporaries and runtime reservations;
they are not a peak-memory estimate.

The proposed real execution sequence is four uninterrupted updates, a fresh
two-update prefix, and a fresh continuation to update four. The boundary
retains distinct fast and slow weights immediately before Lookahead sync at
update three; the epoch flush occurs at update four. Require exact complete
checkpoint/rank state, independent sampling replay, validation/probe histories,
and applied schedule replay across all ranks. Record actual compilation cost,
memory analysis, update/evaluation timing and checkpoint transport.

The [execution plan](full-size-execution-plan-001.json) now freezes those three
stages, the auditor and the CPU recovery prerequisite. The new auditor passed
against the existing real CPU run: 11 uninterrupted updates versus a fresh
six-update prefix and five-update continuation. It checks canonical tensor and
payload identities, all rank state, independent game/D4 draws, source schedule
replay and retained policy/value diagnostics. That result verifies the auditor
on one process; the real distributed full-size result is still required.

No TPU job has launched from this preparation. The current 9×9 intervention
retains the accelerator allocation. The launch operator requires its complete
review, including its conditional second seed, before starting. Every stage
checks exclusive allocation and reserves its remaining checkpoints, peer
transport and producer growth above the RAM floor. Closed negative trials have
been moved to verified peer copies to provide that headroom; all unique states
remain available through private restore locators.

The 256-position schedule reference, synthetic initial sample
offset, short epoch and Lookahead period remain execution fixtures. They do
not select scientific 19×19 batch size, optimizer settings, training horizon
or historical KataGo equivalence. Transformer execution uses its separately
prepared joint model and will follow the completed architecture decision.
