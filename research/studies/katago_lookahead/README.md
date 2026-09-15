The Lookahead component passes 812 tensor comparisons across 116 events from
the actual pinned KataGo trainer. All compared values are exact in this CPU
fixture, within the unchanged prespecified 2e-6 relative / 2e-7 absolute bounds.
The four cases include disabled Lookahead and k=1/3/6 with alpha=0.25/0.3/0.5.
They use 23 predetermined fast-optimizer increments per case, two epochs and
uneven subepochs. No neural training or optimizer settings are selected here.

[The reference exporter](../../recipes/katago_lookahead/export_reference.py)
executes the original initialization, reset, averaging and epoch-flush AST
statements with actual Torch parameters after verifying the entire trainer
source hash. The source resets the synchronization counter at each subepoch
entry, but flushes slow weights to fast weights after the entire subepoch loop.

The first frozen implementation failed donated execution at that epoch flush:
returning the slow parameter tree directly aliased the next call's fast and
slow inputs. [The retained failure](cpu-result-001.json) and
[alias observation](alias-observation-001.json) identify this boundary. The
correction performs the source's copy into independent fast storage outside
the donated executable; numerical equations and tolerances are unchanged.

[The corrected qualification](cpu-result-002.json) passed in 9.32 seconds.
Uninterrupted and fresh-process continued runs have identical complete
checkpoint manifests for all four cases, covering 32 fast/slow/counter arrays.
The k=6 restart occurs with counter 5, immediately before synchronization.
The runner also rejects five malformed configurations and four corrupted
checkpoint states. The qualified snapshot is
`7f996718dcc8195562f0adfafd242b1a67c7a495c553fbdb7bf4f6f4b882474d`.

Both frozen revisions, the pinned Torch trainer, reference arrays, retained
failure and complete small CPU checkpoints are sealed in
[source-bundle-001.json](source-bundle-001.json). The private archive contains
167 members / 1,863,680 bytes; every member passed full byte readback.

Integration with the complete joint Muon learner, source norm/LR cadence and
full-size accelerator execution remains open. The running 9×9 comparison
and its LR-only follow-up are unaffected.
