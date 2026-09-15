The fourth metadata inventory contains 2,500 completed 19×19 games and
1,037,305 positions, with no capped games. Test policy/value targets remain
closed. The frozen inventory is an observation; it does not register a training
population or horizon.

| Split | Games | Positions | Opening families | Largest family |
| --- | ---: | ---: | ---: | ---: |
| train | 2,254 | 934,046 | 2,074 | 9 |
| validation | 125 | 51,560 | 117 | 3 |
| test | 121 | 51,699 | 117 | 3 |

All eight opponent strata are represented in each split. Validation has only
10 teacher self-play games; use that stratum cautiously. Its maximum game
length is 613, versus 645 in training. Family concentration is substantially
lower than in the larger9 validation view.

At 128 uniformly sampled complete games, the expected training batch contains
53,042.54 live positions. The source trainer's nominal million-position
epoch would therefore be about 19 updates. Blindly retaining its 100-update
print cadence would produce **no print-norm snapshots within such epochs**;
the adaptive decay would keep using the initial norm ratios. Scientific
transfer needs an explicit norm-observation cadence as well as an explicit
position-batch reference. The source defaults are verified in the pinned
trainer; this is a mismatch introduced by our complete-game batch adaptation.

Aligning a nominal epoch to the source Lookahead period of six would give
18 updates, about 954,766 positions. That avoids discarding
an unsynchronized fast update at each short epoch flush. This is a candidate
clock convention, not a selected setting.

Copying 128 updates would yield about 6,789,445 position exposures,
or 7.27 passes over this training population. Packed
training plus validation rows need about 2.534 GB before file and episode
metadata. Reserve that in addition to retained learner/optimizer checkpoints.
See [the machine-readable sizing observation](cohort-sizing-004.json) and
[the learning design](LEARNING_DESIGN.md).
