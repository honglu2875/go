The 19×19 joint policy/value comparison remains a later stage. The live corpus
uses one fixed strong teacher and eight opponent strata, with raw teacher labels
on every ply. It is still too early to freeze the main training population.

`inventory-001.json` captured 177 terminal games and 69,420 positions, with no
move-capped games. It selected one terminal training game per available host and
opponent stratum for feature qualification, without decoding test targets.

`feature-loader-qualification-001.json` passed all 30 selected games and 12,021
positions through the pinned native KataGo V7 replay and the existing packed
loader. Boards, legality, inputs, actions and float32 policy/value targets agree
exactly. Two complete games also passed independent native replay under all eight
D4 symmetries. Actions remain int32, including pass=361; the old 9×9-only uint8
trajectory encoding must not be reused here. The 30.9 MB qualification arrays
are isolated in RAM and are not a registered training view.

The separate [joint model prototype](../strong19_joint/README.md) now has
qualified CNN and temporal value readouts, CPU joint-gradient/cache checks,
and full-size abstract parameter/decoding FLOP accounting. Full-size TPU memory,
the joint optimizer and objective, and the final data view are still unqualified
or unregistered. The larger-9×9 paired comparison must be interpreted first.
