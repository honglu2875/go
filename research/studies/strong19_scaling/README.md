The first scientific 19×19 joint policy/value pilot has started. It uses a fixed
view of 2,634 training games (1,091,214 positions) and 146 validation games
(60,284 positions), with one strong KataGo teacher and eight opponent strata.
The raw policy and signed value targets come from the same teacher for every
architecture. Generation continues separately; later games do not enter this
view, and test targets remain closed.

[PILOT_DESIGN.md](PILOT_DESIGN.md) describes the three arms: the source-derived
CNN optimizer, an identical CNN with AdamW, and the selected causal transformer
with AdamW. All use the predeclared 108-update game/D4 replay and regular full
validation. The transformer's encoder initialization 0.01 improved both larger
9×9 seeds; it did not beat the CNN at equal updates. The pilot asks whether that
choice transfers to joint learning on 19×19.

The packed corpus is verified on all configured hosts. Full-size source-CNN
execution and exact checkpoint recovery have passed. The transformer execution
check used eight independent encoder frames per chunk after a 16-frame HBM
failure; both history shapes, all four updates and the complete checkpoint
audit passed. The source-optimizer CNN is the first scientific arm; the CNN
AdamW control and transformer follow it under the fixed pilot registration.
This changes the execution working set, not the model or decoding FLOPs.

The files below preserve the earlier qualification history. They are small
execution fixtures, separate from the scientific cohort above.

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

The separate [joint model prototype](../strong19_joint/README.md) has
qualified CNN and temporal value readouts, CPU joint-gradient/cache checks,
and full-size abstract parameter/decoding FLOP accounting. See the newer pilot
design and execution evidence for the subsequent optimizer, memory and data
decisions. Trained KataGo matches remain a separate validation requirement.
