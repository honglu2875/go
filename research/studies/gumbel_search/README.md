# Gumbel AlphaZero versus low-budget PUCT

The registered exploratory study met its promising criterion: the Gumbel
checkpoint won **62 of 64 games** against the fresh matched PUCT checkpoint.
All 32 opening pairs completed; KataGo independently agreed on all 10,615 played
boards and all 64 final scores. The paired score was 0.96875 with a conservative
95% bounded-score interval of [0.7287, 1], conditional on these opening pairs.
There is no Elo or independent-training-seed claim. Full source, model, native,
protocol and evaluation identities are in pilot_result.json (external or omitted experiment artifact)
and [pilot_spec.json](pilot_spec.json).

Both arms trained from scratch for 16,777,216 real moves at 16 completed search
simulations, with the same 307,461-parameter CNN, initialization, optimizer,
ownership/score supervision, replay settings, rules and multi-host topology.
The intervention is the complete search bundle: Gumbel root sequential halving,
completed-Q policy targets, internal action selection and played-action rule,
versus PUCT with FPU 0.2, root Dirichlet noise and early temperature sampling.
It does not isolate any single component of that bundle.

| Quantity | Fresh PUCT control | Gumbel candidate |
|---|---:|---:|
| Real self-play moves | 16,777,216 | 16,777,216 |
| Global learner updates | 32,668 | 32,695 |
| Completed self-play games | 57,230 | 123,941 |
| Truncated self-play games | 20,451 | 479 |
| Eligible terminal rows | 10,086,539 | 16,589,675 |
| Active neural evaluations | 281,316,914 | 280,537,570 |
| Absolute KataGo wins / losses / caps | 0 / 5 / 3 | 0 / 3 / 5 |

The candidate generated 64.47% more eligible rows and 97.66% fewer truncated
games at the same real-move budget. Those quantities describe the evolving
self-play distribution. They do not by themselves establish algorithmic sample
efficiency. The direct checkpoint panel supports a substantial advantage over
this particular under-tuned 16-simulation PUCT control.

Neither arm recorded a win in the small absolute KataGo anchor at 1 and 16
visits. Eight of the 16 scheduled absolute games reached the 324-ply cap and
have no assigned result. All 4,047 played boards and completed scores agreed
with KataGo. Consequently this study does not establish strong Go play or an
advantage over KataGo. The relative panel used its separately registered
1,296-ply cap and had no incomplete games.

Before training, native search matched the pinned author's Mctx implementation
on 320 finite-tree fixtures: actions and visits were exact; policy differences
were below 2.4e-7. Native actor recovery, real-Go GTP and the full CPU trainer were
qualified. The longer PUCT run reproduced all 336 saved arrays at turn 8,192 from
the earlier score-factor-zero control. Actor records matched after accounting
for exactly one newly serialized default field, `gumbel:null`.

Training attempt windows consumed 11.8963 chip-hours. The three evaluation panels
used 803.27 seconds against the registered 2,400-second allowance. These exclude
the larger reservation denominator for engineering and idle time. CPU work
overlapped some training on separate physical cores, so timing is diagnostic.

The [seed-28 replication](replication_28_spec.json) has now completed and met
its registered criterion. Gumbel won **51 of 64 direct games**, with all
5,161 boards and 64 final scores independently verified by KataGo. The paired
score was 0.796875, with conservative 95% interval [0.5568, 1] across the fresh
opening pairs. Both arms again trained for 16,777,216 real moves from scratch.
The immutable result (external or omitted experiment artifact) retains every identity,
counter, budget and limitation. Training used 11.9989 attempt chip-hours and
the three evaluation panels used 604.11 seconds.

The second-seed advantage is uneven by color: Gumbel won 32/32 as White and
19/32 as Black. Fourteen Black games ended within 25 plies, including all 13
Black losses. These early endings warrant diagnosis before longer training.
The strong absolute anchor still yielded zero wins: PUCT lost eight games;
Gumbel lost five and capped three. Replication supports the relative search
bundle result, but does not establish strength against KataGo or an improvement
in sample efficiency. Fair search tuning, absolute learning curves and stronger
training remain required. The recipe is not promoted to production.
