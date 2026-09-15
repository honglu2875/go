# Research direction after the first 233M visual pilot

This is a proposed next programme, not a claim that the proposed interventions
have been implemented or shown to improve Go. The completed pilot and its
negative results are linked from [README.md](README.md). Measure strength gained
per reserved chip-hour first, then useful rollout throughput and latency. MFU is
a diagnostic; extra speculative work can raise utilization while delaying useful
learning.

## Preserve the two objectives

The expert head learns the improved MCTS distribution. The behavior head learns
the actual opponent's observed action, conditioned on the available history. A
weak opponent is a valid behavior target. Keep a separate value estimate for
search. Do not relabel behavior examples with the expert's preferred move.

The conditional log-density gap

`r(h,a) = log pi_expert(a|h) - log pi_behavior(a|h)`

is not automatically a value advantage. It becomes proportional to an advantage,
up to a history-dependent normalization, only under an additional model such as
`pi_expert(a|h) proportional to pi_behavior(a|h) exp(A(h,a)/tau)`.
Without that assumption, its expectation under the expert is a KL divergence,
and it can reflect style, model error, or support mismatch. Use it first as a
diagnostic of policy disagreement and proposal coverage. If testing an advantage
interpretation, compare it with independently estimated search Q differences,
with held-out games and calibrated uncertainty.

A useful third output may be a cheap **proposal** adapter. It can learn to imitate
whichever full policy will verify a branch, while the behavior output continues
to model the real opponent faithfully. Test shared versus stopped behavior-trunk
gradients separately. A mixed-strength opponent league and held-out opponent IDs
are needed to test style/level calibration; the current archived engine-move corpus
does not establish that ability.

## Make speculation useful to search

The original paired-trace idea and speculative MCTS are related but have different
verification requirements. The published speculative-MCTS approach pipelines
future decisions using partial search results and reuses neural evaluations. Our
causal packet additionally carries history-conditioned opponent forecasts and
can perform several exact board transitions in one graph. Neither agreement
between two forecast traces nor policy-sampling acceptance proves that a fixed
MCTS budget would select those moves. [Speculative Monte-Carlo Tree Search](https://papers.nips.cc/paper_files/paper/2024/hash/a19940b01b77b6acd41ff8b32b334e7c-Abstract-Conference.html)

The next search integration should first populate speculative **evaluation
caches**, committing actual moves only after the ordinary registered search
decision. Give each entry a complete history identity, rules/komi, network
version, policy role and branch position. A board hash alone is insufficient for
a history-conditioned model and exact superko. Compare all search requests,
completed-Q targets, root choices and work counts with a sequential reference.
Only then explore changing the decision rule as an explicitly approximate search
algorithm, evaluated at both equal search work and equal elapsed compute.

Prioritize short horizons and adaptive allocation. Estimate
`expected usable evaluations / (draft + verify + commit + repair time)` on held-out
roots. Include tail latency, stale work, terminal paths, KV bytes and discarded
branches. Test `k` independent proposals only after single-path acceptance is
adequate: width can multiply redundant work. The `visual_loop` prototype charges
for rollback and replacement inference. It measured 0.726x/0.724x at H2/H4;
deferring deep repair into the next packet measured 0.596x/0.539x, also below its
serial control. Both are bounded policy-decoding screens. The latter's full
audit checked 24,224 committed moves; details (external or omitted experiment artifact) preserve
numerical and timing limitations. Neither screen validates MCTS integration.

LayerSkip motivates training early exits and sharing early-layer computation
between drafting and verification. Our present verifier recomputes those layers.
An intervention should retain their hidden activations and KV, execute only the
remaining layers for verification, and validate rejected-prefix repair. Add layer
dropout or exit-specific adapters only as separately registered learning changes.
LLM speedups do not predict TPU Go speedups. [LayerSkip](https://arxiv.org/abs/2404.16710)

## Reduce work per observation as well as dispatch count

The current 3x3 patches cost 11 tokens per 9x9 position and 51 per 19x19 position,
including readout and action. This makes long Go histories expensive. Compare
the current ViT-like embedding with a spatial encoder producing 1, 4 or 9
continuous observation tokens. There is no need for a categorical vocabulary of
all possible boards: a continuous projection can accept the board planes
directly. Preserve legal masks, board geometry, color, komi, pass state and full
causal action history.

A further hypothesis is periodic full-board tokens plus exact changed-patch
tokens. Captures, legal-mask changes and global metadata must all be represented;
a move-coordinate-only delta is insufficient. Require lossless reconstruction of
the observation stream before evaluating a compressed representation. Compare
equal parameter count, equal inference latency and equal training compute as
three different experiments. The completed 232.39M residual-CNN versus 233.14M
transformer screen used 128 matched updates and distinct history contracts. It is a conventional
CNN baseline, not an exact KataGo architecture port. Follow it with a windowed
transformer control, a semantic AdamW decay mask, and compute-matched learning
curves before attributing any gap to architecture alone.

## Improve learning before scaling the model

The D4 pilot is a useful baseline result, and the first 256-update self-play
continuation did not improve the fresh KataGo panel. Start with fresh/old replay
mix, learning rate and update-to-new-game ratio; preserve the full Adam state
and fixed held-out sets. The audited continuation drew 842 distinct fresh expert
games versus 7,730 old-data expert games, with 18.66 versus 1.86 exposures per
available expert position. These are descriptive counts, not a causal finding.
Report the effective number of distinct games as well
as position exposures, since many outcome labels in one game are correlated.
Use collection holdouts to diagnose the shift from the weak old teacher to
self-play, without changing endpoint selection after seeing the results.

Reproduce playout-cap randomization, policy-target pruning, global spatial
pooling and auxiliary future-policy targets in small controlled trials before
claiming a fair comparison with KataGo's training recipe. Their benefits were
established in a different architecture and training regime, so coefficients
must be tested here. The Jane Street paper mentioned in the initial discussion
is David Wu's KataGo work. [KataGo paper](https://arxiv.org/abs/1902.10565),
[author's Jane Street account](https://blog.janestreet.com/accelerating-self-play-learning-in-go/)

Use successive 5x5/7x7/9x9 screening, then fresh 19x19 confirmation, with at least
three independent seeds for any claimed learning gain. Treat extrapolation
across board sizes as a hypothesis. Jones's scaling study used Hex, not Go.
[Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113)
Keep the AlphaZero policy/value/self-play objective as a defined reference rather
than treating every heuristic coefficient as universally optimal.
[AlphaZero](https://arxiv.org/abs/1712.01815)

The pinned KataGo training source also contains optional Muon, Normuon and Aurora
optimizer paths, with auxiliary Adam parameters and configurable Newton–Schulz
iterations. A modern optimizer comparison is therefore a concrete additional
axis, not evidence that our AdamW baseline reproduces current KataGo training.
An eventual pure-JAX port should specify matrix layout/scaling, embedding and
head exclusions, weight decay, precision, complete optimizer-state recovery and
equal-compute learning curves. This window did not test those optimizers.
[Pinned KataGo training source](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/python/train.py)

## Production handoff requirements

Use the qualified snapshot, native build, dataset, checkpoint-group and benchmark
identities already carried by this repository. The remaining production work is
a durable generation/learner coordinator, bounded model staleness, automatic
replay publication, retention and external restore verification. The current
visual learner has exact controlled checkpoint-boundary recovery; host loss and
external durability are not qualified. Keep actor game ownership on fixed native
workers, use bounded queues and local inference, and measure actual cache/NUMA
traffic before attributing a slowdown to cache-line invalidation.

Promotion should require a replicated learning-curve gain against the pinned
KataGo ladder, using fresh paired openings, explicit search/time controls, and
unresolved games retained as unresolved. Include failed attempts, teacher data
generation and engineering/idle reservation time in costs. Do not launch a long
production training campaign merely because a kernel screen or one weak-anchor
win rate looks encouraging.


## Follow the large CNN screen with controlled architecture changes

The completed 128-update comparison favors the CNN on held-out imitation/value
losses, but the CNN took 2.20x the learning time and sees eight recent observations
rather than the transformer's complete history. The separately audited common-
opponent KataGo panel also favored the CNN, even under the most favorable
assignment of unresolved outcomes to the transformer. Those sample bounds are
not confidence intervals. This short screen does not establish a superior long-run
architecture or an optimal schedule for either model.

Two small architectural ablations are more informative next than simply adding
transformer layers. First, compare the tied categorical action readout with a
spatial policy head using patch-local features plus the global causal readout.
Second, compare linear patch projection with a small residual spatial encoder
before causal sequence modeling. Keep expert and behavior objectives distinct,
preserve exact legality, and test future-action leakage for every new mask.
A windowed-transformer control is needed to separate history effects.

Use a longer common schedule with checkpoints at predefined update counts and
measured learning-cost milestones; do not extend a schedule selected from this
screen and call it a compute-matched comparison. Test ownership and score
auxiliaries in both families, since the current pilot left these losses disabled.
An exact modern KataGo architecture/training port remains a separate baseline.
