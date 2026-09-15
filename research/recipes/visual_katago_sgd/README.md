This is one historical KataGo SGD control on our existing CNN and fixed data.
It follows the user's choice to match **256 board positions per update** while
keeping total position exposures comparable. It is separate from the registered
transformer encoder-capacity grid.

The public June2019 trainer at commit
`706ed3ca31c819c8e555194accfe584ff6e5c95f` uses Nesterov SGD, momentum0.9,
and a per-sample rate6e-5 with an initial factor1/3. Appendix C of the paper
specifies the first5M training samples as that warmup. The actual source uses
L2 coefficient3e-5 multiplying TensorFlow's `tf.nn.l2_loss`, which is half the
squared norm. This convention gives a coupled kernel gradient3e-5 times the
weight. We follow the implementation's convention explicitly; do not silently
double it by interpreting the paper's abbreviated squared-norm notation.

| Setting | Registered control |
|---|---|
| Optimizer | Nesterov SGD, momentum0.9 |
| Batch | 256 positions; final partial batch21 |
| Warmup | Per-sample2e-5 through5M positions |
| Main rate | Per-sample6e-5 |
| Mean-loss equivalents at batch256 | 0.00512, then0.01536 |
| L2 | Coupled3e-5, half-squared-norm convention, kernel weights |
| Gradient clipping / Lookahead | Neither |
| Late LR drop / cosine | Neither in this short control |
| Training exposures | Exactly11,469,333 |
| SGD updates | 44,803 |
| Updates per compiled dispatch | Up to32 genuine sequential updates |
| Initialization | Same CNN seed91312427 and parameter bytes |
| Validation | Same full1,170 games /102,339 positions; test closed |

The momentum buffer uses **summed-gradient units**, matching the historical
trainer. Thus the final partial batch also handles accumulated momentum
correctly. We retain the current CNN's folded RepVGG center-gradient multiplier;
its coupled decay matches independently regularized unfused branches. Independent
tests compare those branches with folded updates, rather than merely duplicating
the optimizer code.

The position stream reconstructs every original episode and D4 draw from the
completed CNN control. A fixed permutation (seed91313027) shuffles all exposures
before grouping them into256-position batches. An independent position/symmetry
histogram verifies that no sample was added or lost and that every position is
from the training split. This preserves the **multiset**, not the original order
or game-sized batches. The original game-based sampling is also not exactly the
uniform position sampling of KataGo's moving replay window.

Validation occurs at the first SGD update boundary reaching each original CNN
validation exposure count, at most255 extra positions at intermediate points.
The final exposure count matches exactly. The kernel microbatch is16 rather than
32 to fit the16 positions per TPU device; model parameters and logical batch128
decode FLOPs are unchanged. Independent forward and global-gradient tests cover
the microbatch execution change and masks, including an empty device shard.

The current232M policy-only CNN has a norm-free main trunk and the existing
same-target training helper (main/helper loss weights0.2/0.8). These, BF16
arithmetic,9x9 weak-teacher fixed data and the absence of value/ownership/score
objectives differ from the2019 run. This tests transfer of its SGD recipe;
it does not reproduce that self-play run or validate the modern233M model's
unpublished LR schedule. Its19-day final LR drop is not compressed into this
short experiment. Training update counts, optimizer FLOPs and wall time differ
from the AdamW controls and are reported separately.

SGD carries parameters and momentum through `lax.scan`, emitting only small
metrics. Padding creates no optimizer step. Any rejected update latches off
the remaining updates in the dispatch and stops the run without a numerical
retry. CPU qualification checks independent optimizer equations, distributed
gradients, exact sample transformations, grouped versus separate updates, and
checkpoint/resume equality.

Before learning, host0 allocates persistent space for the complete checkpoint.
The ordinary non-pickle checkpoint is staged in `/dev/shm`, copied into that
already allocated file and hash-verified before atomic publication. This consumes
only the new run's reservation. Existing checkpoint copies remain in place.
Other ranks retain a hash of each complete metrics block, while host0 stores
the full metrics; the auditor checks every rank's block hashes and final digest.

The bounded runner executes one TPU attempt, audits it independently, publishes
curves against both AdamW1e-4 and the selected AdamW1e-3 endpoint, and updates
the pod cost ledger. AdamW1e-3 was selected from four tested rates; SGD is one
pre-registered recipe. Neither is a known global optimum. Go strength and RL
sample efficiency are outside this control's claims.

Primary sources: [paper, Appendices B/C](https://arxiv.org/pdf/1902.10565),
[pinned historical trainer/model](https://github.com/lightvector/KataGo/blob/706ed3ca31c819c8e555194accfe584ff6e5c95f/python/model.py#L1529).
Retrieved source bytes and hashes are retained under
`.gozero/external/research/katago-sgd-2019/706ed3ca31c819c8e555194accfe584ff6e5c95f/`.
