The original larger9 transformer learns more slowly per update than the CNN,
but its measured training throughput gives a useful improvement in the
learning-time comparison. This observation does not change the registered
same-update selection rule.

At the transformer endpoint, use its slowest rank's cumulative learning clock.
Then choose the first observed CNN validation checkpoint whose fastest rank
has already spent at least that much learning time. This compares actual
observations and requires no interpolation of an unobserved loss.

| Paired seed | Transformer updates / time | CNN updates / time | Position KL, T / CNN | Family KL, T / CNN |
| --- | --- | --- | --- | --- |
| seed1 | 4096 / 98.79 min | 3328 / 100.58 min | 0.133494 / 0.142454 | 0.093454 / 0.099193 |
| seed2 | 4096 / 98.96 min | 3328 / 100.84 min | 0.134828 / 0.140394 | 0.094256 / 0.096832 |

The transformer has lower KL on both measures with less measured learning
time in both pairs. It has also processed approximately 49.8 million positions,
versus 40.4 million for these CNN checkpoints. The CNN's later 4,096-update
endpoint remains better at equal position exposure. The scale-0.01 first seed
is shown separately in this historical timing observation. Its subsequent
second-seed confirmation passed; see [the paired result](ENCODER_SCALE_RESULTS.md).

These are **learning-update clocks**. They exclude data generation, sampling,
compilation, evaluation and checkpointing, and they do not establish a playing
strength, RL-efficiency or full-job-time advantage. Parameter/decoding-FLOP
matching also does not imply equal training FLOPs. The time plot shows later
observations; all observations, timing intervals and exposure counts are
retained in [the observation](learning-time-001.json) and its CSV.

The readable plot is `learning-time-001.png`. The two full same-update curves
and the original architecture conclusion remain the primary learnability
evidence for that question.
