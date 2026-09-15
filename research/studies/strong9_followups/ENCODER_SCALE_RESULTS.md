The encoder residual-scale change from 1e-6 to 0.01 passed both registered seeds. We retain 0.01 as the transformer starting point for subsequent experiments. Parameter count (231,181,121), decoding FLOPs, model shapes, learning rate, training draws and targets were unchanged.

| Seed | Original position KL | Candidate position KL | Position gain | Family gain |
| --- | ---: | ---: | ---: | ---: |
| 91312427 | 0.133493900 | 0.131976008 | 1.137% | 0.668% |
| 91312428 | 0.134828091 | 0.131123066 | 2.748% | 1.639% |

Mean paired endpoint improvements are 1.943% per position and 1.153% per opening family. Both seeds also improved both last-three means; neither triggered sustained overfitting. Each completed 4,096 accepted updates, with about 49.8 million position exposures. All-rank metrics, sample replay and validation populations passed the existing audit.

The CNN still leads at equal updates: the candidate remains 1.794% higher in position KL and 3.155% higher in family KL on average. The [learning-time analysis](LEARNING_TIME.md) addresses a separate comparison; lower fixed-data loss is not a playing-strength or RL-efficiency result.

The complete curves show an early improvement with some later crossings; the acceptance decision uses the predeclared endpoint and tail rules. Both final checkpoints retain verified peer copies. The [first-seed note](scale1e-2-first-seed-results-001.md) is historical; the [completed two-seed result](scale1e-2-results-001.json) supersedes its pending status.

Final curves and underlying tables: [seed 1 CSV](scale1e-2/seed1-final-001.csv), [seed 2 CSV](scale1e-2/seed2-final-001.csv). Locally the matching PNG files have been visually checked for readable labels, legends and complete curves.
