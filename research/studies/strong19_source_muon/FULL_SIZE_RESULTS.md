The full-size joint source CNN passed all four ranks, including exact recovery
from a two-update prefix to the four-update endpoint. The resumed checkpoint
matches the uninterrupted model, optimizer/Lookahead state, source sample
clock, draw/D4 generators, validation and training-probe histories. Metrics
match after excluding elapsed-time measurements. The run contains 233,220,870
training parameters and 243,326 live position exposures.

The first attempt exhausted HBM while compiling a 32-frame chunk and performed
no updates. Changing only the execution chunk to 16 frames allowed both the
512- and 1,536-position shapes to compile and run. Compiler temporary buffers
were 21,421,031,424 and 24,030,412,800 bytes respectively. These compiler
measurements differ from the separately recorded device allocator statistics.
Neither is an MFU measurement.

The uninterrupted four updates took 382.23 seconds of learning time; the whole
attempt took 982.43 seconds, including initialization, compilation, evaluation
and checkpoint handling. Prefix and resumed attempts took 758.91 and 764.07
seconds. Evaluation includes its compilation time, so individual timers must
not be added as disjoint categories. This short fixture deliberately exercises
both shapes and recovery boundaries; it is not a scientific training run.

All three qualification checkpoints retain two hash-verified, read-only peer
RAM copies and restore metadata. Removing only the owner's array caches
reclaimed 8,417,432,874 bytes. RAM copies remain volatile across restarts.

This closes the source execution/recovery gate. The selected transformer's
actual execution check and the registered fixed-cohort learning comparison
are separate steps. Detailed numeric evidence is in FULL_SIZE_RESULTS_002.json.
