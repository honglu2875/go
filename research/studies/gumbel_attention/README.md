# Gumbel search with an attention model

The combined recipe has passed CPU execution and recovery qualification.
Source `0f2cf041bc75abdc3f3063ac332f43d6d92acd2ec6be82d333138d44feb267b6`
completed 96 turns and 88 learner updates. A fresh process resumed at turn 48;
all 82 final arrays, full actors, non-timing scientific state, and 20 subsequent
games matched the uninterrupted run. Nine model tests also passed.

The CNN branch on source
`c458f549cae6862befd773fd6692531ff948081e1b4d8b15bf4d82be8008e385`
reproduced the earlier Gumbel CPU run exactly: 60 saved arrays, 55 complete or
capped games, and identical exported weights. The comparison explicitly
accounts for the extra CNN architecture fields and worker placement changing
from CPUs 8–9 to 104–105. No scientific setting changed. The first CNN invocation
used a CPU affinity excluding its workers and failed before rollout; that
failed record is retained alongside the corrected run.

The [multi-host qualification](qualification_spec.json) also passed. Continuous
96-turn execution and fresh-process continuation from turn 48 matched all 328
saved arrays, complete actors, non-timing scientific state and 400 subsequent
games. The result (external or omitted experiment artifact) pins both attempts and CPU checks.

The separate [9x9 shape qualification](shape_9x9_spec.json) completed 131,072
global moves, 189 learner updates and 1,055 games without truncation. All four
hosts exported identical weights. It used 0.1713 attempt chip-hours; the
result (external or omitted experiment artifact) records source, native and model identities.

The [registered architecture pilot](pilot_spec.json) completed and met its
exploratory criterion. The attention checkpoint won **55 of 64 games** against
the reused Gumbel CNN, with all 6,343 boards and 64 final scores verified by
real KataGo. Its paired score was 0.859375 with conservative 95% interval
[0.6193, 1]. It won 26/32 as Black and 29/32 as White; no direct game ended
within 25 plies. The immutable analysis (external or omitted experiment artifact) pins every result.

Training used the same 16,777,216 real moves and unchanged non-model settings.
The full trainer was byte-identical. The attention trunk had six blocks and
311,425 parameters; the CNN had four blocks and 307,461. The candidate completed
195,565 self-play games, truncated 25 and generated 16,736,768 terminal rows,
compared with 123,941 games, 479 truncations and 16,589,675 rows for the CNN.
Its 32,701 learner updates slightly exceed the control's 32,695 because warmup
depends on completed-game data. These distribution diagnostics are not an
independent sample-efficiency result.

The new training attempt used 6.9948 chip-hours versus 5.8375 for the reused
control. Maximum training segments were 1,535.94 and 1,283.38 seconds. Compute
and architecture-specific tuning were not matched. The two new evaluation
panels used 456.58 seconds of the registered 2,400-second allowance.

The strong KataGo anchor yielded **zero wins, five losses and three caps**.
Capped games have no assigned outcome. This single-seed screen supports an
independent-seed follow-up and absolute learning curves; it does not establish
strong Go play or promote the architecture to production. Current official
KataGo already uses more advanced transformers, so attention itself is not a
novel contribution.

The [independent seed-28 replication](replication_28_spec.json) completed, and
its immutable analysis (external or omitted experiment artifact) **did not meet the registered
replication criterion**. Attention won 39/64 direct games (60.94%), with paired
95% interval [36.93%, 84.95%]. The criterion required at least 65% and a lower
bound above 50%. All 5,671 boards and 64 scores matched KataGo. Wins were 19/32
as Black and 20/32 as White, with no games ending within 25 plies.

The unchanged attention model/trainer used 16,777,216 moves, 32,687 learner
updates and 7.1048 new attempt chip-hours. It produced 208,181 completed games,
646 caps and 16,549,197 terminal rows. The reused independent-seed CNN produced
189,393 games, 259 caps and 16,638,648 rows. The attention candidate's strong
anchor was eight losses, with all 716 boards and eight scores matching and no
caps. New direct and anchor panels used 291.04 seconds of the registered
2,400-second budget. The control's prior compute and anchor remain recorded.

The separate [registered checkpoint curves](../architecture_curves/README.md)
show positive conservative average differences against historical KataGo in
both seeds. Their full criterion also failed because one CNN checkpoint fell
below the completion threshold. Taken together, the evidence is mixed: the
architecture learns faster against this historical opponent at the measured
budgets, while its final direct advantage did not meet the independent-seed
criterion. Neither study supports production promotion or strong-KataGo claims.
