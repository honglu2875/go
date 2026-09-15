# Static score utility control

This complete clone of `low_visit_puct` adds a separate score head. It predicts
the expected bounded terminal score value
`s = (2/pi) * atan(current_player_margin / (2 * board_size))`.
Terminal ownership labels plus the signed komi observation reconstruct that
margin without a new replay field or binding ABI. The win head keeps the true
win/loss target. Both candidate and control train the score head with MSE at
coefficient 1.0 and ownership at coefficient 1.5.

The candidate search utility is `(win_value + 0.3 * score_value) / 1.3`; the
control factor is zero. Rust terminal evaluation uses the same formula and
selected scoring profile. PUCT exploration and FPU reduction are divided by the
same denominator, so normalization to the native [-1,1] interface does not
silently increase exploration relative to utility. Recorded game scores,
win/loss labels and external adjudication stay on the ordinary score contract.

The arctangent and static scale 2 follow the pinned KataGo scorer's utility
shape. Factor 0.3 is its constructor's static factor, not a claim about every
modern training configuration. Our head directly learns expected bounded score
value; KataGo combines score-distribution estimates with static and dynamic
utilities and additional search machinery. This recipe is one component
control, not that complete implementation. See the pinned
[utility calculation](https://github.com/lightvector/KataGo/blob/92ee95c0a4b25fec214da00951ab69e97e207729/cpp/search/searchhelpers.cpp)
and [parameter definitions](https://raw.githubusercontent.com/lightvector/KataGo/92ee95c0a4b25fec214da00951ab69e97e207729/cpp/search/searchparams.h).

The motivation is to test whether graded score preferences improve low-budget
search and endgame behavior where win/loss estimates provide little distinction.
The earlier long games do not establish that this is their dominant cause.
The score-head draw is appended after all inherited parameters. Both arms have
the same complete parameter tree and initialization, including the auxiliary
head. Neither model receives KataGo training labels or weights.

The CPU trainer, native/GTP path, gradients and one-device/four-device updates
are qualified. Fresh CPU continuation matched 60 arrays, complete actor state
and 33 subsequent games exactly. The 9x9 control/candidate pilot was registered
before launch in `research/studies/score_utility/pilot_spec.json`; both arms
completed 4.19 million moves. Neither won in its eight-game KataGo panel; five
games in total hit the cap. The immutable analysis and limitations are in
`research/studies/score_utility/README.md`. No strength, sample-efficiency or
production promotion was established.
