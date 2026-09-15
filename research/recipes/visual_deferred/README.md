# Deferred repair in continuous policy decoding

This complete clone keeps the trained 233M model unchanged. After a rejection,
it commits the legal replacement to the exact board and marks its observation
as pending in the deep cache. The next packet computes a cheap shallow
prediction for that pending root, then verifies the pending observation and new
proposals together. Maximal coupling uses the actual full-depth target produced
by that verification, never the stale parent prediction.

The timer includes rollback, all pending work, host coupling, and a final full
cache drain. CPU checks cover forced zero-prefix replacement, multiple pending
packets, exact native boards, deep pending-root targets and final cache lengths.
The frozen CPU profile and both registered multi-host TPU runs passed their
execution checks. The full-model audit replayed 24,224 committed moves and 639
terminal paths; the largest reported full-reference policy TV was 0.005295.
Median throughput relative to the contemporaneous sequential control was 0.596x
at horizon 2 and 0.539x at horizon 4, failing the 1.20x timing gate. Removing eager
deep repair did not improve this bounded loop. See
`research/studies/visual_causal/deferred_result.json` for timing scopes and
numerical limitations. Individual acceptance probabilities and uniforms were
not retained for independent draw replay.

This is expert-policy sampling on both players. The separate behavior output
remains part of the model, but this intervention does not retrain it or replace
MCTS decisions. Its source, configurations, control timing and failed attempts
must remain pinned like other research recipes.
