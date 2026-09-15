The explicit Rust `pass_alive_area` profile matched the pinned KataGo C++ scorer
at every one of 1,351,680 intersections in 8,192 checked positions. Coverage is
1,024 positions at each of sizes 1, 3, 5, 7, 9, 13, 19 and 25. The qualification
also replayed all four retained real candidate/KataGo games through the official
GTP binary: all boards and final numeric scores matched. Raw area margins
differed in 1,879 sampled positions and in three of the four actual games.

result.json (external or omitted experiment artifact) pins the source, native build, external compiler and
source files, oracle executable, generated cases, pointwise outputs, GTP engine,
weights, configuration and original execution report. Inputs and transcripts
remain in `runs/qualification/scoring-3b3e372f`. The first adapter attempt failed
at its default 19x19 compile-time capacity; that failed record is retained. The
successful adapter uses the same scoring source with an explicit 25x25 capacity.
No neural scores are used by the ownership oracle.

The profile preserves board stones and complete superko history. Search terminal
values, final outcomes and spatial labels share one explicit actor setting. Raw
Tromp–Taylor area remains the default. Rust regression tests cover the observed
pass-dead stone, all eight board symmetries, both colors, and a komi where the
scoring choice changes the winner. Native binding tests check current-player
ownership signs, outcome consistency, truncation exclusion and exact recovery.

This is tested scoring compatibility, not a proof over all legal Go positions,
a general Japanese-rules implementation, or evidence of stronger play. The
four historical matches remain four losses; their original failed reports are
unchanged. The new ownership recipe can now compare training methods against
KataGo with matching adjudication.
