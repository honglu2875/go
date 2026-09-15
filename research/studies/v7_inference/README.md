The persistent native V7 feature worker passed `qualification-001.json` on
30 real 19×19 training games and 12,021 positions. All spatial/global inputs,
boards and legal masks match the previously qualified offline representation
exactly. Full histories, requested suffixes, alternate legal branches, shortened
histories, pass=361, invalid inputs, terminal rejection and recovery after a
rejected request are covered. Small 3×3 and 9×9 histories also match the offline
worker. Test targets were not decoded.

The implementation is `../../recipes/v7_inference/`. Its bounded binary protocol
uses a persistent C++ process for Go replay and V7 extraction, with Python
process/array glue. Each request replays the complete action tape and emits only
the requested suffix, including the pending leaf's board. It does not cache Go
positions across requests. Native MCTS must score terminal leaves directly.

Build receipt: `.gozero/build/katago-v7-stream-001/receipt.json`.
The binary SHA256 is
`f2fd5e4b00ba3daa76fd5f2f67c09f37f65bafb9a20d8be972f47b172c526676`.
It links the exact hash-verified KataGo objects used by the offline V7 worker,
at source revision `92ee95c0a4b25fec214da00951ab69e97e207729`.

This completes the feature-boundary qualification only. Model export, retained
history/cache ownership, batched TPU policy/value inference, native search
integration and real games against pinned KataGo opponents remain to be built
and qualified. The current 9×9 learning runs are unchanged.
