The later strength harness needs the exact 22 spatial and 19 global V7 features
used in training. The older visual RPC adapter supplies six different planes and
cannot be used with these checkpoints directly.

Qualify a persistent C++ replay worker linked to the already pinned KataGo
feature objects. Requests provide complete action histories and optional suffix
starts. The worker replays the full legal history, emits only requested board
rows including the pending leaf, and returns binary arrays with a bounded framed
header. Python handles process and model glue; board advancement and feature
extraction remain in native code. Terminal leaves are scored by native search
and must be rejected by this evaluator.

Compare every emitted spatial/global/board/legal row against the existing
independently qualified offline V7 worker on real training histories. Check
branching, pass=361, suffixes, invalid/terminal histories, framing bounds, and
continued operation after a rejected request. Keep test targets closed. This
qualifies the feature boundary only; it does not validate model export, cached
policy/value inference, MCTS integration, or Go strength.
