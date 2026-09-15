The architecture and joint value-head implementation follow the larger 9×9
decision; no 19×19 training experiment is registered yet. The policy-only budget
options are in `../research_scaleup_20260915/19-policy-budget-options-001.json`.
Reducing the single-token connector from 64 to 16 channels analytically matches
19×19 parameter and decoding budgets without narrowing the 768-wide encoder or
transformer. Value heads, all deployed outputs, and complete-history HBM need
fresh tracing and real full-shape qualification.

Raw value labels are signed player-to-move values in [-1,1]. Joint models and
MCTS backup must use that exact perspective. Overall MSE is insufficient on the
lopsided mixed-strength games: also report phase and opponent strata, including
teacher self-play. Test labels remain closed during selection.

Do not call the historical SGD recipe the published b40 model's official
optimizer. The existing provenance study `../visual_katago/OFFICIAL_TRAINING.md`
verified Muon in the b40 export metadata and an author-reported early-SGD/later-Muon
training history; its exact LR schedule and batch size remain unverified. The
separate 2019 SGD transfer control did complete 44,803 updates over the same
11,469,333 exposures as the old AdamW comparison, finishing at KL 0.424920 versus
0.420361 for selected AdamW. See `../visual_katago/sgd_results/REPORT.md`. This
does not establish which optimizer is best on the new teacher or 19×19 corpus.

The teacher is the pinned modern tf3-b11c768 network, approximately 70.4M exported
parameters, not the historical 233M CNN. Stronger play against intermediate
KataGo networks and stronger play than the modern teacher are distinct tests.
The older causal/visual GTP adapters use different feature/model contracts; they
cannot be pointed at these checkpoints without a new validated adapter.
