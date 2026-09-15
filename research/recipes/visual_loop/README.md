# Continuous speculative policy decoding

`loop.py` anchors the first proposal in the full expert policy, uses shallow
causal drafting, verifies the exact path at full depth, then commits the accepted
prefix and scores any residual replacement. Retained KV, rollback and final
cache completion are included in the timed recurrence. The target is the expert
policy on both players. This does not preserve MCTS decisions by construction.

The CPU test covers multi-packet continuation and forced replacement against
native boards and full-prefix predictions. Small and 233M multi-host TPU checks
passed. The full audit replayed 24,343 committed moves. At global batch 128,
horizons 2 and 4 achieved 0.726x and 0.724x median sequential throughput over
three repetitions, failing the 1.20x screen. Complete histories start at position
64 and add up to 32 actual moves; terminal paths remain recorded. Maximum final
full-reference policy TV was 0.004351 under BF16.

The audit retains raw action tapes and final cache metadata. Individual proposal
probabilities and acceptance uniforms were not persisted in this loop screen;
the separate original-packet followup does reproduce every acceptance draw.
Neither result establishes hardware MFU, MCTS acceleration or Go strength.

See the loop registration, CPU/TPU qualification and result in the visual study.
The `visual_deferred` clone investigates postponing replacement verification.
