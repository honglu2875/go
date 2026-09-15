# Paired causal traces: first execution qualification

The CPU qualification passed on source
`bcefb86b40a2e5c24dbf47f0a5aea470061baf4904add612604800635e543518`.
The immutable receipt (external or omitted experiment artifact) links the full events, HLO,
parameters, config and native binary. Five model tests and fourteen native
binding tests also passed, including causal masking, cached versus fresh
attention, distinct head gradients, stale tickets, atomic batch rejection and
terminal versus capped games. Three native resolver tests cover branch priority
and prefix eligibility.

All three multi-ply modes produced exact prefixes of the sequential policy's
real action streams, including episode resets and terminal scores. The probe
used a 28,768-parameter untrained transformer, four 3x3 games, 64 packets, two
player views, four samples and four predicted plies per packet.

| Opponent predictor | Real moves | Moves / game / dispatch | Legality corrections |
|---|---:|---:|---:|
| Untrained behavior head, independent draws | 344 | 1.3438 | 157 |
| Oracle play head, independent draws | 351 | 1.3711 | 159 |
| Oracle play head, shared draws | 479 | 1.8711 | 217 |

The sequential control advanced 496 moves to cover every compared prefix, at
one move per game per dispatch. Different resolved lengths reach different
states within the fixed packet window; the table is an execution diagnostic,
not a controlled speed comparison. CPU windows were approximately 0.1 seconds
and included concurrent work and instrumentation. They cannot establish a
throughput gain or MFU.

The independent-draw oracle illustrates a limitation: knowing the opponent's
policy does not reveal its random sample. The shared-draw oracle is an explicit
unavailable-information upper bound. Many continuations stopped after an
illegal unmasked proposal was corrected by Rust. Supplying the known root
legal mask and learning future legality are possible follow-ups, with exact
Rust checks retained. Historical behavior calibration, MCTS policy distillation,
learned model experiments and TPU scaling remain untested by this qualification.

The subsequent [registered root-mask probe](root_mask_spec.json) also
passed (external or omitted experiment artifact). Both variants used the same parameters and
native binary; only the first decoded move was masked using Rust's exact root
legality. All 496 shared sequential events agreed across variants, and all
multi-ply streams remained exact sequential prefixes.

| Opponent predictor | Without root mask | With root mask | Resolved-move ratio |
|---|---:|---:|---:|
| Untrained behavior head, independent draws | 1.3438 | 1.7383 | 1.294 |
| Oracle play head, independent draws | 1.3711 | 1.7734 | 1.293 |
| Oracle play head, shared draws | 1.8711 | 2.5820 | 1.380 |

The middle columns are moves per game per dispatch at the same 64-packet
budget. Behavior-mode legality corrections decreased from 157 to 96 despite
more resolved moves. These remain acceptance diagnostics on different-length
prefixes, with an untrained 3x3 model. They establish neither throughput nor
learning improvement. Six model tests and fifteen native binding tests passed
for the root-mask implementation.
