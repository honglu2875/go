# Static score utility pilot

The registered fresh-control comparison did not establish a strength or sample-
efficiency gain. Both arms advanced 4,194,304 real moves from scratch with the
same 307,461-parameter CNN, ownership loss 1.5, score loss 1, FPU reduction 0.2
and 16 search simulations. The intervention was use of the separately learned
bounded score value in search, at factor 0 versus 0.3. Full definitions and
identities are in [pilot_spec.json](pilot_spec.json) and the immutable
pilot_result.json (external or omitted experiment artifact).

| Quantity | Factor-zero control | Factor-0.3 candidate |
|---|---:|---:|
| Global learner updates | 8,092 | 8,093 |
| Completed self-play games | 20,127 | 19,668 |
| Truncated self-play games | 1,969 | 2,409 |
| Eligible terminal rows | 3,498,946 | 3,354,657 |
| KataGo games completed / scheduled | 5 / 8 | 6 / 8 |
| Wins / losses among completed games | 0 / 5 | 0 / 6 |

The external screen used the pinned official 9x9 KataGo weights and engine,
at 1 and 16 visits, with paired colors for empty and C3/G7 openings. All played
boards and completed score margins agreed with KataGo. Five games reached the
registered 324-ply cap and have no assigned result. This observed zero-win floor
and incomplete paired comparison cannot establish superiority or equivalence.
The candidate also produced fewer eligible terminal rows and more truncated
self-play games; these are distribution diagnostics, not strength estimates.

Both arms learned a score head. The experiment isolates its use by search,
not score supervision itself, and is not a complete KataGo utility/search
reproduction. A single seed and short horizon limit the conclusion. Control
evaluation overlapped candidate training on separate physical CPU cores, so
training times are diagnostic rather than an isolated systems comparison.

The two training attempt windows cost 3.2234 chip-hours. This excludes reserved
engineering/idle time; the allocation ledger reports that larger denominator.
No production promotion follows from this pilot. CPU gradients, sharding,
native terminal-utility semantics, GTP integration and exact checkpoint recovery
were qualified before launch and remain reusable.
