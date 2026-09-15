# Distributed cancellation qualification

This bounded CPU-only SPMD fixture deliberately exits host rank2 with code23
after three seconds. Peers wait up to sixty seconds. Every trainer creates one
descendant that ignores SIGTERM, so merely stopping the leader cannot pass the
cleanup check. The new launcher must publish attempt-scoped cancellation on all
four hosts, collect failed/cancelled rank receipts promptly, and leave every
recorded descendant absent or a zombie with the same process identity. JAX and
TPUs are never initialized by this recipe.

The pod attempt is expected to have status `failed`. A separate qualification
receipt must verify the expected injected failure, cancellation delivery, peer
statuses, source integrity, bounded latency and actual remote process states.
This tests a reachable-host failure, not a network partition or learner recovery.
