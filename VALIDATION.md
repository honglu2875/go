The sanitized source publication passed the shared Python infrastructure suite:
119 tests ran, 97 passed and 22 were skipped for optional fixtures. The run used
CPU JAX and a separately qualified local Rust extension for the native replay
checks that accept an explicit fixture. The binary itself is not published.

An initial restricted run lacked that native fixture and permission for local
socket tests; those environmental prerequisites were supplied for the passing
run. It was not an accelerator training or playing-strength evaluation.

All published Python files are syntax checked; JSON, TOML and dependency locks
are parsed. Publication checks reject credentials, original machine identifiers,
private network addresses, original affinity assignments, symlinks and binary
artifacts. The exact staged Git contents are checked against the public SHA-256
manifest before pushing.

Historical model tests and full training qualifications require their external
datasets, runtime and native artifacts. The shared-suite result does not imply
that every historical recipe has been rerun from this sanitized publication.
