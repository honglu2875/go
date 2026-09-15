# Legacy checkpoint cold storage

`ops/archive_checkpoint_arrays.py` can reversibly archive uncompressed NPZ
payloads from completed historical pod attempts. It never changes original
checkpoint manifests, parameters, replay contents or NPZ encoding. It gzip-wraps
the original file bytes and checks decompression against the original SHA-256
before publishing an immutable receipt and removing a redundant raw copy.

The archive store is `.gozero/checkpoint-archives/<original-sha256>.npz.gz`.
Receipts are beside checkpoint directories, named
`turn-NNNNNNNNN.arrays-archive.json`. A checkpoint whose raw array file is archived
must be explicitly restored before a historical frozen reader or analyzer is run.
Restoration reproduces all original bytes and passes the original strict
four-file checkpoint reader. Model exports and the other checkpoint files remain
in place. Receipt and archive hashes belong in subsequent artifact inventories.

The operator requires a pinned plan and a real-checkpoint round-trip qualification
before removing raw files. Unit tests cover exact array/state restoration,
corrupt archives, and writable/changed originals. Partial operations retain a
per-checkpoint receipt before removal and a progress journal for audit/recovery.

Use a frozen operator for every action. `plan` selects only read-only NPZ files
over 100 MB with ZIP_STORED members from closed attempts. `qualify` decompresses
one selected real checkpoint to a separate retained directory and invokes the
ordinary checkpoint reader. `apply` archives the frozen plan. `restore` takes a
receipt path and expected receipt SHA-256 and restores its original path.

This is storage compression on the existing pod, not external durability. The
uncompressed byte size remains the space requirement for restoration. Existing
historical raw-file artifact indexes require that restoration first.
