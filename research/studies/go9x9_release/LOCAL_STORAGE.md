The finalized 9×9 release remains backed up at the immutable Hub revision
recorded in `completion.json`. Its complete local release mirror has moved to
the peer identified in `local-mirror-001.json`. Use that locator for the current
copy; the earlier completion record preserves the historical upload location.

`mirror-relocation-plan-001.json` covers 54 files, 4,969,566,646 logical bytes.
The copy was fully hash-verified before removing 62 old archive paths, including
eight additional hard links to the same covered files in the completed packaging
directory. It reclaimed 4,969,660,416 allocated bytes. Original distributed game
NPZs, packed V7 training arrays and all learning checkpoints were preserved.

The relocation command exited with a relative-path error while writing its last
locator, after successfully writing the copy/retirement receipt. It was not
rerun. `mirror-relocation-post-audit-001.json` records a separate full destination
read-back, exact retirement-journal coverage, absence of all planned old paths,
and the repaired locator. All checks passed. Preserve the original operator and
plan as executed evidence; do not rerun this one-shot relocation.

Deployment locators and relocation receipts are private operational records and
must not be copied into a public source export without sanitization.
