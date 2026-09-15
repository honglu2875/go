This repository is a source backup and a starting point for new research runs.
It contains the Rust engine, Python library, cloneable JAX recipes, evaluation
tools, scientific protocols and research summaries available at publication.

Real SSH targets, account home paths and workspace paths have been replaced
with generic examples. Machine inventories, private process/deployment
receipts, credentials, execution logs, local caches, temporary directories,
datasets, model weights and binary archives are not included. Text reports
and their compact numerical summaries have been sanitized. Raster and vector
figures are excluded because rendered glyphs and metadata can retain deployment
information; source plotting code and CSV tables remain.

`ops/hosts.example.json` is an example only. Copy it to the ignored
`ops/hosts.json`, configure your environment and adapt historical recipe
deployment defaults before launching work. The source snapshot mechanism will
record those choices for your new run.

Historical source IDs, checksums and result claims identify the original
experiments. Anonymization changes some bytes, and private artifacts are not
reconstructed by this publication. Historical audit scripts intentionally
reject changed or missing inputs. To reproduce an experiment, obtain its
external data and model references, inspect the recipe, create a new immutable
source/configuration snapshot, and run the validation gates again. This
publication does not claim that every historical end-to-end experiment can be
replayed from this Git repository alone.

`PUBLIC_SOURCE_MANIFEST.json` inventories the actual published files and their
SHA-256 hashes. It excludes itself to avoid a circular digest. The initial
Git commit has no parent from the repository's previous project.
