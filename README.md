# GoZero

Go engine, self-play infrastructure, and experiments on causal visual models
for Go. Rust owns board rules, search, replay and persistent environment actors.
Models and optimizers use plain JAX with small Python interfaces; there is no
Flax dependency.

This repository contains research in progress. A strong trained 19×19 engine
and faster reinforcement learning than KataGo remain objectives, not results.

```text
crates/              Rust rules, MCTS, actors, Python bindings and GTP
packages/gozero/     Importable infrastructure and source snapshot tools
research/recipes/    Cloneable models, trainers, objectives and configurations
research/studies/    Protocols, analyses, comparisons and negative results
production/         Space for promoted training recipes
eval/               Validation, model export and real KataGo match harnesses
ops/                Distributed launch, feature preparation and recovery tools
tests/              Shared infrastructure checks
```

## Research direction

The current causal candidate encodes a board into one soft token while retaining
spatial features for its policy readout. A causal transformer processes the
board/action history. The policy combines local board features with a query
from the temporal representation. A shared encoder supports repeated passes
and a training-only first-pass objective.

The comparison program uses fixed teacher data, matching model width, parameter
counts and complete decoding FLOPs, including the encoder. Learning-rate and
encoder/readout studies precede larger-data comparisons. Validation includes
both position-weighted and opening-family-weighted metrics and a fixed training
probe. Supervised loss improvements and stronger Go play are evaluated separately.

Relevant starting points:

- [Completed value-objective repair](research/studies/strong19_value_debug/CE_RESULTS.md)
- [Longer common-AdamW comparison](research/studies/strong19_long_pair/README.md)
- [Research workflow](research/README.md)
- [Architecture and infrastructure design](DESIGN.md)
- [Expert/behavior heads and multi-step search](MULTISTEP_SEARCH.md)
- [Shared encoder and first-pass auxiliary objective](research/recipes/first_pass_aux/README.md)
- [Readout follow-up results](research/studies/readout_followups/RESULTS.md)
- [Larger fixed-data comparison](research/studies/strong9_scaling/README.md)
- [Joint policy/value preparation](research/studies/strong19_joint/README.md)
- [Joint training and exact continuation checks](research/studies/strong19_train/README.md)
- [Trained export, serving and KataGo interoperability](research/studies/strong19_eval/README.md)
- [Native features and cached search inference](research/studies/joint_inference/README.md)
- [KataGo evaluation](eval/README.md)

Some early studies explore distinct expert and opponent-behavior policies,
batched trace prediction, draft models and speculative rollout. They include
negative throughput results. The later fixed-data architecture ablations use
one policy target; their results do not establish a benefit from opponent
modeling or speculative self-play.

## Development

Python and Rust versions are pinned in `.python-version` and
`rust-toolchain.toml`. Python dependencies use `uv.lock`.

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen gozero clone research/recipes/harness_probe my_probe
uv run --frozen gozero snapshot research/recipes/my_probe \
  --config research/recipes/my_probe/smoke.json
uv run --frozen gozero verify .gozero/snapshots/<snapshot-id>
```

JAX recipes need the model dependencies; `uv sync --frozen --extra tpu` installs
the locked TPU stack. Native integration needs a compatible Rust extension.
See [Rust components](crates/README.md) and [operational tools](ops/README.md)
for build and deployment interfaces.

Copy `ops/hosts.example.json` to the ignored `ops/hosts.json` and provide your
own deployment configuration before remote work. Example addresses use the
reserved `.invalid` domain. Research configurations describe numerical
experiments; resource limits and affinity settings need configuration for the
target environment.

## Reproducibility and publication

Each experiment can freeze source, configurations and dependencies into a
content-addressed snapshot. Dataset, native binary and checkpoint identities
are recorded separately. A source snapshot is not a hermetic runtime image.

This is a sanitized source publication. Credentials, real machine identifiers,
hardware inventories, private execution logs, datasets, checkpoints and binary
source archives are excluded. Some historical scientific reports retain
references to external evidence that is not bundled. Deployment placeholders
change source hashes: create fresh snapshots and registrations for new runs;
do not treat the sanitized files as the original audit inputs. Details are in
[PUBLICATION.md](PUBLICATION.md).

Upstream attribution and component-specific licenses remain alongside the
derived code; see [third-party notices](THIRD_PARTY_NOTICES.md).

The [Muon reference study](research/studies/katago_muon/README.md) and
[surrounding schedule equations](research/studies/katago_muon_schedule/README.md)
prepare a later KataGo-informed baseline. The functional numerical port is JAX;
reference-only export tools use the pinned upstream implementation. These are
numerical checks, with neural-learner integration and full-size throughput still
to be qualified. The LR follow-up helpers preserve the complete frozen source
while changing only the learning-rate schedule scale.
