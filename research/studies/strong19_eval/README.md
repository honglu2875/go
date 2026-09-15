The trained policy/value export, cached inference, concurrent RPC/GTP path and
real KataGo interoperability now pass CPU checks for both model families. The
larger-9×9 queue retains the TPU. These checks use the small real-19×19 execution
models from `../strong19_train/`; four training updates provide changed, trained
weights for testing, not scientific learnability or architecture-strength evidence.

| Evidence | Result and scope |
| --- | --- |
| `artifact-cpu-003.json` | Nine portable-artifact tests: exact parameter/schema/source identities, corruption and missing arrays, detached read-back, feature-provider compatibility and candidate context binding. |
| `feature-equivalence-001.json` | All 7,992 pre-move positions in the 19-game fixture have identical offline/streamed V7 spatial bytes, globals and legal masks. Both binaries and their shared compiler objects are independently pinned. |
| `cpu-result-002.json` | Both trained exports preserve full-checkpoint outputs exactly. Cached predictions, branches and actual Rust MCTS leaves match full JAX outputs; 48 comparisons per model. |
| `rpc-unit-001.json` | Concurrent clients preserve request/state/slot identities. Wrong protocol, extra policy heads and inconsistent player-to-move values are rejected. |
| `rpc-cpu-result-001.json` | Two actual concurrent GTP clients per model reproduce independent Rust searches across reset/pass histories: 16 searches per model, 184 total RPC predictions checked against full JAX. |
| `match-cpu-result-001.json` | Four complete paired-color games against the pinned early KataGo checkpoint, 252 played moves and 264 board checks including separate terminal/reset fixtures. All terminal scores agree; 19 selected full-history predictions also match cached inference. |

`gozero.joint_artifacts` writes portable main policy/value parameters using the
checkpoint container and original parameter indices. CNN training-helper arrays
are omitted; the original complete optimizer/recovery checkpoint stays intact.
Exports bind source, model/schema, dataset feature contract and training version
without absolute deployment paths. Candidate descriptors also bind context size.

The first inference case is retained as `cpu-result-001.json`: it incorrectly
required the offline feature-export executable and streaming executable to have
the same binary hash. They have different entry points. Case 002 instead requires
the explicit, passed feature-equivalence receipt with both binary identities,
matching shared compiler objects and exact fixture outputs. No model arithmetic
or input-equivalence tolerance was weakened. Loaded-export outputs equal the
original checkpoint exactly; the largest cached/full output error is 3.58e-7.

`gozero.joint_rpc` reuses the qualified bounded connection/slot queue with a
separate protocol identity and exactly one policy plus value outputs. GTP
processes run Rust rules/search and socket glue; the owner process alone runs
JAX. The historical transport digest field contains a canonical V7 native-state
digest. In concurrent GTP testing, the largest full/cached output error is
8.35e-7; independent searches agree on chosen moves, search work, values and boards.

`eval/joint_match.py` adds the trained adapter to a paired-color real-engine
pipeline. It pins the candidate, search, native library, KataGo binary, weights,
configuration and seeds; retains transcripts/SGFs; checks every played board;
and requires matching terminal scores. Move caps remain unresolved outcomes,
never inferred draws or wins. A separate two-pass empty-board fixture checks
terminal scoring and reset without adding game outcomes. Execution-fixture
weights are rejected for a scientific benchmark purpose.

The first match case uses `kata1-b6c96-s938496-d1208807` at two visits, with two
candidate simulations excluding root evaluation. CNN games lasted 5 and 2 moves;
transformer games lasted 199 and 46. All candidates won against this very early
opponent, which is useful only as interoperability evidence. Each architecture
has just one opening pair and its conservative score interval is [0, 1]. This
does not compare architectures or establish useful 19×19 playing strength.
The longest sampled full-history prediction has 128 preceding moves; 512/1,024
audit points were specified but not reached. Total pair-controller times were
48.44 seconds for trained inference, 30.44 for concurrent GTP, and 60.85 for the
KataGo panel. They are CPU execution-check times, not TPU/search performance results.

Full-size TPU serving, measured memory/latency and distributed execution remain
open. Later scientific matches require selected full-size trained checkpoints,
a useful pinned opponent/search level, registered opening families and enough
pairs for uncertainty estimates. Scientific 19×19 training settings remain
unregistered; the larger-9×9 comparison and LR-only review still come first.
