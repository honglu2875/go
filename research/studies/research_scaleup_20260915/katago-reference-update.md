KataGo's current `tf3-b11c768` networks are nested-bottleneck transformers,
with roughly 70.4M exported parameters. Our fixed 19×19 teacher is one of them.
The existing 232M CNN comparison uses the established `b40c768nbt` family.
Keep that controlled comparison, and distinguish it from current KataGo strength
benchmarks. A CNN learnability win alone does not establish a new state of the art.

Primary reference checked 2026-09-15:
[KataGo architecture guide](https://github.com/lightvector/KataGo/blob/master/docs/NetworkArchitectures.md).
The pinned local training source is commit
`92ee95c0a4b25fec214da00951ab69e97e207729`; its `modelconfigs.py` independently
identifies the CNN and transformer block types.
