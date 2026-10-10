# Research defaults

`strong19.json` records the selected defaults for new fixed-data 19x19 research:
G8 temporal fuzzy TopK, AdamW peak LR 0.0015 and 64 complete games per update.
The [cloneable default configuration](../recipes/strong19_fuzzy_topk/g8_default.json)
is an exact copy of the selected, audited sweep configuration. It has 298,257,732
parameters, width 768 and 18 temporal layers, with 683 groups of 8 candidate
features per temporal FFN. The encoder and policy/value objectives are unchanged.

The exposure-based cosine schedule uses 1,024 updates, 80 warmup updates,
terminal LR 0.00045, and validation every 32 updates. The evidence is a
256-update prefix of this schedule; full-horizon convergence remains untested.
The user selected G8 on October 10 after reviewing the completed sweep and MoE
comparison. See the [promotion rationale](../studies/strong19_fuzzy_topk/PROMOTION.md)
and [selection record](promotions/20261010-g8.json).

[The previous dense default](strong19_dense_20261006.json) is preserved verbatim.
Dense and conventional temporal MoE remain comparison baselines. G8 matches
logical active decoding FLOPs while allowing more total parameters; its current
implementation does not establish an inference-speed or MFU improvement.

Changing these defaults does not rewrite historical registered experiments or
their schedules. A new experiment must freeze the resolved configuration and
explicitly register any overrides.
