# Research defaults

`strong19.json` records the selected defaults for new fixed-data 19x19 research:
AdamW peak LR 1e-3 and 64 complete games per update. The linked cloneable dense
configuration includes the complete model, optimizer and exposure-based
schedule. MoE remains an ablation until its comparison passes.

Changing these defaults does not rewrite historical registered experiments or
their schedules. A new experiment must freeze the resolved configuration and
explicitly register any overrides.
