# Conditional padding runtime diagnostic

This clone changes only the execution path of the completed 19×19 transformer.
`diagnose.py` runs the original dense encoder, conditional dense encoder, and
conditional skip-padding encoder with identical parameters, optimizer and loss.
The latter two reuse one compiled executable; `encoder_counts` changes only
which encoder chunks execute. Actual causal-history counts and supervision are
unchanged. Full state remains in host RAM and only small diagnostic records are
written. The previous throughput screen failed its state tolerance and remains
failed; passing this control is not a production qualification.

`qualify_control.py` requires exact CPU equivalence for full two-step AdamW state
and metrics, in FP32 and BF16 on four devices including an empty shard.
