# Refilling native visual self-play slots

This clone owns the complete visual generator and model-facing code. It assigns
independent episode seeds by logical game ID and can reuse a physical cache slot
as soon as its game finishes. Both arms checkpoint only after the same complete
1,024-game group, keeping checkpoint frequency out of the comparison.

CPU and multi-host controlled boundary recovery passed, including heterogeneous
fixture lengths. The first full dynamic-block comparison reduced the critical
generation segment from 608.08 to 380.16 seconds (1.60x), but changed 230 of 1,024
action tapes and failed its exactness gate. Every generated legality mask and
terminal outcome still passed native replay. Retain both immutable attempts.

The registered followup sets `max_block=1` in both arms. This fixes the neural
computation shape independently of peer histories. The CPU and full TPU arms
matched all target arrays and normalized episode records. The TPU audit checked
all 1,024 identical game traces and 99,118 moves; the generation segment fell
from 716.919 to 474.205 seconds (1.5118x), passing the registered gate. This is one
paired timing result; a separately registered second pair remains unrun. It does
not establish hardware MFU or Go strength. See
`research/studies/visual_causal/refill_result.json` and `stable_refill_result.json`.
