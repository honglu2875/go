# Exact native observation suffix encoding

The Rust replay ABI applies every historical action and checks exact positional
superko, but encodes only the observations requested after a retained prefix.
The model, history semantics and MCTS targets remain unchanged.

The full paired scorer passed its registered gate: 48,323 MCTS leaf predictions
and 289,938 returned heads matched bitwise, while critical median scorer time
improved 1.6501x. Encoded observation rows fell from 6,239,769 to 107,029.
The measurement includes replay, packing, dispatch, retained-cache completion and
head return; it excludes shared MCTS advancement, root prefill and external
queueing. It is not an end-to-end MFU or strength claim.

The suffix service subsequently passed multi-host full-model inference and GTP
qualification. It was used in the completed online self-play generation and
fresh real-KataGo panels. Complete native/library/source and candidate closures
stage automatically before distributed startup. See the native suffix and
suffix service reports in `research/studies/visual_causal`.
