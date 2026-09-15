This is a runtime-only clone of the registered causal transformer. It is not
an encoder or learnability intervention. The original decoding function is
retained as append_move_reference. The new function carries the KV arrays
through the layer scan and updates only new token slots, avoiding stacked
full-cache scan outputs. Parameters, observation/action encoding, policy
head and dense attention arithmetic are unchanged.

CPU checks cover full-history and reference equivalence, ragged/inactive
rows, malformed pointers and exact rejection without changing cache bytes.
A TPU comparison must observe donated cache outputs and report alias/temp
memory as well as latency. The original ~82 ms result remains part of the
registered learning comparison. No speedup is claimed until measured.
