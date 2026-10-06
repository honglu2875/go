# Temporal MoE at tuned research defaults

Cloneable pure-JAX trainer with the same visual encoder, causal transformer,
contextual spatial readout, signed-value objective and exact complete-game
replay as the tuned dense control. `dense.json` is the default configuration;
`temporal.json` enables four top-two temporal experts and `balance_low.json`
changes only their load-balancing coefficient. Models use no batch norm.

The default peak LR is 1e-3 with 64 global games per update. Scientific
configuration and data identities are resolved before freezing the recipe.
See [the protocol](../../studies/strong19_moe_batch64/PROTOCOL.md). New temporary
trial arrays use guarded RAM; selected states receive two durable disk copies.
All existing checkpoint files remain intact.
