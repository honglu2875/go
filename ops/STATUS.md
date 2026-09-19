This public source backup omits live deployment state, host inventories and
private process records. Configure deployment bindings before running the tools.

The completed 108-update objective repair is summarized in
`research/studies/strong19_value_debug/CE_RESULTS.md`: transformer value MSE
improved from 0.542233 to 0.150406, with policy KL 0.935420. The source-optimizer
CNN remains ahead in policy KL at 0.801642. These are supervised comparisons.

The next comparison is registered in `research/studies/strong19_long_pair/`.
Both architectures use AdamW and the repaired value objective, the same frozen
data and identical 512-update sampling draws. Full-size CNN execution
qualification passed; the longer comparison is in progress. Final paired
results and trained-model KataGo comparisons remain pending.

See `PUBLICATION.md` for source-backup scope and reproducibility limits.
