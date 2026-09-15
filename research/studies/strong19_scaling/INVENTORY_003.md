The third completed-game inventory contains 2,064 terminal games and 856,540
positions, with no move-capped games. Every record's complete file hash was
read, while test policy/value arrays stayed closed. Generation continues;
this observation is not the final scientific training population.

| Split | Games | Positions | Opening families | Largest family, games |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,858 | 770,234 | 1,733 | 7 |
| Validation | 103 | 42,599 | 97 | 2 |
| Test metadata only | 103 | 43,707 | 100 | 2 |

Opening families are assigned consistently to one split. The largest family
accounts for less than 0.4% of training games; the concentration found in the
earlier 9×9 validation pool is not present in this current metadata inventory.
Trajectory-level duplicate auditing remains a separate requirement.

All eight opponent strata have completed training and validation examples.
Training strata range from 192 to 315 games and 68,569 to 109,644 positions.
Validation strata currently contain only 6–25 games; teacher self-play has
seven validation games. This is too little for a precise per-stratum model
ranking. Continue collecting before fixing the scale-up population or horizon.

The longest retained game has 645 positions, which fits the previously
qualified 1,536-position execution bucket. Full-population feature preparation,
storage reservation, split/trajectory auditing and the scientific batch and
optimizer conventions must still be fixed before training. The inventory is
[inventory-003.json](inventory-003.json); it does not supersede the separate
feature-loader execution qualification.
