# Single-token attention pooling: matched 19x19 stage

| Arm | Parameters | Policy KL | Value MSE | Learning hours |
| --- | ---: | ---: | ---: | ---: |
| flat | 232,011,540 | 0.676803 | 0.111533 | 4.761 |
| attention | 232,023,236 | 0.677231 | 0.106376 | 4.785 |

One paired seed, fixed data and targets; fixed 256-update endpoint of a 512-update schedule. No playing-strength or final-512 claim.

All-rank state, optimizer, sampler, D4 draws and fixed evaluation populations were audited. Each endpoint has a hash-verified disk peer copy including every rank state.

Endpoint policy KL relative gain: -0.06%; last-three mean gain: -0.02%.

Use a second seed and a longer continuation before treating a small difference as established. Review the train/validation curves and all overfit flags before extending.
