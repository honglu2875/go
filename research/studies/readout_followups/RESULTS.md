All three preregistered readout screens are closed. The existing bilinear rank-64
readout with first-pass auxiliary supervision remains selected.

| Architecture | Final validation KL | Change from paired parent |
|---|---:|---:|
| Bilinear rank 64 parent | 0.39701045 | +0.00% |
| rank128 | 0.40705645 | +2.53% |
| additive | 0.41474593 | +4.47% |
| refinement | 0.40341508 | +1.61% |

Positive percentages mean worse KL. Each changed architecture used the exact same
11,469,333 position/symmetry exposures as its paired parent. All full checkpoints,
all-rank metrics, common initialized weights and evaluation populations passed
auditing. The registered first-seed improvement threshold was 0.5%; none qualified
for a second seed. Negative screens are retained rather than tuned after seeing
the endpoint. This does not establish that these changes fail on other data or
training schedules. All validation and fixed training-probe curves kept falling;
none triggered the registered sustained-overfit diagnostic.

The selected model has 231,181,121 parameters, width 768, one board token, a
24-block spatial encoder applied twice with shared weights, and 18 causal
transformer layers. The policy reads the retained spatial grid through a local
term and a rank-64 temporal/spatial bilinear term. Its complete cached decode
cost is 37.4281 GFLOPs per move at 128 previous moves, including the encoder;
the 232,431,872-parameter CNN control costs 37.3483 GFLOPs. These are logical
multiply-add counts, not achieved MFU or evidence of Go strength.

The next study is `../strong9_scaling/registration-001.json`: the selected
transformer and CNN, two paired seeds, a fixed 4,096-update horizon and regular
validation on the finalized strong-teacher corpus. Test targets remain closed.
