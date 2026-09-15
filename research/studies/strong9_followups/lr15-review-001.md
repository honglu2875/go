The larger9 comparison is closed and independently reaudited. Transformer
validation KL is higher on both seeds, averaging3.82% for positions and4.36%
for opening families, while learning time is about20% lower. Both its training
probe and validation continue falling; no sustained-overfit flag or late
gradient clipping appears. Group updates are nonzero. These observations
motivate an optimization probe without identifying the cause of the gap.

Test peak LR1.5e-3 versus the existing1e-3 parent, with final LR4.5e-4 versus
3e-4. Keep4096 updates,64-update warmup and ratio0.3. All other configuration
values and all1310 source files are identical within each paired seed. This
preserves the initialization, complete-game/D4 draws, evaluation populations,
architecture, parameter count and full decoding FLOPs.

Run seed1 first. Both primary endpoint KL metrics must improve by at least0.5%,
with nonregressing last-three means and no sustained-overfit flag, to confirm
with seed2. Require the same rule on the second seed before accepting the rate.
Review each result before selecting another LR or modifying the encoder. This
is a selected first probe, not an assertion of an optimal rate or Go strength.
