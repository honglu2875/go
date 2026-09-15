This control removes the training helper from the frozen KataGo-derived CNN
at snapshot `4eff9303401f25868248047c26418d1fc5d2390f9b6f7794242802b130fc7a89`.
The main policy is trained directly with coefficient 1. The historical
objective was 0.2 main-policy CE + 0.8 helper CE. The helper applies batch
statistics to the final trunk; it is not an intermediate-depth exit.

`katago.py`, the trainer, optimizer, initialization and full learning config
are unchanged. The only numerical intervention is `policy_model.losses`.
Its executed forward path uses no batch statistics. Keep the original
parameter tree to verify exact initial array identity: 232,431,872 stored
parameter slots, 232,134,784 active parameters, and 297,088 unused helper
slots. The helper receives zero loss gradient; the unchanged AdamW may
decay unused weight slots. It cannot affect the main output. The 9x9 CNN
inference graph and 37.348255744 GFLOP/move dense arithmetic are unchanged.

`main_full_qualification.json` executes four updates at the full model and
global batch of 128 games, covering all three history buckets on 16 TPU
chips. `main_1024.json` repeats the historical seed, 1,024 updates, data
draws, D4 augmentation, validation population and AdamW LR 1e-3 schedule.
Inherited configs/scripts are reference material, not queued experiments.

The CPU tests check global gradients and optimizer rejection, role-based
decay/RepVGG transforms, position-weighted metrics, and the main-only
objective's zero helper gradients and independence from other batch items.
`qualify_main.py` traces complete inference and differentiated training;
`audit_qualification.py` checks the full TPU attempt and durable checkpoint
before full learning. `audit_learning.py` reconstructs every actual draw and
checks all replicas and checkpoint bytes. Compare the result with the
unchanged historical CNN, retaining both results. This intervention combines
removing the helper loss with removing its batch-statistics feature path;
it does not separately identify those two effects.

This is fixed-data learnability on the same weak native-MCTS teacher corpus.
It does not measure playing strength or achieved MFU. Freeze this entire
recipe and the selected config before execution; never train editable code.
