# Fixed-student calibration diagnostic

This complete clone retains the previous board/history model and learner. `train.py` dispatches diagnostic configurations to `diagnose.py`; ordinary training configurations still execute the unmodified parent `train_student.py`. No optimizer update occurs in the diagnostic.

Both the empty-board and exact-board students use their original trained parameters and bfloat16 arithmetic. CPU qualification checks one expert and one observed-behavior validation game. The multi-host diagnostic covers the entire existing validation split, with 32 games per host batch and local inference meshes. It retains raw per-position logits, values, global shard/episode/ply identifiers, compiled HLO and model/source identities.

`analyze.py` independently reconstructs policy KL, raw illegal probability, pass probabilities, behavior NLL, outcome MSE, win Brier score and fixed-bin value calibration. Results are stratified by prospectively fixed ply ranges, prior passes, teacher shards and target pass probability. Behavioral observations do not become expert or value targets. The full run must reproduce all original validation aggregates within a fixed absolute tolerance of 0.0002; this is a numerical check, not a new scientific success criterion.

High outcome MSE can coexist with calibrated uncertainty, as the metric tests explicitly demonstrate. Phase analysis is descriptive on previously inspected validation games. Neither positions nor shards are independent training seeds, and these results cannot establish stronger play or RL sample efficiency.

The frozen configuration bounds compute and raw archive sizes. Source, dataset manifests, every selected episode, checkpoint lineage and all raw output hashes are independently checked. A changed feature layout or learning intervention needs its own clone and registration.
