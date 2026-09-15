Five full 1024-update training runs finished during the overnight session. None of these full learning runs was interrupted. The next variant was interrupted during its short qualification. No pod experiment is currently running.

All completed comparisons use the same 11,469,333 training-position exposures and 102,339 validation positions. The test set remained closed.

| Completed configuration | Peak LR | Final validation KL | Learning minutes | Decode ms |
|---|---:|---:|---:|---:|
| C64, two convolutions | 3e-4 | 0.540367 | 43.40 | 81.99 |
| C128, two convolutions | 3e-4 | 0.521731 | 43.62 | 81.96 |
| C128 with pooled readout | 3e-4 | 0.544788 | 43.63 | 81.86 |
| C768, six convolutions; 30-layer decoder | 3e-4 | 0.511634 | 40.44 | 72.58 |
| CNN reference | 1e-3 | 0.420361 | 31.98 | 16.38 |

The C64/C128 width sweep is complete at all three requested learning rates:

| Encoder | 1e-4 KL | 3e-4 KL | 6e-4 KL |
|---|---:|---:|---:|
| C64 | 0.640906 | 0.540367 | 0.644910 |
| C128 | 0.628599 | 0.521731 | 0.651201 |

The larger encoder improves final KL by 1.94% over C128, reduces learning time by 7.29%, and reduces cached decoding latency by 11.44%. It uses six stride-1, SAME-padded 3×3 convolutions with 768 channels, 30 causal decoder layers of width768 and FFN2256, and about232.55M parameters. Complete encoder-plus-decoder inference arithmetic remains within1% of the CNN reference at9×9, batch128 and128 past moves. The gain is small and appears late; the CNN still has lower loss. Pooling current-board tokens into the global readout did not help at the matched learning rate.

The pointwise-correction variant adds a shared768→1 projection and bias from encoder features into board-move logits, with the same single policy objective. Its CPU checks and cached-decode agreement checks completed. All four workers were cancelled at approximately04:12UTC while compiling evaluation, before the first learning update. Full qualification, its training run and the prepared learning-rate follow-ups remain unfinished. The rank logs were collected and the stale coordinator record was closed explicitly as interrupted; the original failure evidence is preserved.

The pooled-readout and large-encoder checkpoint arrays are now missing from /dev/shm on all configured hosts. They had been verified at run completion but were still temporary; final archival had not happened. Their frozen source, metrics, receipts and successful historical audits remain. This prevents resuming or benchmarking those exact weights. The CNN checkpoint and all three C128 persistent archives have been hash-verified after the interruption. Deferring durable writes until the end of the session was a storage mistake.

Remaining work: persist checkpoints immediately at each run boundary; reproduce the large-encoder weights if needed; complete the pointwise variant’s qualification and full3e-4 ablation; then choose further LR or encoder experiments from that result. A future uninterrupted run should keep job supervision durable while retaining explicit cancellation and runtime limits.

These are single-seed supervised learnability comparisons on a fixed weak-teacher dataset, not a demonstration of Go strength or faster reinforcement learning.

Completed learning curves (external or omitted experiment artifact) · Machine-readable handoff (external or omitted experiment artifact) · Interruption evidence (external or omitted experiment artifact)
