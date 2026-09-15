# Value rescaling changes decisions; the fresh strength screen failed

Disabling Gumbel completed-value rescaling changed many late-game decisions for the fixed exact-board student. A separately registered 80-game real-KataGo ablation then failed both primary criteria. The native search defaults remain unchanged, and this study does not promote a stronger model or a production recipe.

| Fresh evaluation | Rescaling on | Rescaling off |
| --- | --- | --- |
| Historical b6c96, one visit | 2 wins, 30 losses | 5 wins, 25 losses, 2 caps |
| Scheduled historical score | 6.25% | 15.625–21.875% |
| Historical completion | 100% | 93.75% |
| Stronger b18c384, one visit | 0 wins, 4 losses | 0 wins, 4 losses |
| Stronger b18c384, 16 visits | 0 wins, 4 losses | 0 wins, 4 losses |

The historical panel uses 16 opening units, with both colors and both arms paired. Caps remain unknown `[0,1]`. The off-minus-on scheduled score is bounded by +9.375 to +15.625 percentage points, but the registered 20,000-draw paired bootstrap outer interval is **−3.125 to +34.375 points**. Its lower endpoint is not positive, and off completion misses the 95% threshold. These are descriptive intervals conditional on this fixed weak model and these openings, not independent-training-seed evidence or Elo estimates. Historical b6c96 and the pinned 2023 b18 checkpoint are not current best KataGo.

All 80 scheduled games finished their bounded execution in 485.40 seconds: 78 completed, two capped, no process failures. Independent analysis verified loaded model/native identities, the recursive input closure, paired search configurations and external seeds, all GTP moves/search responses, **6,768 boards and 78 completed scores** against real KataGo and native replay. The audit hashes all 592 retained raw files. Both arms use the same 939,968-parameter student, original bfloat16 inference, 16 Gumbel simulations, value scale 0.1, maxvisit initialization 50, rules, komi and move cap; only `rescale_values` differs. The evaluator is CPU-only and trains no model.

The preceding diagnostic replays all **899 student turns after opponent passes**, plus 128 fixed hash-selected other turns, from 72 previously retained real-KataGo games. Corpus preparation verifies all 8,472 original boards and 65 completed scores. The native inspection API exposes completed-root priors, visits, unrounded value sums and terminal child values without changing search. Its qualification passed 22 Python binding, 22 Rust actor and 12 Rust search tests, including terminal-pass backup perspective and unchanged search finish.

The 12-position qualification and full 1,027-position diagnostic reproduced every original student move, root search value and neural/terminal/simulation count. Independent reconstruction of root policies from raw search evidence was bit-exact across 66,288 retained neural leaf rows. The full diagnostic took 75.87 seconds on four CPU worker groups. It compares the student and the explicitly identified seed-27 CNN component of its four-model teacher mixture, each with rescaling enabled/disabled. No KataGo labels are used for training.

Across the 899 after-pass positions, student passes increased **25→79**: 62 play-to-pass changes and eight pass-to-play changes, among 455 total action changes. Applying the unscaled transformation to the same scaled-search evidence changes 322 actions and selects 78 passes, isolating a direct transformation effect in addition to changed exploration. Median student completed-child value range was 0.000601, median root network value −0.999649 and median pass prior 0.00450. Teacher passes increased 1→12; its median pass prior was 0.00146. On 128 sampled other turns, student choices changed 39 times and passes increased 7→12.

Native counterfactual passing loses in 898 of the 899 after-pass positions. Both models and settings pass in the sole winning-pass position. These counterfactual scores use the native pass-alive scorer, not new KataGo adjudication. All net extra passes concede native losses, and 820 student positions still select play with rescaling disabled. In fresh historical games, both settings answered an opponent pass with a pass 14 times, but opportunities differed (104 on, 217 off), including the off arm's two caps. The retained-position pass increase did not establish a completion improvement in fresh play. This evidence shows a contribution from rescaling, without establishing a terminal-backup bug or stronger play.

Reproduction identities are collected in artifacts.json (external or omitted experiment artifact). The complete clone is [endgame_search](../../recipes/endgame_search/README.md). Main artifacts:

- Native inspection source `1bc92cbd36c8de3afe5e64987f9b08a2c4ed29320de3395c46c96a85442b6441`; binary SHA256 `6bf12686b663316658d323eae90a056f6ad18928f862fc5946d0ea12476ed8af`.
- Corpus `runs/endgame-corpus-339f0f30/corpus.json`; SHA256 `ab673d3e36ecb18842b632830f094c990224e188fe7026a0fad03d0e95720bba`.
- Full diagnostic registration [diagnostic_spec.json](diagnostic_spec.json); SHA256 `d71099cfb9ca13db39d93824b2901646349467d378b908296147b9746c15615e`. Audit result.json (external or omitted experiment artifact), SHA256 `60fcef4eb91a866258a34b0a070b7a843f8f64657d4f2d6bf98fe1e13e766989`.
- Fresh games registration [katago_spec.json](katago_spec.json); SHA256 `ac2bc5c894692be9c9b38c4f691a96f9c6a6378bfe52840f7a7c6aa9a21ba8fe`. Execution source `e037beb7be40aa46e6461e2762f9cbc0e219f3c561cfc11f12b8c74c11fbc1a8`, output `runs/eval/endgame-rescale-e037beb7`.
- Fresh games audit katago_result.json (external or omitted experiment artifact); SHA256 `9483608a9ec89456f85d2badc412a8c679c219545688ee9d0c315f36a66c3341`, analysis source `ad223b589aa206074724b73c6b34452b9c7cc3d0db4d614e26865a8efccd522b`.

Implementation failures remain retained. Preparation `3b4ac91d…` incorrectly counted duplicate final-board displays; `6c1f4dfa…` incorrectly assumed a final row in the bulk pre-action API. Both stopped before searches. A caller protocol-hash typo was rejected before workers (`runs/endgame-rejected-339f0f30`). The first registered qualification (`runs/endgame-qualification-339f0f30`) rejected the teacher's explicit score-utility object despite its zero factor. The corrected source preserves that outcome-only setting; a new qualification registration links the failed attempt. No diagnostic search or fresh-game outcome was discarded or rerun. Global `cargo fmt --check` also encountered existing repository formatting differences; only the changed Rust files were formatted, and no repository-wide formatting success is claimed.

There were no new TPU attempts in this study. Reservation time still includes CPU research and idle TPU allocation; it remains separately accounted in the reservation ledger. See [NEXT.md](NEXT.md) for the separated learning, systems and production follow-ups.
