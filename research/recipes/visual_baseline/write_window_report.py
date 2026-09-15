"""Summarize audited artifacts without turning incomplete probes into results."""
import argparse,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify,read_json,canonical_json
from gozero.checkpoints import sha256
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=Path,required=True);a=p.parse_args();verify(SOURCE)
root=SOURCE.parents[2];study=root/'research/studies/visual_causal'
if a.output.exists() or a.index.exists():raise FileExistsError('Report/index already exists')
learning=read_json(study/'cnn_learning_result.json');matches=read_json(study/'cnn_katago_result.json')
deferred=read_json(study/'deferred_result.json')
ledger=read_json(root/'research/studies/runtime_qualification/reservation_ledger.json')
if learning['status']!='passed' or matches['audit_status']!='passed' or deferred['status']!='passed':raise ValueError('Unqualified reports')
if ledger['observed_open_attempts']:raise ValueError('Pod attempts remain open')
window_start=datetime(2026,9,12,1,39,55,tzinfo=timezone.utc).timestamp()
window_end=window_start+8*3600
attempt_intervals=[]
for item in ledger['recorded_pod_attempts']:
 path=root/'runs'/item['attempt']/'result.json'
 if sha256(path)!=item['result_sha256']:raise ValueError('Ledger attempt changed')
 r=read_json(path)
 if r['end_unix_time']>window_start and r['start_unix_time']<window_end:
  attempt_intervals.append((max(window_start,r['start_unix_time']),min(window_end,r['end_unix_time']),r['end_unix_time']))
attempt_intervals.sort()
if any(b[0]<a[1] for a,b in zip(attempt_intervals,attempt_intervals[1:])):raise ValueError('Overlapping full-pod attempts')
window_attempt_hours=sum(b-a for a,b,_ in attempt_intervals)*16/3600
last_attempt_end=max(c for _,_,c in attempt_intervals)
lines=['# Eight-hour Go research report — 12 September 2026','',
'The large CNN baseline is now implemented, trained and compared with the new causal visual transformer. In the short matched-update screen, the CNN fits the held-out weak-teacher corpus better. The real KataGo panel below is a separate measurement. No strong 19x19 engine, equal-compute architecture winner, hardware MFU improvement or faster-than-KataGo RL result has been established.','',
'## Large architecture baseline','',
'Both pure-JAX models trained from scratch for 128 updates on identical complete-episode and D4 draws, with 128 sequences across the 16-chip/four-host pod. Expert policy, observed behavior and value objectives were shared; auxiliary draft losses were disabled. The CNN has 232,389,632 parameters and 49 width 512 residual blocks. The transformer has 233,137,152 parameters and 18 width 1024 causal decoder layers. The CNN sees eight recent observations and preceding-action planes; the transformer sees the full causal visual/action history. This is a practical baseline with an explicit history difference.','',
'The immutable 9x9 corpus contains 11,871 expert and 8,192 behavior episodes from four earlier native-search self-play teacher runs. The expert head learns stored MCTS distributions and the value head learns terminal outcomes; the behavior head predicts actual archived moves, including capped games without outcome supervision. These are observed engine behaviors, not scripted opponents. No KataGo data entered training. The 80/10/10 splits hold out entire games, although board positions can repeat across games and no unseen-opponent claim is made.','',
'Each update samples 64 expert and 64 behavior sequences, with identical whole-game rotations/reflections in both arms. Both runs saw exactly 769,036 expert-position and 790,401 behavior-position exposures. AdamW uses betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01 and gradient clipping at 1.0; the learning rate warms up for 32 steps to 1e-4, then decays by cosine to 3e-5. Expert cross-entropy, observed-action cross-entropy and terminal value MSE each have weight 1. Score, ownership and draft losses are disabled.','',
'The transformer is trained with exact observations and past actions supplied throughout the complete sequence, using causal masking and parallel losses at each move. It does not generate its training contexts or learn a future-board reconstruction loss. The CNN evaluates the corresponding eight-observation window at every position. Both behavior losses update their shared trunks.','',
'| Held-out test/cost | CNN | Transformer |','| --- | ---: | ---: |']
for label,key in [('Expert KL','expert_kl'),('Behavior cross-entropy','behavior_ce'),('Value MSE','value_mse')]:
 vals=[learning['arms'][x]['test']['metrics'][key] for x in ('cnn','transformer')];lines.append(f'| {label} | {vals[0]:.5f} | {vals[1]:.5f} |')
for label,key in [('Critical-rank learning seconds','critical_learning_seconds'),('Full attempt seconds','attempt_seconds'),('Full attempt chip-hours','attempt_chip_hours')]:
 vals=[learning['arms'][x][key] for x in ('cnn','transformer')];lines.append(f'| {label} | {vals[0]:.3f} | {vals[1]:.3f} |')
lines+=['','All 1,024 rank-update draws and both complete parameter/Adam checkpoints passed independent audit. The CNN took 2.20x the learning time. Initialization, inductive biases and the inherited shape-based decay mask also differ; one seed and 128 updates cannot settle long-run architecture performance. [Training audit](cnn_learning_result.json), [recipe and limitations](../../recipes/visual_baseline/README.md).','',
'Both inference owners passed full-model native search/GTP qualification before external matches: 17,408 leaf comparisons per arm and 32 real GTP probe boards per arm. Floating-point tolerance is explicit; this is not a bitwise equivalence claim. [Service audit](cnn_service_result.json).','',
'## Real KataGo panel','',
'Each arm played 128 games: 32 paired-color openings against each of two fixed historical checkpoints. Candidate search used 16 simulations excluding its root; KataGo used one visit and one thread. These are explicit search budgets, not equal compute. Counts are wins/losses/unresolved move caps.','',
'| KataGo checkpoint | CNN | Transformer |','| --- | ---: | ---: |']
anchors=list(matches['arms']['cnn']['anchors'])
for anchor in anchors:
 vals=[]
 for name in ('cnn','transformer'):
  r=matches['arms'][name]['anchors'][anchor];unfinished=r['scheduled_games']-r['status_counts'].get('completed',0)
  vals.append(f"{r['completed_wins']} / {r['completed_losses']} / {unfinished}")
 lines.append(f'| `{anchor}` | {vals[0]} | {vals[1]} |')
boards=sum(v['checked_positions'] for arm in matches['arms'].values() for v in arm['anchors'].values())
completed=sum(v['status_counts'].get('completed',0) for arm in matches['arms'].values() for v in arm['anchors'].values())
lines += ['',f'Raw GTP commands/replies, SGFs, all {boards:,} recorded boards and {completed} completed scores passed independent audit. Caps are retained without assigned outcomes; the complete-panel strength qualification therefore fails whenever any cap remains. The observed score bounds still favor the CNN at both anchors even if all unresolved outcomes are assigned in the transformer\'s favor. These are sample bounds, not confidence intervals. The models faced common opponents; no direct CNN-versus-transformer games were played. No Elo or general architecture superiority is inferred. [Full panel and missing-outcome bounds](cnn_katago_result.json).','',
'## Rollout and learning findings','',
'- Fixed one-position scoring blocks plus continuously refilled native actor slots produced exactly the same 1,024 game traces, seven target arrays and game records as the blocked control. The generation segment fell from 716.919 to 474.205 seconds: 1.5118x. This is one paired timing result, not a hardware MFU or strength claim. [Audit](stable_refill_result.json).',
'- Native suffix observation encoding passed its separate gate at 1.6501x scorer speed, with 48,323 leaf predictions and 289,938 head outputs bitwise equal. Timing scopes differ; do not multiply speedups. [Audit](native_suffix_result.json).',
'- Dynamic-block refill was fast but changed 230 of 1,024 action tapes and failed the exactness gate. Complete eager-repair speculative loops were slower than sequential policy decoding: 0.726x at H2 and 0.724x at H4. D4 training improved draft agreement but did not make the independent-root speculative screen faster. [Refill failure](refill_result.json), [loop](loop_result.json), [D4 speculation](speculation_d4_result.json).',
f"- The final deferred-repair loop also failed its timing screen: " + ', '.join(f"{r['median_speedup']:.3f}x at H{r['horizon']}" for r in deferred['conditions'] if r['horizon']) + f". Its audit replayed {deferred['native_committed_moves_checked']:,} committed moves and {deferred['terminal_paths']} terminal paths; maximum reported full-policy TV was {deferred['maximum_reported_full_policy_tv']:.6f}. Pending work and final deep cache drain are timed; initial prefill and compilation are excluded. This is bounded policy sampling without MCTS, and individual acceptance probabilities/uniforms were not retained for independent replay. [Audit](deferred_result.json).",
'- The first full-Adam online continuation did not improve its fresh KataGo panel. Reconstructed expert-data reuse was 18.66 exposures per available position for new self-play versus 1.86 for old data; this motivates a controlled replay-ratio experiment but does not explain the decline causally. [Online games](online_katago_result.json), [reuse audit](online_reuse_result.json).','',
'## Infrastructure and next work','',
'Clone-owned model/training recipes live in research/recipes; shared Rust rules, native bindings, datasets, artifacts and checkpoints live outside those recipes. Cargo/uv locks, content-addressed source/config snapshots, rank checkpoint groups and retained failures support reproduction. Both large baseline checkpoints include full Adam moments.','',
'Host 1 ran out of disk space while staging the transformer checkpoint. The unchanged service passed after verified lossless archival of 20 older arrays restored about 5.35 GB free. The restore/reader round trip used the user-provided /dev/shm scratch; committed checkpoints and archives remain on persistent disk. The failed staging attempt and repair receipts are preserved. [Repair receipt](../../../runs/maintenance/host1-archive-repair-result.json).','',
f'The nominal eight-hour allocation (01:39:55–09:39:55 UTC) corresponds to 128 chip-hours on 16 chips. The {len(attempt_intervals)} recorded, nonoverlapping pod attempts intersecting that window account for {window_attempt_hours:.3f} chip-hours inside it, including startup, compilation, evaluation and checkpointing. The final registered TPU probe ended at {datetime.fromtimestamp(last_attempt_end,timezone.utc).strftime("%H:%M:%S UTC")}, just after the nominal window; full attempt costs and subsequent allocation time remain in the ledger. It has {len(ledger["recorded_pod_attempts"])} closed attempts and no observed open attempts at publication, and also covers work before this window. Full teacher/data-generation costs matter when comparing learning systems. [Ledger](../runtime_qualification/reservation_ledger.json).','',
'The final user-requested CNN comparison took priority over optional pending work. The separately registered collection-holdout model diagnostic and second fixed-block refill timing pair remain unrun. They are not negative results.','',
'Keep the CNN as a serious control. Next test spatial policy readouts and small CNN observation encoders inside the causal transformer, a matched-history control, longer common schedules, and equal-compute learning curves. Retain the distinct expert/behavior objectives; their log-density ratio is not automatically a value advantage. Production training still needs external checkpoint durability, host-loss/mid-update recovery and a sustained generation/learner coordinator. [Detailed design](NEXT.md).','']
a.output.write_text('\n'.join(lines))
artifacts={}
for name in ['cnn_baseline_registration.json','cnn_learning_result.json','cnn_qualification.json','cnn_service_result.json','cnn_katago_design.json','cnn_katago_registration.json','cnn_katago_result.json','cnn_transformer_service_retry.json','stable_refill_registration.json','stable_refill_result.json','refill_result.json','native_suffix_result.json','suffix_service_result.json','learning_pilot_result.json','augmentation_result.json','augmented_katago_result.json','online_learning_result.json','online_katago_result.json','online_reuse_result.json','loop_result.json','speculation_d4_result.json','deferred_registration.json','collection_registration.json','deferred_tpu_qualification.json','deferred_result.json','collection_result.json']:
 path=study/name
 if path.is_file():artifacts[str(path.relative_to(root))]=sha256(path)
for path in [a.output,study/'README.md',study/'NEXT.md',root/'research/recipes/visual_baseline/README.md',root/'.gozero/datasets/causal-5c5f2c0d/manifest.json',root/'.gozero/datasets/visual-causal-ca730950/manifest.json',root/'eval/visual_causal/baseline_cnn.json',root/'eval/visual_causal/baseline_transformer.json',root/'runs/maintenance/host1-archive-repair-result.json',root/'research/studies/runtime_qualification/reservation_ledger.json']:
 artifacts[str(path.resolve().relative_to(root))]=sha256(path)
a.index.write_bytes(canonical_json({'schema_version':1,'kind':'visual_eight_hour_artifact_index','operator_snapshot':SOURCE.name,'created_utc':datetime.now(timezone.utc).isoformat(),'artifacts':artifacts}))
print(json.dumps({'report':str(a.output),'report_sha256':sha256(a.output),'index':str(a.index),'index_sha256':sha256(a.index)}))
