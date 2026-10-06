"""Report the prospectively matched batch-64 dense/MoE comparison."""
import csv, json, math
from execute_run import ROOT, STUDY, read, sha, require, publish
METRICS=('expert_kl','family_kl','value_mse','value_family_mse','expert_top1')


def gates(row,base):
    return dict(endpoint_policy=all(row['endpoint'][k]<=.995*base['endpoint'][k] for k in ('expert_kl','family_kl')),
        tail_policy=all(row['tail'][k]<=base['tail'][k] for k in ('expert_kl','family_kl')),
        value=all(row[s]['value_mse']<=1.05*base[s]['value_mse'] for s in ('endpoint','tail')),
        no_overfit=not row['sustained_overfit'],tokens=row['all_tokens_retained'],
        time=row['learning_seconds']<=1.2*base['learning_seconds'])


def select(table,control='dense'):
    choices=[k for k in table if k!=control and all(gates(table[k],table[control]).values())]
    return min(choices,key=lambda k:table[k]['endpoint']['expert_kl']) if choices else control


def report(registration,records):
    audits={};logs={};table={};histories={}
    for arm,record in records.items():
        path=ROOT/record['audit'];require(sha(path)==record['audit_sha256'],'Audit identity changed')
        a=read(path);audits[arm]=a
        require(a['status']=='passed' and a['all_rank_metrics_and_saved_state_verified'] and a['steps']==256
            and a['positions']==7001181 and a['dataset_manifest_sha256']==registration['dataset_manifest_sha256'],'Incomplete matched audit')
        histories[arm]={}
        for field,population in (('validation_history',registration['validation_population_sha256']),('training_probe_history',registration['probe_population_sha256'])):
            require(all(r['episode_ids_sha256']==population for r in a[field]),'Evaluation population changed')
            histories[arm][field]={r['turn']:r['metrics'] for r in a[field]}
        rows=[json.loads(x) for x in (ROOT/'runs'/record['attempt']/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()]
        require([r['turn'] for r in rows]==list(range(1,257)),'Incomplete fresh trajectory');logs[arm]=rows
        val=histories[arm]['validation_history'];probe=histories[arm]['training_probe_history']
        router=[k for k in rows[0] if k.startswith('moe_')]
        table[arm]=dict(parameters=a['parameters'],position_exposures=a['positions'],updates=256,batch_games=64,peak_lr=.001,
            endpoint={k:val[256][k] for k in METRICS},tail={k:sum(val[t][k] for t in (192,224,256))/3 for k in METRICS},
            training_probe={k:probe[256][k] for k in METRICS},learning_seconds=rows[-1]['cumulative_learning_seconds'],
            sustained_overfit=any(r['sustained'] for r in a['overfit_observations']),
            all_tokens_retained=all(r.get('moe_dropped_tokens',0)==0 for r in rows),
            router_endpoint={k:rows[-1][k] for k in router},router_maxima={k:max(r[k] for r in rows) for k in router if k in ('moe_dead_expert_fraction','moe_load_cv2')})
    require(audits['temporal']['initial_parameters_sha256']==audits['balance_low']['initial_parameters_sha256'],'MoE arm initialization differs')
    for arm in ('temporal','balance_low'):
        for r,b in zip(logs[arm],logs['dense'],strict=True):
            require(all(r[k]==b[k] for k in ('turn','bucket','positions','learning_rate','canonical_update_equivalent')),'Data/schedule clock differs')
        for k in METRICS:
            require(math.isclose(histories[arm]['validation_history'][0][k],histories['temporal']['validation_history'][0][k],rel_tol=2e-6,abs_tol=2e-6),'Initial MoE validation differs')
    selected_moe=select({k:table[k] for k in ('temporal','balance_low')},control='temporal')
    winner=selected_moe if all(gates(table[selected_moe],table['dense']).values()) else 'dense'
    result=dict(status='passed',kind='tuned_default_temporal_moe_ablation',table=table,inputs=records,
        selected_moe=selected_moe,provisional_winner=winner,gates_vs_dense={k:gates(v,table['dense']) for k,v in table.items() if k!='dense'},
        low_balance_gates=gates(table['balance_low'],table['temporal']),budgets=registration['budgets'],
        scope='One-seed fixed-data 256-update prefix of a 1024-update exposure-based schedule. Dense timing is historical; learning time excludes compile/evaluation/loading. No playing-strength, MFU or convergence claim.')
    publish(STUDY/'comparison.json',result)
    with (STUDY/'curves.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['arm','split','update','learning_seconds',*METRICS])
        for arm,fields in histories.items():
            clock={0:0.,**{r['turn']:r['cumulative_learning_seconds'] for r in logs[arm]}}
            for field,rows in fields.items():
                for t,m in sorted(rows.items()):w.writerow([arm,field,t,clock[t],*[m[k] for k in METRICS]])
    text=['# Tuned-default dense and temporal MoE comparison','','All arms: LR 1e-3, batch 64 games, 256 updates, 7,001,181 matched position exposures.','',
        '| Arm | Parameters | Policy KL | Value MSE | Learning hours |','| --- | ---: | ---: | ---: | ---: |']
    for arm,r in table.items():text.append(f"| {arm} | {r['parameters']:,} | {r['endpoint']['expert_kl']:.6f} | {r['endpoint']['value_mse']:.6f} | {r['learning_seconds']/3600:.3f} |")
    text+=['',f"Provisional winner: {winner}. Retained MoE checkpoint: {selected_moe}.",'',result['scope']]
    (STUDY/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return result
