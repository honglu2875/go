"""Prospective, bounded LR decisions; selection uses validation only."""
import math
from execute_run import ROOT,read,require,sha

METRICS=('expert_kl','family_kl','value_mse','value_family_mse','expert_top1')


def summarize(record,equivalent=128):
    path=ROOT/record['audit'];require(sha(path)==record['audit_sha256'],'Audit changed')
    a=read(path);size=record['batch_games'];turn=equivalent*128//size;spacing=16*128//size
    require(a['status']=='passed' and a['all_rank_metrics_and_saved_state_verified'] and a['parameters']==232011540,
            'Wrong model/audit')
    require(a['steps']>=turn and turn*size==equivalent*128,'Incomplete comparison horizon')
    val={r['turn']:r for r in a['validation_history']};probe={r['turn']:r for r in a['training_probe_history']}
    require(all(t in val and t in probe for t in range(0,turn+1,spacing)),'Missing validation')
    require(all(r['episode_ids_sha256']=='7bef19b033a315916e31de6216a0119c35065c14fe04dbace94e7430ecd58358' for r in val.values()),'Validation population changed')
    require(all(r['episode_ids_sha256']=='9bf1249ea023afe59d781a8758cb824d5af82e519516d2feec162127c2bb09fe' for r in probe.values()),'Probe population changed')
    rows=[__import__('json').loads(x) for x in (ROOT/'runs'/record['attempt']/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()][:turn]
    require([r['turn'] for r in rows]==list(range(1,turn+1)),'Nonfresh or incomplete log')
    return dict(peak_lr=record['peak_lr'],batch_games=size,steps=turn,position_exposures=sum(r['positions'] for r in rows),
        endpoint={k:val[turn]['metrics'][k] for k in METRICS},
        tail={k:sum(val[t]['metrics'][k] for t in (turn-2*spacing,turn-spacing,turn))/3 for k in METRICS},
        training_probe={k:probe[turn]['metrics'][k] for k in METRICS},
        learning_seconds=rows[-1]['cumulative_learning_seconds'],
        sustained_overfit=any(r['sustained'] for r in a['overfit_observations'] if r['turn']<=turn),
        initial_parameters_sha256=a['initial_parameters_sha256'])


def gates(row,base,margin=.005):
    return dict(endpoint_policy=all(row['endpoint'][k]<=(1-margin)*base['endpoint'][k] for k in ('expert_kl','family_kl')),
        tail_policy=all(row['tail'][k]<=base['tail'][k] for k in ('expert_kl','family_kl')),
        value=row['tail']['value_mse']<=1.05*base['tail']['value_mse'],no_overfit=not row['sustained_overfit'])


def select(table,control='control'):
    passing=[k for k,v in table.items() if k!=control and all(gates(v,table[control]).values())]
    return min(passing,key=lambda k:(table[k]['endpoint']['expert_kl'],table[k]['endpoint']['family_kl'],table[k]['peak_lr'])) if passing else control


def next_lr(table,round_index,previous_new=None):
    best=select(table);rates=sorted(v['peak_lr'] for v in table.values());peak=table[best]['peak_lr']
    if round_index>=2:return dict(action='stop',reason='Two additional LR probes exhausted',incumbent=best)
    if round_index and previous_new!=best:
        return dict(action='stop',reason='The latest refinement did not change the screened incumbent',incumbent=best)
    if best!='control' and peak in (min(rates),max(rates)):
        rate=peak*(.625 if peak==min(rates) else 1.6);reason='Extend the improving boundary'
    else:
        neighbors=sorted((k for k in table if k!=best),key=lambda k:table[k]['endpoint']['expert_kl'])
        neighbor=next((k for k in neighbors if table[k]['tail']['value_mse']<=1.1*table['control']['tail']['value_mse']
                       and not table[k]['sustained_overfit']),None)
        if neighbor is None:return dict(action='stop',reason='No stable adjacent direction to refine',incumbent=best)
        # Use the immediate sampled neighbor toward the best alternative.
        side=1 if table[neighbor]['peak_lr']>peak else -1
        adjacent=min((x for x in rates if (x-peak)*side>0),key=lambda x:abs(math.log(x/peak)))
        rate=math.sqrt(peak*adjacent);reason='Refine the interval toward the best stable alternative'
    rate=float(f'{rate:.3g}')
    if not .00025<=rate<=.004 or any(abs(math.log(rate/x))<math.log(1.08) for x in rates):
        return dict(action='stop',reason='Registered LR range/resolution reached',incumbent=best)
    return dict(action='run',peak_lr=rate,reason=reason,incumbent=best,equivalent_updates=128)
