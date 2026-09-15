"""Observe a registered LR pair through a complete validation boundary."""
import argparse
import json
import math
from pathlib import Path
import time

import diagnose_v2 as diagnostic
import lr_source
from lr_compare import read,sha

ROOT=Path(__file__).resolve().parents[3]


def rename(value):
    if isinstance(value,dict):
        return {dict(cnn_kl='parent_kl',transformer_kl='candidate_kl').get(k,k):rename(v) for k,v in value.items()}
    if isinstance(value,list):return [rename(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--stage',choices=('seed1','seed2'),required=True);p.add_argument('--through-turn',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or not 256<=a.through_turn<=4096 or a.through_turn%256:raise ValueError('New output and complete boundary required')
    if sha(a.plan)!=a.plan_sha256:raise ValueError('LR registration changed')
    plan=read(a.plan);item=plan['stages'][a.stage]
    if sha(Path(lr_source.__file__))!=plan['source_helper_sha256']:raise ValueError('LR source helper changed')
    parent_source=ROOT/'.gozero/snapshots'/item['parent_snapshot'];candidate_source=ROOT/'.gozero/snapshots'/item['snapshot']
    lr_source.identical_source(parent_source,candidate_source)
    audit_path=ROOT/item['parent_audit']['path']
    if sha(audit_path)!=item['parent_audit']['sha256']:raise ValueError('Parent audit changed')
    parent_attempt=read(audit_path)['attempt']
    folder=a.plan.parent/plan['trial'];process=read(folder/(a.stage+'-process-001.json'))
    if process['plan_sha256']!=a.plan_sha256 or process['snapshot']!=candidate_source.name:raise ValueError('Wrong candidate process')
    log=folder/(a.stage+'-controller-001.log')
    attempts=[json.loads(x)['attempt_id'] for x in log.read_text().splitlines()[:10] if json.loads(x).get('kind')=='pod_attempt']
    if len(attempts)!=1:raise ValueError('Expected a single candidate attempt')
    arms={};evidence={}
    for name,attempt,snapshot in [('parent',parent_attempt,parent_source),('candidate',attempts[0],candidate_source)]:
        base=ROOT/'runs'/attempt
        if read(base/'launch.json')['snapshot_id']!=snapshot.name:raise ValueError('Attempt source differs')
        artifacts=base/'rank-0/artifacts';config=read(artifacts/'resolved_config.json')
        if config!=read(snapshot/'resolved_config.json'):raise ValueError('Run configuration differs')
        rows=[r for r in diagnostic.read_rows(artifacts/'metrics.jsonl') if r['turn']<=a.through_turn]
        if [r['turn'] for r in rows]!=list(range(1,a.through_turn+1)) or not all(r['accepted'] for r in rows):
            raise ValueError('Missing or rejected updates')
        evaluations=[r for r in diagnostic.read_rows(artifacts/'evaluations.jsonl') if r['turn']<=a.through_turn and r['kind'] in ('visual_heldout','visual_training_probe')]
        keyed={(r['kind'],r['turn']):r for r in evaluations}
        if len(keyed)!=len(evaluations) or set(keyed)!={(k,t) for k in ('visual_heldout','visual_training_probe') for t in range(0,a.through_turn+1,256)}:
            raise ValueError('Incomplete paired validation/probe history')
        arms[name]=dict(rows=rows,evaluations=keyed,config=config)
        evidence[name]=dict(attempt=attempt,snapshot=snapshot.name,
            metrics_prefix_sha256=diagnostic.sha(diagnostic.canonical(rows)),
            evaluation_prefix_sha256=diagnostic.sha(diagnostic.canonical(evaluations)),
            optimizer_windows=diagnostic.optimizer_windows(rows))
    ratio=plan['peak_learning_rate']/arms['parent']['config']['learner']['learning_rate']
    for x,y in zip(arms['parent']['rows'],arms['candidate']['rows']):
        for key in ('turn','bucket','local_entries_sha256','local_symmetries','expert_positions'):
            if x[key]!=y[key]:raise ValueError('Owner draw/exposure mismatch: '+key)
        if not math.isclose(y['learning_rate'],x['learning_rate']*ratio,rel_tol=3e-7,abs_tol=1e-12):
            raise ValueError('Observed LR schedule differs from registered ratio')
    curves=[dict(turn=t,populations={k:rename(diagnostic.cohorts(arms['parent']['evaluations'][k,t],arms['candidate']['evaluations'][k,t]))
        for k in ('visual_heldout','visual_training_probe')}) for t in range(0,a.through_turn+1,256)]
    result=dict(kind='registered_lr_prefix_diagnostic',status='passed',created=time.time(),through_turn=a.through_turn,
        plan_sha256=a.plan_sha256,operator_sha256=sha(Path(__file__)),diagnostic_helper_sha256=sha(Path(diagnostic.__file__)),
        stage=a.stage,evidence=evidence,curves=curves,owner_draws_and_exposures_identical=True,
        scope='Partial owner-host diagnostics, with pinned source pair and observed LR ratio. Complete all-rank audits and registered endpoint decisions remain separate; no run or schedule changed.')
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status='passed',through_turn=a.through_turn,sha256=sha(a.output),latest={k:{m:v[m]['relative_delta'] for m in ('expert','family')} for k,v in curves[-1]['populations'].items()})))


if __name__=='__main__':main()
