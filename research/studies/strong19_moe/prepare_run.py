"""Register a new fixed-data MoE learning screen after qualification gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'packages/gozero/src'),str(ROOT/'research/recipes/strong19_moe')]
from gozero.snapshots import freeze,canonical_json
import train_config


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,c):
    with p.open('xb') as f:f.write(canonical_json(c))
    p.chmod(0o444)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--steps',type=int,required=True);parser.add_argument('--balance',type=float,required=True);a=parser.parse_args()
    if a.label not in ('balance010-001','balance001-001') or not 16<=a.steps<=64 or a.steps%16 or a.balance!={'balance010-001':.01,'balance001-001':.001}[a.label]:raise ValueError('Scope differs')
    prereqs=[STUDY/n for n in ('cpu-qualification-004.json','kernel-qualification-004.json','system-qualification-001.json','system-review-001.json','budget-001.json','archive-20261004-result-001.json')]
    prereqs+=[ROOT/'research/studies/strong19_recovery/flat-stage-003/audit.json',ROOT/'research/studies/strong19_recovery/cohort-disk-backups-001.json']
    for p in prereqs:
        if json.loads(p.read_text())['status']!='passed':raise ValueError('Prerequisite did not pass: '+str(p))
    base=json.loads((ROOT/'research/studies/strong19_recovery/flat-config-001.json').read_text())
    system=json.loads((STUDY/'system-config-001.json').read_text())
    c={**base,'model':{**base['model'],'moe':{**system['moe'],'balance_weight':a.balance}},
       'checkpoint_every':32 if a.steps==64 else 64,
       'checkpoint_disk':{**base['checkpoint_disk'],'peer':3}}
    train_config.validate(c)
    p=STUDY/(a.label+'-config.json');write(p,c)
    snapshot=freeze(ROOT,Path('research/recipes/strong19_moe'),p,ROOT/'.gozero/snapshots')
    replay=json.loads((ROOT/'research/studies/strong19_recovery/draw-replay-001.json').read_text())['draws']
    budget=json.loads((STUDY/'budget-001.json').read_text())['budgets']['moe']
    parameters=budget['parameters'];raw=parameters*12+4
    count=math.ceil(a.steps/c['checkpoint_every'])
    reserve=math.ceil(raw*1.01)+(32<<20)
    operators={str(p.relative_to(ROOT)):sha(p) for p in [STUDY/n for n in ('prepare_run.py','execute_run.py','audit_run.py','replicate.py','compare_learning.py','run_learning_stage.py','PLAN.md')]}
    plan=dict(kind='registered_joint19_run',snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),
        created=time.time(),purpose='learning',schedule_steps=512,steps=a.steps,checkpoint_every=c['checkpoint_every'],
        parameters=parameters,replica_peer=3,expected_positions=replay[a.steps-1]['cumulative_positions'],validation_positions=64371,
        output_directory=a.label+'-stage',audit_python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'),
        operators=operators,prerequisites={str(p.relative_to(ROOT)):sha(p) for p in prereqs},
        shm_floor_bytes=64<<30,memory_floor_bytes=96<<30,producer_growth_reserve_bytes=0,additional_reserve_by_host={str(h):0 for h in range(4)},
        disk_floor_bytes=2<<30,disk_checkpoint_reserve_by_host={str(h):(reserve*count if h==0 else (reserve*count+(6<<30) if h==3 else 0)) for h in range(4)},
        timeout_seconds=13000,scope='Fresh-state MoE screen against the qualified dense prefix, same data/seed/draws/512-update schedule and objectives, two half-width experts of four; no test target access.',
        reference_snapshot='86bd0fda4df06a20302f5033616213ff7a62814368698b40f41a8a2c77b33b6d',
        reference_attempt='pod-20260928T030525Z-44bb05aa',balance_weight=a.balance,z_weight=.001,
        nominal_active_flop_ratio=json.loads((STUDY/'budget-001.json').read_text())['active_flop_ratio'])
    path=STUDY/(a.label+'-plan.json');write(path,plan)
    print(json.dumps(dict(status='prepared',plan=str(path.relative_to(ROOT)),plan_sha256=sha(path),snapshot=snapshot.name,steps=a.steps,positions=plan['expected_positions'],disk_reserve=plan['disk_checkpoint_reserve_by_host'])))

if __name__=='__main__':main()
