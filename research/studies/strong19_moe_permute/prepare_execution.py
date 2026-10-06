"""Register a kernel-qualified candidate for a complete-learner comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'research/recipes/strong19_moe_permute'),str(ROOT/'packages/gozero/src')]
from gozero.snapshots import canonical_json,freeze,verify
from qualify_execution import validate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('xb') as stream:stream.write(canonical_json(value))
    path.chmod(0o444)


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True)
    p.add_argument('--kernel-audit',type=Path,required=True)
    p.add_argument('--tile',type=int,nargs=3,required=True);p.add_argument('--permutation',action='store_true')
    a=p.parse_args()
    if not re.fullmatch('[a-z][a-z0-9-]{0,60}',a.label):raise ValueError('Invalid label')
    audit=json.loads(a.kernel_audit.read_text())
    if audit['status']!='passed':raise ValueError('Complete kernel audit must pass')
    verify(ROOT/'.gozero/snapshots'/audit['snapshot'])
    selected=[c for report in audit['reports'] for c in report['cases']
              if c['tiling']==a.tile and c['permutation_vjp']==a.permutation]
    expected={(n,h,g,collapse) for n,h,g in [(2888,1536,False),(2888,512,True),(12288,1024,True),(18432,1024,True),(8,1024,True)] for collapse in (False,True)}
    if (len(selected)!=10 or {(c['tokens'],c['hidden'],c['gated'],c['collapsed']) for c in selected}!=expected
            or any(c['status']!='passed' or max(e['relative_l2'] for e in c['errors'])>.01 for c in selected)):
        raise ValueError('Candidate combination did not pass all registered kernel shapes')
    prerequisites=[a.kernel_audit.resolve(),STUDY/'cpu-qualification-001.json',STUDY/'unit-tests-002.json',
                   ROOT/'research/studies/strong19_moe/balance010-001-stage/result.json']
    for path in prerequisites:
        path.relative_to(ROOT)
        if json.loads(path.read_text())['status']!='passed':raise ValueError('A prerequisite did not pass')
    baseline=ROOT/'research/studies/strong19_moe/balance010-001-config.json'
    c=json.loads((ROOT/'research/studies/strong19_moe_remat/system-proposal-config-001.json').read_text())
    for key in list(c):
        if key.startswith('chunk_'):del c[key]
    c.update(kind='moe_execution_qualification',reference_config=json.loads(baseline.read_text()),
             variants=['original8','candidate8'],candidate_moe_overrides=dict(tiling=a.tile,permutation_vjp=a.permutation),
             group_relative_l2_limits=dict(params=1e-4,first=1e-3,second=5e-3),
             first_moment_min_cosine=.99999,metric_rtol=.005,metric_atol=1e-4)
    c['reference_config_sha256']=hashlib.sha256(canonical_json(c['reference_config'])).hexdigest()
    validate(c)
    config=STUDY/(a.label+'-config.json');write(config,c)
    snapshot=freeze(ROOT,Path('research/recipes/strong19_moe_permute'),config,ROOT/'.gozero/snapshots')
    operators=[STUDY/n for n in ('EXECUTION_PLAN.md','prepare_execution.py','execute_execution.py','audit_execution.py','review_execution.py')]
    plan=dict(status='prepared',kind='moe_execution_qualification_registration',created=time.time(),
              snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),
              must_follow_successful_attempt=audit['attempt'],timeout_seconds=3600,
              output_directory=a.label+'-stage',accelerator_job_started=False,
              prerequisites={str(path.relative_to(ROOT)):sha(path) for path in prerequisites},
              operators={str(path.relative_to(ROOT)):sha(path) for path in operators},
              scope='Four fresh complete-model cases at two buckets. Full parameters and optimizer moments compared in RAM under predeclared drift/memory limits. No learned checkpoint or learning claim.')
    path=STUDY/(a.label+'-registration.json');write(path,plan)
    print(json.dumps(dict(status='prepared',snapshot=snapshot.name,registration=str(path.relative_to(ROOT)),sha256=sha(path),accelerator_job_started=False)))


if __name__=='__main__':main()
