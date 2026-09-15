"""Qualify complete real-board Muon training and fresh-process recovery."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import read_json,verify


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sha=checkpoints.sha256
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Plan changed')
    plan=read_json(a.plan)
    if sha(Path(__file__))!=plan['operator_sha256']:raise ValueError('Operator changed')
    parent=ROOT/plan['parent_operator']
    if sha(parent)!=plan['parent_operator_sha256']:raise ValueError('Qualified recovery auditor changed')
    spec=importlib.util.spec_from_file_location('qualified_joint_harness',parent)
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    snapshot=ROOT/'.gozero/snapshots'/plan['snapshot_id'];verify(snapshot)
    c=read_json(snapshot/'resolved_config.json')
    if (c['training']['purpose']!='qualification' or c['training']['optimizer']!='muon_explicit'
            or c['dataset']['manifest_sha256']!=plan['dataset_sha256']):
        raise ValueError('Wrong Muon qualification configuration')
    folder=ROOT/plan['run_directory']
    if folder.exists():raise FileExistsError(folder)
    started=time.monotonic()
    report=dict(kind='real19_muon_training_cpu_continuation_qualification',created=time.time(),
        plan_sha256=a.plan_sha256,snapshot_id=snapshot.name,status='running')
    try:
        full=harness.run_stage(a.python,snapshot,folder/'full')
        prefix=harness.run_stage(a.python,snapshot,folder/'prefix',stop=2)
        resumed=harness.run_stage(a.python,snapshot,folder/'resumed',resume=Path(prefix['latest_checkpoint']['path']))
        for result in (full,prefix,resumed):
            if result['optimizer_family']!='standard_muon_aux_adam':raise ValueError('Wrong optimizer family')
        report['continuation']=harness.audit_model(folder,snapshot,full,prefix,resumed)
        saved,arrays,_=checkpoints.read(Path(full['latest_checkpoint']['path']),
            expected_manifest_sha256=full['latest_checkpoint']['manifest_sha256'])
        meta=saved['optimizer_metadata']
        if meta['kind']!='katago_muon_aux_adam_state':raise ValueError('Wrong optimizer state codec')
        roles=meta['parameter_schema']
        muon_groups={'normal','normal_attn','normal_gab','gab_mlp','tab_module'}
        muon_indices=[i for i,r in enumerate(roles) if r['group'] in muon_groups]
        adam_indices=[i for i,r in enumerate(roles) if r['group'] not in muon_groups]
        if not muon_indices or not adam_indices:raise ValueError('Both optimizer algorithms must be exercised')
        if any(f'v_{i:04d}' in arrays for i in muon_indices) or any(f'v_{i:04d}' not in arrays for i in adam_indices):
            raise ValueError('Mixed optimizer state coverage differs')
        report.update(status='passed',muon_parameter_leaves=len(muon_indices),adam_parameter_leaves=len(adam_indices))
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report.update(seconds=time.monotonic()-started,
            scope='Small complete CNN policy/value model; real 19x19 whole games; four simulated CPU devices; full Muon momentum, auxiliary Adam moments, RNGs, counters and diagnostics. Fixed explicit group settings, no Lookahead. Not a learning comparison, full-size TPU qualification or complete published training reproduction.')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        a.output.chmod(0o444)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
