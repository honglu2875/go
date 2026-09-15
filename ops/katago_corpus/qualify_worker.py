"""Regression gate for the complete worker and the storage API it actually calls."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    source=Path(__file__).resolve().parent
    d=json.loads((source/'deployment.json').read_text())
    env=Path(d['environment']);root=Path(d['root'])
    sys.path.insert(0,str(env/'site-packages'))
    from flygo.data.corpus import audit_game,atomic_json
    from flygo.data.generate import run_worker
    from flygo.storage import Limits,StorageBudget,StoragePressure
    old_env=root/'environments/6865d7cbb11eeddf0ed68859401fc0bcd52682131aa9762f8115f2a76b092e65'
    old=json.loads((old_env/'snapshot.json').read_text())
    new=json.loads((env/'snapshot.json').read_text())
    assert old['native_sha256']==new['native_sha256']
    changed=[name for name,h in new['source_sha256'].items() if h!=old['source_sha256'][name]]
    assert changed==['storage.py'], changed
    previous=json.loads((root/'qualification/result.json').read_text())
    assert previous['status']=='passed'
    atomic_json(source/'qualification-models-first.json',previous)
    test_root=root/'storage-api-qualification'
    limits=Limits(files_cap=32*(1<<20),free_files_floor=0,available_memory_floor=0)
    budget=StorageBudget(test_root,limits)
    budget.check(files=128)
    try:budget.check(files=-1)
    except ValueError:pass
    else:raise AssertionError('Negative reservation accepted')
    try:budget.check(files=33*(1<<20))
    except StoragePressure:pass
    else:raise AssertionError('Storage cap was ignored')
    with budget.reserve(files=16*(1<<20),heap=0,purpose='worker interface regression'):
        try:budget.check(files=17*(1<<20))
        except StoragePressure:pass
        else:raise AssertionError('Pending reservation was ignored')
    qroot=root/'worker-qualification';qroot.mkdir(exist_ok=True)
    artifacts=qroot/'artifacts'
    if not artifacts.exists():artifacts.symlink_to(root/'artifacts',target_is_directory=True)
    contract={**json.loads(Path(d['contract']).read_text()),'max_moves':8,
              'qualification_only':'bounded real worker regression, never production data'}
    c=dict(storage_root=str(qroot),run_id='worker-api-v1',host_index=0,concurrent_games=2,
           workers=[dict(cpus=list(range(8)),opponent_index=0)],contract=contract,
           storage_limits=dict(files_cap=24*(1<<30),free_files_floor=64*(1<<30),available_memory_floor=96*(1<<30)))
    config=qroot/'config.json';atomic_json(config,c)
    started=time.time();run_worker(config,0,batches=1)
    records=[audit_game(p) for p in sorted((qroot/'corpora').glob('*/host-0/worker-00/*.npz'))]
    assert len(records)==2 and all(r['rows']==8 and not r['terminal'] for r in records)
    status=json.loads((qroot/'runs/worker-api-v1/worker-00/status.json').read_text())
    assert status['state']=='stopped' and status['completed_games']==2
    result=dict(previous,producer_snapshot=d['snapshot'],native_sha256=new['native_sha256'],
        unchanged_model_gate_snapshot=old['snapshot'],changed_sources=changed,
        storage_api_regression=True,full_worker_regression=dict(games=2,positions=16,state='stopped',elapsed_seconds=time.time()-started),
        qualification_contract_isolated=True,completed=time.time())
    atomic_json(root/'qualification/result.json',result)
    atomic_json(source/'qualification-result.json',result)
    print(json.dumps(dict(status='passed',producer_snapshot=d['snapshot'],full_worker=result['full_worker_regression'])),flush=True)


if __name__=='__main__':main()
