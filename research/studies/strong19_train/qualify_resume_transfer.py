"""Exercise the frozen joint checkpoint stager and reader on all four hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json, verify
from gozero.pod import load_hosts, SSH_OPTIONS


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_json(value))
    path.chmod(0o444)


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=int, default=1)
    args = parser.parse_args(); snapshot = args.snapshot.resolve(); manifest = verify(snapshot)
    if not 1 <= args.case <= 999:
        raise ValueError('Invalid qualification case number')
    case = f'{args.case:03d}'
    if args.output.exists():
        raise FileExistsError(args.output)
    workspace = STUDY / f'transfer-fixture-{case}/workspace'
    if workspace.exists():
        raise FileExistsError('Transfer fixture already exists')
    fs = os.statvfs('/dev/shm')
    if fs.f_bavail * fs.f_frsize < 64 * (1 << 30) + (1 << 20):
        raise ValueError('Insufficient fixture headroom')
    attempt_name = 'pod-20260915T083600Z-' + snapshot.name[:8]
    attempt = workspace / 'runs' / attempt_name
    logical = attempt / 'rank-0/artifacts/checkpoints/turn-000000002'
    owner = Path('/dev/shm/gozero-staged-checkpoints') / logical.relative_to(workspace / 'runs')
    if owner.exists():
        raise FileExistsError('Fixture RAM namespace occupied')
    started = time.monotonic(); mapping = [dict(host=h, jax_rank=r) for h, r in enumerate((1, 3, 0, 2))]
    arrays = dict(p_0000=np.arange(77, dtype=np.float32).reshape(7, 11) / 100,
                  m_0000=np.full((7, 11), .25, np.float32), v_0000=np.full((7, 11), .5, np.float32),
                  step=np.asarray(2, np.int32))
    payload = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                         sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in arrays.items()}
    array_digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array_digest.update(canonical_json([key, list(value.shape), str(value.dtype)]))
        array_digest.update(value.tobytes())
    group = dict(schema_version=1, kind='visual_replicated_checkpoint_group', snapshot_id=snapshot.name,
        turn=2, config_sha256='c'*64, owner_checkpoint_path=str(owner), host_jax_mapping=mapping,
        host_manifests={}, replicated_arrays_elements_sha256=array_digest.hexdigest())
    directories = []
    for host in range(4):
        directory = owner if host == 0 else attempt / f'rank-{host}/artifacts/checkpoints/turn-000000002'
        state = dict(kind='visual_replicated_rank_state', snapshot_id=snapshot.name, host_rank=host,
            jax_rank=mapping[host]['jax_rank'], config_sha256=group['config_sha256'], turn=2,
            dataset_manifest_sha256='d'*64, model_schema=[dict(path='fixture.weight', shape=[7, 11], dtype='float32')],
            optimizer_metadata=dict(step=2), counters=dict(updates=2, expert_positions=512),
            owns_replicated_arrays=host == 0, numpy_rng=dict(fixture_seed=100+host))
        group['host_manifests'][str(host)] = checkpoints.write(directory, state=state,
            arrays=arrays if host == 0 else {}, actors='{}')
        directories.append(directory)
    for directory in directories:
        publish(directory.with_suffix('.group.json'), group)
    publish(logical.with_suffix('.group.json'), group)
    publish(logical.with_suffix('.temporary.json'), dict(kind='temporary_ram_checkpoint',
        cache_path=str(owner), logical_path=str(logical), manifest_sha256=group['host_manifests']['0']))
    from gozero.ram_checkpoints import seal
    seal([owner / name for name in ('manifest.json', 'arrays.npz', 'state.json', 'actors.json')]
         + [owner.with_suffix('.group.json')])
    group_path = logical.with_suffix('.group.json')
    plan = dict(kind='joint_resume_transfer_fixture_plan', created=time.time(), snapshot=snapshot.name,
        workspace=str(workspace), attempt=str(attempt), group_path=str(group_path), group_sha256=checkpoints.sha256(group_path),
        payload=payload, operator_sha256=checkpoints.sha256(Path(__file__)),
        scope='Synthetic four-rank checkpoint transport and exact reader validation only. No model training, JAX distributed initialization or TPU allocation.')
    publish(STUDY / f'resume-transfer-plan-{case}.json', plan)
    hosts = load_hosts(snapshot / 'ops/hosts.json')
    def stage_source(host):
        subprocess.run(['ssh', *SSH_OPTIONS, host.ssh, shlex.join(['mkdir', '-p', str(snapshot)])], check=True, timeout=30)
        subprocess.run(['rsync', '-a', '--ignore-existing', '--rsync-path=taskset -c 0,1 rsync',
                        '-e', shlex.join(['ssh', *SSH_OPTIONS]), str(snapshot) + '/', host.ssh + ':' + str(snapshot) + '/'],
                       check=True, timeout=120, stdout=subprocess.DEVNULL)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(stage_source, hosts))
    staging = STUDY / f'resume-transfer-staging-{case}'
    with (STUDY / f'resume-transfer-staging-{case}.log').open('xb') as log:
        subprocess.run([sys.executable, '-B', str(snapshot / 'ops/stage_joint_checkpoint.py'),
            '--workspace-root', str(workspace), '--manifest', str(group_path), '--expected-sha256', plan['group_sha256'],
            '--cpu-list', '0,1', '--output', str(staging)], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    code = '''import hashlib,json,sys
from pathlib import Path
snapshot=Path(sys.argv[1]);plan=json.loads(sys.argv[2]);sys.path.insert(0,str(snapshot/'packages/gozero/src'))
from gozero.snapshots import verify,read_json
from gozero.joint_resume import resume_path,collect
from gozero import checkpoints
verify(snapshot);files=collect(Path(plan['workspace']),Path(plan['group_path']),plan['group_sha256'],snapshot=snapshot.name)
group=read_json(Path(plan['group_path']));observed=[]
for host in range(4):
 path=resume_path(Path(plan['workspace']),Path(plan['attempt']),host,2,snapshot.name)
 state,arrays,_=checkpoints.read(path,expected_manifest_sha256=group['host_manifests'][str(host)])
 assert state['host_rank']==host and state['numpy_rng']==dict(fixture_seed=100+host)
 actual={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
 assert actual==(plan['payload'] if host==0 else {})
 observed.append(dict(role=host,arrays=len(arrays),state_exact=True))
print(json.dumps(dict(status='passed',files=len(files),roles=observed,snapshot=snapshot.name)))
'''
    python = ROOT / '.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
    def check(host):
        command = ['taskset', '-c', '0,1', str(python), '-B', '-c', code, str(snapshot), json.dumps(plan)]
        out = subprocess.check_output(['ssh', *SSH_OPTIONS, host.ssh, shlex.join(command)], text=True, timeout=180)
        return dict(host=host.rank, **json.loads(out))
    with ThreadPoolExecutor(4) as pool:
        observations = list(pool.map(check, hosts))
    result = dict(kind='joint_resume_four_host_transfer_qualification', status='passed', created=time.time(),
        snapshot=snapshot.name, plan_sha256=checkpoints.sha256(STUDY / f'resume-transfer-plan-{case}.json'),
        staging_receipt_sha256=checkpoints.sha256(staging / 'receipt.json'), hosts=observations,
        seconds=time.monotonic()-started, scope=plan['scope'], tpu_jobs_launched=False)
    publish(args.output, result)
    print(json.dumps(dict(status='passed', seconds=result['seconds'], result_sha256=checkpoints.sha256(args.output))), flush=True)


if __name__ == '__main__':
    main()
