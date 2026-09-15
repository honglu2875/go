#!/usr/bin/env python3
"""Read-only qualification of every host's full parent checkpoint dependency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.checkpoint_forks import contract
from gozero.snapshots import canonical_json, read_json, verify

REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
root=Path('/workspace/go'); expected=json.load(sys.stdin)
spec=expected['spec']; entry=expected['entry']; source=root/'.gozero/snapshots'/spec['parent_snapshot']
sys.path.insert(0,str(source/'packages/gozero/src'));sys.path.insert(0,str(source/'ops'))
from gozero.snapshots import verify,read_json
from gozero.checkpoints import sha256,read
from preflight import inspect_host
verify(source)
receipt_path=root/spec['native']['receipt'];receipt=read_json(receipt_path)
assert sha256(receipt_path)==spec['native']['receipt_sha256']
assert sha256(receipt_path.parent/receipt['filename'])==receipt['binary_sha256']==spec['native_binary_sha256']
verify(root/'.gozero/snapshots'/receipt['snapshot_id'])
checkpoint=root/entry['path'];group_path=checkpoint.with_suffix('.group.json')
assert sha256(group_path)==spec['parent_group_sha256'] and read_json(group_path)==spec['parent_group']
state,arrays,actors=read(checkpoint,expected_manifest_sha256=entry['manifest_sha256'])
assert state['jax_rank']==entry['jax_rank'] and state['snapshot_id']==source.name
assert state['world_size']==4 and state['turn']==spec['parent_group']['turn']
assert state['config_sha256']==sha256(source/'resolved_config.json')==spec['parent_config_sha256']
assert state['native_sha256']==receipt['binary_sha256']
assert {k:v for k,v in state['counters'].items() if not k.endswith('_seconds')}==entry['counters']
h=hashlib.sha256()
for name in sorted(k for k in arrays if k.startswith('p_'))+sorted(k for k in arrays if k.startswith('v_'))+['key']:
    h.update(arrays[name].tobytes())
assert h.hexdigest()==spec['parent_group']['replicated_state_sha256']
result=inspect_host()
result.update(parent_checkpoint_verified=True,parent_snapshot=source.name,jax_rank=state['jax_rank'],
              host_rank=entry['host_rank'],parent_manifest_sha256=entry['manifest_sha256'],
              native_binary_sha256=receipt['binary_sha256'],checkpoint_arrays=len(arrays))
print(json.dumps(result,sort_keys=True))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE)
    _, spec, _, _ = contract(a.workspace_root, SOURCE, read_json(SOURCE / 'resolved_config.json'))
    a.output.mkdir(parents=True, exist_ok=False)
    interpreter = '/workspace/go/.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
    report = {'schema_version': 1, 'kind': 'continued_selfplay_host_preflight', 'status': 'failed', 'snapshot': SOURCE.name,
              'remote_program_sha256': hashlib.sha256(REMOTE.encode()).hexdigest(), 'started_unix': time.time(), 'hosts': [],
              'scope': 'Read-only complete checkpoint hashes, source and native dependencies, resources and visible device holders. Unreadable process descriptor tables prevent proof of complete device idleness.'}

    def inspect(entry):
        rank = entry['host_rank']; host = f'go-user@worker-{rank}.example.invalid'
        command = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                   shlex.join([interpreter, '-B', '-c', REMOTE])]
        result = subprocess.run(command, input=json.dumps({'spec': spec, 'entry': entry}), text=True,
                                capture_output=True, timeout=90)
        (a.output / f'host-{rank}.stdout.log').write_text(result.stdout)
        (a.output / f'host-{rank}.stderr.log').write_text(result.stderr)
        data = json.loads(result.stdout) if result.returncode == 0 else {}
        ready = result.returncode == 0 and data['parent_checkpoint_verified'] and data['disk_free_bytes'] > 2 * 1024**3 and not data['visible_device_holders']
        return {'host_rank': rank, 'host': host, 'returncode': result.returncode, 'status': 'passed' if ready else 'failed',
                'inspection': data, 'stdout_sha256': sha256(a.output / f'host-{rank}.stdout.log'),
                'stderr_sha256': sha256(a.output / f'host-{rank}.stderr.log')}

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(inspect, entry) for entry in spec['rank_checkpoints']]
            for future in futures:
                report['hosts'].append(future.result())
        if len(report['hosts']) != 4 or any(h['status'] != 'passed' for h in report['hosts']):
            raise ValueError('Parent checkpoint dependency or host readiness failed')
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['finished_unix'] = time.time()
        with (a.output / 'result.json').open('xb') as stream:
            stream.write(canonical_json(report))
        print(json.dumps({k: v for k, v in report.items() if k != 'hosts'}), flush=True)


if __name__ == '__main__':
    main()
