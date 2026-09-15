#!/usr/bin/env python3
"""Read-only SSH checks of the fixed observer, data manifests and host resources."""
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
from gozero.snapshots import canonical_json, verify

REMOTE = r'''
import json,sys
from pathlib import Path
root=Path('/workspace/go')
source=root/'.gozero/snapshots/89df20c48969445095df4e2393e2070cbb2acab59dacff6834fd1ce684f072d6'
sys.path.insert(0,str(source/'packages/gozero/src'));sys.path.insert(0,str(source/'ops'))
from gozero.snapshots import verify,read_json
from gozero.checkpoints import sha256
from gozero.causal_artifacts import validate
from preflight import inspect_host
verify(source)
result=inspect_host()
descriptor_path=source/'eval/board_state_distillation/exact.json'
assert sha256(descriptor_path)=='4f1cd49a3834c1f6aefcca7827a8ef873d123b689097050081bf0b2a1e5fb541'
descriptor=read_json(descriptor_path);validate(root,descriptor)
manifest=root/'.gozero/datasets/board-causal-13bcd5cd/manifest.json'
assert sha256(manifest)=='6c935aa44ec03893878f43f2456e01be96240b478489547235175441cb58169e'
parent=read_json(manifest)['parent_dataset']
assert sha256(Path(parent['path'])/'manifest.json')==parent['manifest_sha256']
result.update(fixed_observer_validated=True,observer_descriptor_sha256=sha256(descriptor_path),
              observer_weights_sha256=descriptor['model_export_sha256'],dataset_manifest_sha256=sha256(manifest),
              remote_helper_snapshot=source.name)
print(json.dumps(result,sort_keys=True))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); a.output.mkdir(parents=True, exist_ok=False)
    interpreter = '/workspace/go/.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
    report = {'schema_version': 1, 'kind': 'state_expert_host_preflight', 'status': 'failed', 'snapshot': SOURCE.name,
              'remote_program_sha256': hashlib.sha256(REMOTE.encode()).hexdigest(), 'started_unix': time.time(), 'hosts': [],
              'scope': 'Read-only resources, visible device holders, data manifests and complete observer lineage. Unreadable descriptor tables prevent proof of complete device idleness; full shard validation remains in the trainer.'}
    def inspect(rank):
        host = f'go-user@worker-{rank}.example.invalid'
        command = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                   shlex.join([interpreter, '-B', '-'])]
        result = subprocess.run(command, input=REMOTE, text=True, capture_output=True, timeout=90)
        (a.output / f'host-{rank}.stdout.log').write_text(result.stdout)
        (a.output / f'host-{rank}.stderr.log').write_text(result.stderr)
        if result.returncode != 0:
            return {'host': host, 'returncode': result.returncode, 'command': command, 'status': 'failed'}
        data = json.loads(result.stdout)
        ready = data['fixed_observer_validated'] and data['disk_free_bytes'] > 1024**3 and not data['visible_device_holders']
        return {'host': host, 'returncode': result.returncode, 'command': command, 'status': 'passed' if ready else 'not_ready',
                'inspection': data, 'stdout_sha256': sha256(a.output / f'host-{rank}.stdout.log'),
                'stderr_sha256': sha256(a.output / f'host-{rank}.stderr.log')}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(inspect, rank) for rank in range(4)]
            for future in futures:
                report['hosts'].append(future.result())
        if any(h['status'] != 'passed' for h in report['hosts']):
            raise ValueError('At least one host failed readiness checks')
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['finished_unix'] = time.time()
        with (a.output / 'result.json').open('xb') as f:
            f.write(canonical_json(report))
        verify(SOURCE)
        print(json.dumps({**{k: v for k, v in report.items() if k != 'hosts'},
            'hosts': [{k: v for k, v in h.items() if k not in ('inspection', 'command')} for h in report['hosts']]}), flush=True)


if __name__ == '__main__':
    main()
