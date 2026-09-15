"""Qualify all hosts before any production drain; record each bounded handover."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

SOURCE = Path(__file__).resolve().parent
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
PYTHON = '/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    with path.open('x') as f: json.dump(data, f, indent=2); f.write('\n')
    path.chmod(0o444)


def main():
    d = json.loads((SOURCE/'deployment.json').read_text())
    qualification_path = SOURCE/'qualification-result-001.json'
    q = json.loads(qualification_path.read_text())
    assert q['status'] == 'passed' and q['producer_snapshot'] == d['snapshot']
    assert q['games'] == 24 and q['positions'] == 96 and q['qualification_contract_isolated']
    assert [(c['teacher_threads'], c['opponent_threads']) for c in q['cases']] == [(7, 1), (4, 4)]
    assert all(c['restart_preserved_publications'] and c['state'] == 'stopped' for c in q['cases'])
    contract = json.loads(Path(d['contract']).read_text())
    host_source = (SOURCE/'deploy_host.py').read_text()
    compile(host_source, '<qualified-host-handover>', 'exec')
    root, env = Path(d['root']), Path(d['environment'])
    receipt_dir = SOURCE/'handover-001'; receipt_dir.mkdir(exist_ok=False)
    def payload(host, mode):
        c = dict(storage_root=str(root), run_id=d['run_id'], host_index=host,
                 concurrent_games=8, research_cpus=[0,1], drain_seconds=86400,
                 contract=contract, producer_snapshot=d['snapshot'], producer_environment=str(env),
                 storage_limits=dict(files_cap=24*(1 << 30), free_files_floor=64*(1 << 30),
                                     available_memory_floor=96*(1 << 30)),
                 workers=[dict(cpus=list(range(start, start+8)), opponent_index=i,
                               teacher_threads=4 if i == 7 else 7, opponent_threads=4 if i == 7 else 1)
                          for i, start in enumerate((8,16,24,32,40,48,56,64))])
        return dict(mode=mode, config=c, qualified_sources=q['source_sha256'],
                    native_sha256=q['native_sha256'], qualification_sha256=sha(qualification_path),
                    contract_id=d['contract_id'])
    def remote(host, mode):
        target = f'go-user@worker-{host}.example.invalid'
        if mode == 'verify' and host:
            subprocess.run(['rsync', '-a', '--ignore-existing', '-e', shlex.join(SSH),
                            str(env), f'{target}:{root}/environments/'], check=True, timeout=180)
        command = 'taskset -c 0,1 '+shlex.join([PYTHON, '-B', '-c', host_source, json.dumps(payload(host, mode))])
        result = json.loads(subprocess.check_output(SSH+[target, command], text=True, timeout=180))
        save(receipt_dir/f'host-{host}-{mode}.json', result)
        print(json.dumps(dict(mode=mode, **result)), flush=True)
        return result
    # Immutable source staging and read-only process/model/resource verification
    # can run concurrently; production handovers are deliberately sequential.
    with ThreadPoolExecutor(4) as pool:
        verified = list(pool.map(lambda h: remote(h, 'verify'), range(4)))
    save(receipt_dir/'preflight.json', dict(hosts=verified, completed=time.time(),
         operator_sha256=sha(Path(__file__)), host_operator_sha256=sha(SOURCE/'deploy_host.py'),
         qualification_sha256=sha(qualification_path)))
    launched = [remote(host, 'handover') for host in range(4)]
    save(receipt_dir/'cluster.json', dict(status='handover_started', hosts=launched, completed=time.time(),
         producer_snapshot=d['snapshot'], production_contract=d['contract_id'],
         semantics='Each v2 worker starts only after its recorded old worker and two engines exit; old games drain'))


if __name__ == '__main__':
    main()
