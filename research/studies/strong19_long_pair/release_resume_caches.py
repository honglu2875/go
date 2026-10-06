"""Release only redundant peer staging of the completed CE prefix."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if any(not (p.parent / 'result.json').exists() for p in (ROOT / 'runs').glob('pod-*/launch.json')):
        raise ValueError('An accelerator attempt is open')
    source = ROOT / 'research/studies/strong19_value_debug'
    result = json.loads((source / 'ce-continuation-001/result.json').read_text())
    assert result['status'] == 'passed'
    receipt = source / 'ce-prefix-001/replica.json'
    assert sha(receipt) == 'c7c90b2c69e2c63773912816ad32b74829185da8b65b007beb648d57f230bfd7'
    record = json.loads(receipt.read_text())
    local, peer = record['copies']
    assert local['host'] == 0 and peer['host'] == 3 and local['files'] == peer['files']
    for name, expected in local['files'].items():
        assert sha(Path(local['path']) / name) == expected['sha256']
    expected = local['files']['arrays.npz']
    verify_code = f'''from pathlib import Path
import hashlib,json
p=Path({peer['path']!r});expected={peer['files']!r}
for name,item in expected.items():
 with (p/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
print(json.dumps(dict(status='passed',retained=str(p))))
'''
    preserved = json.loads(subprocess.check_output(SSH + ['go-user@worker-3.example.invalid',
        'taskset -c 0,1 /home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -c ' + shlex.quote(verify_code)], text=True, timeout=90))
    # Owner and the distinct peer replica above remain intact. The paths below
    # were transport caches for the now-completed continuation, never owners.
    path = str(Path(local['path']) / 'arrays.npz')
    code = f'''from pathlib import Path
import hashlib,json,stat
p=Path({path!r});e={expected!r}
assert not any(x.is_symlink() for x in (p,*p.parents))
s=p.stat();assert stat.S_ISREG(s.st_mode) and not s.st_mode&0o222 and s.st_size==e['bytes']
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==e['sha256']
p.unlink()
print(json.dumps(dict(status='passed',path=str(p),reclaimed_bytes=e['bytes'],metadata_preserved=True)))
'''
    (STUDY / 'resume-cache-release-intent-001.json').write_text(json.dumps(dict(
        created=time.time(),attempt=record['attempt'],array=expected,peers=[1,2,3],
        preserved_owner=local['path'],preserved_peer=preserved,operator_sha256=sha(Path(__file__))),indent=2)+'\n')
    def release(host):
        row=json.loads(subprocess.check_output(SSH + [f'go-user@worker-{host}.example.invalid',
            'taskset -c 0,1 /home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -c '+shlex.quote(code)],text=True,timeout=90))
        return dict(host=host,**row)
    # Sequential mutations with each completed receipt flushed immediately.
    rows=[]
    for host in (1,2,3):
        row=release(host);rows.append(row)
        with (STUDY/'resume-cache-release-events-001.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
    outcome=dict(status='passed',completed=time.time(),rows=rows,retained_copies=2)
    with (STUDY/'resume-cache-release-result-001.json').open('x') as f:
        json.dump(outcome,f,indent=2);f.write('\n')
    print(json.dumps(outcome))


if __name__ == '__main__':
    main()
