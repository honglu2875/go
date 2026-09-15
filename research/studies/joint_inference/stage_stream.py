"""Stage the qualified V7 worker on the pod and check its framed protocol.

This copies a host-0 build; it does not build on peers or start a model server.
Existing differing artifacts are rejected. Only the named build directory is
written; probes use the two reserved auxiliary CPUs and finish immediately.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / '.gozero/build/katago-v7-stream-001'
RECEIPT_SHA = 'f385efc7c4ca86417908307daa452a9dca5bfea1833f9059b00b4ba89a5a79b2'
BINARY_SHA = 'f2fd5e4b00ba3daa76fd5f2f67c09f37f65bafb9a20d8be972f47b172c526676'
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']

REMOTE = r'''
import base64,hashlib,json,math,os,stat,struct,subprocess,sys,tempfile,time
from pathlib import Path
mode=sys.argv[1];payload=json.load(sys.stdin)
root=Path('/workspace/go')
parent=root/'.gozero/build';dest=parent/'katago-v7-stream-001'
assert root.resolve()==root and parent.resolve()==parent and parent.is_dir()
assert not dest.is_symlink()
def inspect():
    disk=os.statvfs(parent)
    out=dict(exists=dest.exists(),free_bytes=disk.f_bavail*disk.f_frsize,files={})
    if not dest.exists():return out
    assert dest.is_dir()
    for name,record in payload['files'].items():
        p=dest/name;s=p.lstat()
        assert stat.S_ISREG(s.st_mode) and s.st_size==record['bytes']
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==record['sha256'], 'Existing artifact differs: '+str(p)
        if name=='feature_stream':assert os.access(p,os.X_OK)
        out['files'][name]=dict(sha256=digest,bytes=s.st_size,mode=oct(stat.S_IMODE(s.st_mode)))
    return out
before=inspect()
if mode=='inspect':
    assert before['exists'] or before['free_bytes']>2*1024**3+sum(r['bytes'] for r in payload['files'].values())
    out=before
elif mode=='install':
    if not before['exists']:
        assert before['free_bytes']>2*1024**3+sum(r['bytes'] for r in payload['files'].values())
        stage=Path(tempfile.mkdtemp(prefix='.stage-v7-',dir=str(parent)))
        # Validate the complete payload before writing any artifact.
        data={name:base64.b64decode(record['data'],validate=True) for name,record in payload['files'].items()}
        for name,raw in data.items():
            assert name in ('feature_stream','feature_stream.cpp','receipt.json')
            assert len(raw)==payload['files'][name]['bytes']
            assert hashlib.sha256(raw).hexdigest()==payload['files'][name]['sha256']
        for name,raw in data.items():
            with (stage/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            (stage/name).chmod(0o555 if name=='feature_stream' else 0o444)
        assert not dest.exists() and not dest.is_symlink()
        stage.rename(dest)
    out=dict(before=before,after=inspect(),copied_from='worker-0',compiled_on_this_host=False)
elif mode=='probe':
    assert before['exists']
    area=361;rows=4
    request=dict(version=1,id=37,size=19,komi=7.5,
                 histories=[[],[0],[361],[0,361,1]],starts=[0,1,1,3])
    raw=json.dumps(request,separators=(',',':')).encode()
    wire=struct.pack('<I',len(raw))+raw
    start=time.monotonic()
    run=subprocess.run([str(dest/'feature_stream')],input=wire,stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,timeout=10,check=True)
    response=run.stdout
    assert len(response)>=4
    n=struct.unpack('<I',response[:4])[0];assert 0<n<=65536
    header=json.loads(response[4:4+n]);spatial_bytes=rows*area*22;global_bytes=rows*19*4
    audit_bytes=rows*(2*area+1)
    assert header==dict(version=1,id=37,status='ok',size=19,rows=rows,offsets=[0,1,2,3,4],
                        spatial_bytes=spatial_bytes,global_bytes=global_bytes,audit_bytes=audit_bytes)
    body=response[4+n:];assert len(body)==spatial_bytes+global_bytes+audit_bytes
    spatial=body[:spatial_bytes];assert max(spatial)<=1 and all(spatial[::22])
    glob=body[spatial_bytes:spatial_bytes+global_bytes]
    assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',glob))
    audit=body[spatial_bytes+global_bytes:]
    for i,occupied in enumerate(([],[0],[],[0,1])):
        row=audit[i*(2*area+1):(i+1)*(2*area+1)]
        stones=bytearray(area);legal=bytearray([1])*(area+1)
        for point in occupied:stones[point]=1;legal[point]=0
        assert row[:area]==stones and row[area:]==legal
    out=dict(files=before['files'],request_sha256=hashlib.sha256(wire).hexdigest(),
             response_sha256=hashlib.sha256(response).hexdigest(),response_bytes=len(response),
             payload_bytes=len(body),rows=rows,seconds=time.monotonic()-start,
             stderr_bytes=len(run.stderr),cpu_affinity=sorted(os.sched_getaffinity(0)))
else:raise ValueError('Unknown operation')
out.update(mode=mode,observed=time.time(),hostname=os.uname().nodename)
print(json.dumps(out))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != Path(__file__).resolve().parent or output.exists():
        raise ValueError('Use a new receipt in this study')
    files = {}
    for name in ('feature_stream', 'feature_stream.cpp', 'receipt.json'):
        raw = (BUILD / name).read_bytes()
        files[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    if files['receipt.json']['sha256'] != RECEIPT_SHA or files['feature_stream']['sha256'] != BINARY_SHA:
        raise ValueError('Qualified build changed')
    if (BUILD / 'feature_stream.cpp').read_bytes() != (ROOT / 'research/recipes/v7_inference/feature_stream.cpp').read_bytes():
        raise ValueError('Stream source changed')
    records = dict(kind='qualified_v7_stream_pod_staging', status='running', started=time.time(),
                   source_host='worker-0', files=files, operations=[],
                   operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope='Copy qualified host-0 artifacts and execute four deterministic native rows; no TPU, model, dataset, or running service changes.')
    log = output.with_suffix('.events.jsonl')
    if log.exists():
        raise ValueError('Event path exists')

    def one(host, mode):
        data = {name: dict(record) for name, record in files.items()}
        if mode == 'install':
            for name in data:
                data[name]['data'] = base64.b64encode((BUILD / name).read_bytes()).decode('ascii')
        command = 'taskset -c 0,1 python3 -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(mode)
        completed = subprocess.run(SSH + [f'go-user@worker-{host}.example.invalid', command],
                                   input=json.dumps(dict(files=data)).encode(),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if completed.returncode:
            raise RuntimeError(f'Host {host} {mode} failed: ' + completed.stderr.decode(errors='replace')[-4000:])
        return dict(host=host, **json.loads(completed.stdout))

    def record(value):
        records['operations'].append(value)
        with log.open('a') as f:
            f.write(json.dumps(value) + '\n')

    try:
        # Finish all independent preflights before any mutation.
        with ThreadPoolExecutor(4) as pool:
            preflights = list(pool.map(lambda host: one(host, 'inspect'), range(4)))
        for row in preflights:
            record(row)
        for host in range(1, 4):
            record(one(host, 'install'))
        with ThreadPoolExecutor(4) as pool:
            probes = list(pool.map(lambda host: one(host, 'probe'), range(4)))
        for row in probes:
            record(row)
        if len({row['response_sha256'] for row in probes}) != 1:
            raise ValueError('Native output differs across hosts')
        records.update(status='passed', response_sha256=probes[0]['response_sha256'])
    except BaseException as error:
        records.update(status='failed', error=repr(error))
        raise
    finally:
        records['finished'] = time.time()
        if log.exists():
            records['events_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
        with output.open('x') as f:
            json.dump(records, f, indent=2)
            f.write('\n')
        output.chmod(0o444)
        print(json.dumps(dict(status=records['status'], receipt=str(output),
                              sha256=hashlib.sha256(output.read_bytes()).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
