"""Stage pinned raw games through the owner, then prepare on a RAM-rich peer."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
PLAN_SHA = '9270f03a2080e19e3036d30d3d184dd6d56fdfeb4309824f87c64046350bc5f6'
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       'go-user@worker-3.example.invalid']
PYTHON = '/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote(code, raw=b''):
    return json.loads(subprocess.check_output(SSH + ['taskset -c 0,1 ' + PYTHON + ' -c ' + shlex.quote(code)],
                                             input=raw, timeout=120))


RECEIVER = '''import hashlib,io,json,os,sys,tarfile
from pathlib import Path
folder=Path(FOLDER);expected=EXPECTED
folder.mkdir(parents=True,exist_ok=True)
fs=os.statvfs('/dev/shm')
if fs.f_bavail*fs.f_frsize<80*(1<<30):raise ValueError('Receiver RAM floor')
raw=sys.stdin.buffer.read()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
 members=archive.getmembers()
 if {m.name for m in members}!=set(expected) or len(members)!=len(expected):raise ValueError('Archive coverage differs')
 for m in members:
  if not m.isfile() or Path(m.name).name!=m.name:raise ValueError('Unexpected input path')
  data=archive.extractfile(m).read()
  if hashlib.sha256(data).hexdigest()!=expected[m.name]:raise ValueError('Input changed')
  path=folder/m.name
  if path.exists():
   if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[m.name]:raise ValueError('Existing input differs')
  else:
   with path.open('xb') as f:f.write(data)
   path.chmod(0o444)
print(json.dumps(dict(files=len(expected),verified=True)))
'''


def main():
    plan_path = STUDY / 'cohort-plan-002.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('Plan changed')
    plan = json.loads(plan_path.read_text())
    if sha(STUDY / 'prepare_cohort.py') != plan['operator_sha256']:
        raise ValueError('Packer changed')
    result_path = STUDY / 'cohort-staging-002.json'
    if result_path.exists():
        raise FileExistsError('Staging already completed')
    spec = importlib.util.spec_from_file_location('pack', STUDY / 'prepare_cohort.py')
    pack = importlib.util.module_from_spec(spec); spec.loader.exec_module(pack)
    raw_root = '/dev/shm/gozero-cohort-inputs/' + PLAN_SHA
    started = time.time()
    for offset in range(0, len(plan['records']), 32):
        rows = plan['records'][offset:offset+32]
        raw = pack.fetch(rows)
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as archive:
            for name, data in sorted(raw.items()):
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        expected = {r['game_id'] + '.npz': r['sha256'] for r in rows}
        result = remote(RECEIVER.replace('FOLDER', repr(raw_root)).replace('EXPECTED', repr(expected)), stream.getvalue())
        if result != dict(files=len(rows), verified=True):
            raise ValueError('Receiver verification differs')
        print(json.dumps(dict(kind='cohort_inputs_staged', games=offset+len(rows))), flush=True)
    files = [p for p in (ROOT / 'packages/gozero/src/gozero').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [STUDY / n for n in ('prepare_cohort.py', 'qualify_features.py', 'inventory-005.json', 'cohort-plan-002.json')]
    files += [ROOT / '.gozero/build/katago-v7-12e944b6' / n for n in ('receipt.json', 'feature_worker')]
    entries = {str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files}
    archive_path = ROOT / '.gozero/cohort-prepare-002.tar'
    with tarfile.open(archive_path, 'x') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    digest = sha(archive_path)
    base = '/workspace/go/.gozero/cohort-staging/' + digest
    code = '''import hashlib,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
entries=ENTRIES;base=Path(BASE);raw=sys.stdin.buffer.read()
if hashlib.sha256(raw).hexdigest()!=DIGEST:raise ValueError('Bundle changed')
base.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
 members=archive.getmembers()
 if {m.name for m in members}!=set(entries) or len(members)!=len(entries):raise ValueError('Coverage differs')
 for m in members:
  if not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:raise ValueError('Unsafe member')
  data=archive.extractfile(m).read();e=entries[m.name]
  if len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:raise ValueError('Source changed')
  path=base/m.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o555 if path.name=='feature_worker' else 0o444)
python='/workspace/go/.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'
argv=['taskset','-c','0,1',python,'-B',str(base/'research/studies/strong19_scaling/prepare_cohort.py'),'prepare','--plan',str(base/'research/studies/strong19_scaling/cohort-plan-002.json'),'--plan-sha256',PLAN,'--output',str(base/'cohort-result.json'),'--raw-inputs',RAW_ROOT,'--reserved-bytes','73716960384']
with (base/'prepare.log').open('xb') as log:
 p=subprocess.Popen(argv,cwd=base,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1'))
identity=Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
receipt=dict(kind='scientific19_cohort_preparation_process',host_index=3,pid=p.pid,process_identity=identity,started=time.time(),base=str(base),source_sha256=DIGEST,plan_sha256=PLAN)
(base/'process.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    for key, value in [('ENTRIES', entries), ('BASE', base), ('DIGEST', digest), ('PLAN', PLAN_SHA), ('RAW_ROOT', raw_root)]:
        code = code.replace(key, repr(value))
    receipt = remote(code, archive_path.read_bytes())
    value = dict(kind='scientific19_cohort_staging', status='passed', started=started, finished=time.time(),
                 source_sha256=digest, source_files=entries, plan_sha256=PLAN_SHA, raw_input_files=len(plan['records']),
                 raw_input_bytes=sum(r['bytes'] for r in plan['records']), process=receipt,
                 source_operator_sha256=sha(Path(__file__)))
    with result_path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    with (STUDY / 'cohort-process-002.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status='passed', process=receipt)), flush=True)


if __name__ == '__main__':
    main()
