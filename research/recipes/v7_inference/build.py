"""Link the streaming adapter against hash-verified, pinned feature objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reference=ROOT/'.gozero/build/katago-v7-12e944b6/receipt.json'
    if sha(reference)!='9e7d8d4757c4214e25ea34a8ff3cbe57e1817abd7daf535117e0850eb0a6bd07':
        raise ValueError('Pinned feature-build receipt differs')
    receipt=json.loads(reference.read_text())
    for name,digest in receipt['objects'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Pinned object changed: '+name)
    a.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('feature_stream.cpp');copied=a.output/source.name
    copied.write_bytes(source.read_bytes());copied.chmod(0o444)
    original=receipt['command'];binary=a.output/'feature_stream'
    command=original[:]
    command[command.index(next(x for x in command if x.endswith('/feature_worker.cpp')))]=str(copied.resolve())
    command[command.index('-o')+1]=str(binary.resolve())
    before=time.time();result=subprocess.run(command,capture_output=True,text=True)
    (a.output/'build.stdout.log').write_text(result.stdout);(a.output/'build.stderr.log').write_text(result.stderr)
    data=dict(kind='katago_v7_stream_build',status='passed' if result.returncode==0 else 'failed',returncode=result.returncode,
              reference_receipt_sha256=sha(reference),source_sha256=sha(copied),objects=receipt['objects'],
              source_revision=receipt['source_revision'],command=command,seconds=time.time()-before,
              compiler=subprocess.check_output(['/usr/bin/c++','--version'],text=True),operator_sha256=sha(Path(__file__)))
    if result.returncode==0:binary.chmod(0o555);data['binary_sha256']=sha(binary)
    out=a.output/'receipt.json'
    with out.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    out.chmod(0o444);print(json.dumps(dict(status=data['status'],receipt=str(out),sha256=sha(out))),flush=True)
    result.check_returncode()


if __name__=='__main__':main()
