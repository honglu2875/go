#!/usr/bin/env python3
"""Build an auditable adapter against the already pinned KataGo object files."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); verify(SOURCE)
    root=args.workspace_root.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    revision='92ee95c0a4b25fec214da00951ab69e97e207729'
    external=root/'.gozero/external/katago-source'/revision/('KataGo-'+revision)
    build=root/'.gozero/build/katago-92ee95c0'
    objroot=build/'CMakeFiles/katago.dir'
    names=['core/'+name+'.cpp.o' for name in ('global','hash','rand','rand_helpers','fancymath','sha2','fileutils','makedir','md5','bsearch','timer')]
    names+=['game/'+name+'.cpp.o' for name in ('board','rules','boardhistory','graphhash')]
    names+=['neuralnet/nninputs.cpp.o']
    objects=[objroot/name for name in names]
    if not all(x.is_file() for x in objects): raise ValueError('Pinned KataGo build is incomplete')
    source=Path(__file__).with_name('feature_worker.cpp')
    binary=args.output/'feature_worker'
    command=['/usr/bin/c++','-O3','-DNDEBUG','-std=c++17','-DNO_GIT_REVISION','-DNO_LIBZIP',
             '-DUSE_AVX2','-DUSE_EIGEN_BACKEND','-I'+str(external/'cpp'),
             str(source),*[str(x) for x in objects],'-lz','-lpthread','-o',str(binary)]
    result=subprocess.run(command,capture_output=True,text=True)
    (args.output/'build.stdout').write_text(result.stdout); (args.output/'build.stderr').write_text(result.stderr)
    receipt={'kind':'katago_v7_feature_worker','source_snapshot':SOURCE.name,'source_revision':revision,
        'source_sha256':sha256(source),'command':command,'returncode':result.returncode,
        'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),
        'objects':{str(x.relative_to(root)):sha256(x) for x in objects}}
    if result.returncode == 0:
        binary.chmod(0o555); receipt['binary_sha256']=sha256(binary)
    (args.output/'receipt.json').write_bytes(canonical_json(receipt))
    (args.output/'receipt.json').chmod(0o444)
    print(json.dumps({'returncode':result.returncode,'receipt_sha256':sha256(args.output/'receipt.json')}),flush=True)
    result.check_returncode()
if __name__ == '__main__': main()
