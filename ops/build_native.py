#!/usr/bin/env python3
"""Build the Python extension from an immutable source snapshot and attest it."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def build(args):
    snapshot=args.snapshot.resolve();verify(snapshot)
    root=args.artifacts_root.resolve();destination=root/'.gozero/native'/snapshot.name
    if destination.exists():
        receipt=json.loads((destination/'receipt.json').read_text())
        if receipt['snapshot_id']!=snapshot.name or digest(destination/receipt['filename'])!=receipt['binary_sha256']:
            raise ValueError('Existing native artifact is inconsistent')
        print(destination/'receipt.json');return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.build-',dir=destination.parent))
    environment={**os.environ,'RUSTUP_HOME':str(root/'.gozero/rustup'),'CARGO_HOME':str(root/'.gozero/cargo'),
                 'PYO3_PYTHON':sys.executable}
    cargo=root/'.gozero/cargo/bin/cargo';target=root/'.gozero/build/native'
    command=[str(cargo),'build','--release','--locked','--offline','-p','go-bridge',
             '--manifest-path',str(snapshot/'Cargo.toml'),'--target-dir',str(target)]
    start=time.time()
    try:
        with (temporary/'build.log').open('w') as log:
            subprocess.run(command,env=environment,cwd=snapshot,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        binary=temporary/'lib_gozero_native.so';shutil.copyfile(target/'release/lib_gozero_native.so',binary)
        receipt={'schema_version':1,'snapshot_id':snapshot.name,'filename':binary.name,'binary_sha256':digest(binary),
                 'cargo_lock_sha256':digest(snapshot/'Cargo.lock'),'command':command,
                 'rust_version':subprocess.check_output([str(root/'.gozero/cargo/bin/rustc'),'--version'],env=environment,cwd=snapshot,text=True).strip(),
                 'python':sys.version.split()[0],'abi':'CPython abi3 >= 3.12','started_unix':start,'finished_unix':time.time()}
        verify(snapshot)
        (temporary/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        binary.chmod(0o444);os.rename(temporary,destination)
    except BaseException:
        # Failed build logs are retained with a new attempt identity.
        print('Native build failed; log retained at '+str(temporary/'build.log'),file=sys.stderr)
        raise
    print(destination/'receipt.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--artifacts-root',type=Path,required=True)
    build(parser.parse_args())
