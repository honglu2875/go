"""Publish a cancellation request for one verified attempt; never signal a PID."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    args=p.parse_args();snapshot=args.snapshot.resolve();attempt=args.attempt.resolve()
    sys.path.insert(0,str(snapshot/'packages/gozero/src'))
    from gozero.snapshots import canonical_json,verify
    manifest=verify(snapshot)
    if attempt.parent.name!='runs' or not re.fullmatch(r'pod-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}',attempt.name):
        raise ValueError('Invalid attempt path')
    if snapshot.parent.name!='snapshots' or attempt.parent.parent!=snapshot.parents[2]:
        raise ValueError('Attempt and source must share the artifact root')
    attempt.mkdir(parents=True,exist_ok=True)
    value={'schema_version':1,'kind':'cancel_attempt','snapshot_id':manifest['snapshot_id'],
           'attempt_id':attempt.name,'requested_unix':time.time()}
    fd,name=tempfile.mkstemp(prefix='.cancel-',dir=attempt)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(canonical_json(value));stream.flush();os.fsync(stream.fileno())
        os.replace(name,attempt/'cancel.json')
    finally:
        Path(name).unlink(missing_ok=True)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
