"""Bounded qualification: kill one identified trainer after a committed checkpoint."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--checkpoint-turn',type=int,required=True);p.add_argument('--timeout',type=float,default=60.)
    args=p.parse_args();snapshot=args.snapshot.resolve();attempt=args.attempt.resolve()
    if not 0<args.timeout<=120 or not 1<=args.checkpoint_turn<=1_000_000:raise ValueError('Bounded injection required')
    if (attempt.parent.name!='runs' or not re.fullmatch(r'pod-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}',attempt.name)
            or snapshot.parent.name!='snapshots' or attempt.parent.parent!=snapshot.parents[2]):
        raise ValueError('Attempt and snapshot must share the artifact root')
    sys.path.insert(0,str(snapshot/'packages/gozero/src'))
    from gozero.pod import load_hosts
    from gozero.checkpoints import read,sha256
    from gozero.snapshots import canonical_json,read_json,verify
    manifest=verify(snapshot)
    host=next(h for h in load_hosts(snapshot/'ops/hosts.json') if h.hostname==socket.gethostname().split('.')[0])
    rank=attempt/f'rank-{host.rank}';checkpoint=rank/'artifacts/checkpoints'/f'turn-{args.checkpoint_turn:09d}'
    group_path=checkpoint.parent/(checkpoint.name+'.group.json');deadline=time.monotonic()+args.timeout
    while not group_path.exists():
        if (rank/'result.json').exists():raise RuntimeError('Trainer ended before injection boundary')
        if time.monotonic()>=deadline:raise TimeoutError('No committed checkpoint before injection deadline')
        time.sleep(.01)
    group=read_json(group_path);state,_,_=read(checkpoint)
    if (group['snapshot_id']!=manifest['snapshot_id'] or state['snapshot_id']!=manifest['snapshot_id']
            or group['turn']!=args.checkpoint_turn or state['turn']!=args.checkpoint_turn
            or sha256(checkpoint/'manifest.json')!=group['rank_manifests'][state['jax_rank']]):
        raise ValueError('Checkpoint group identity differs')
    start=read_json(rank/'start.json');process=read_json(rank/'process.json')
    if start['snapshot_id']!=manifest['snapshot_id'] or str(snapshot/manifest['recipe']/'train.py') not in start['command']:
        raise ValueError('Trainer source identity differs')
    if (rank/'result.json').exists():raise RuntimeError('Trainer already finished')
    pid=process['pid']
    if type(pid) is not int or pid<=1 or process['pgid']!=pid:raise ValueError('Invalid private trainer process')
    fd=os.pidfd_open(pid)
    try:
        observed=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if observed[19]!=process['start_ticks'] or int(observed[2])!=pid or int(observed[3])!=pid:
            raise ValueError('Recorded trainer process was replaced or changed session')
        receipt={'schema_version':1,'kind':'checkpoint_boundary_failure_injection','status':'sent',
                 'snapshot_id':manifest['snapshot_id'],'attempt_id':attempt.name,'host_rank':host.rank,
                 'checkpoint_turn':args.checkpoint_turn,'group_sha256':sha256(group_path),'pid':pid,
                 'start_ticks':process['start_ticks'],'signal':'SIGKILL','sent_unix':time.time()}
        signal.pidfd_send_signal(fd,signal.SIGKILL)
        with (rank/'injected_failure.json').open('xb') as stream:stream.write(canonical_json(receipt))
        print(json.dumps(receipt),flush=True)
    finally:os.close(fd)


if __name__=='__main__':main()
