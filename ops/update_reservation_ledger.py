#!/usr/bin/env python3
"""Extend the observed allocation ledger, preserving its previous immutable bytes."""
import argparse
from datetime import datetime,timezone
import fcntl
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json,verify


def require(value,message):
    if not value:raise ValueError(message)


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def update(root,path,expected):
    require(sha256(path)==expected,'Previous ledger hash differs')
    previous=read_json(path);now=time.time()
    require(previous['kind']=='known_reservation_window' and previous['schema_version']==1
            and previous['window_start_unix']<=previous['window_end_unix']<now
            and type(previous['reserved_chips'])is int and previous['reserved_chips']>0,'Unsupported prior ledger')
    closed=[];open_attempts=[]
    for launch_path in sorted((root/'runs').glob('pod-*/launch.json')):
        launch=read_json(launch_path);attempt=launch_path.parent.name
        require(launch['attempt_id']==attempt and launch['reserved_chips']==previous['reserved_chips']
                and previous['window_start_unix']<=launch['start_unix_time']<now,'Attempt identity, allocation or window differs')
        result_path=launch_path.parent/'result.json'
        if not result_path.exists():
            open_attempts.append({'attempt':attempt,'launch_sha256':sha256(launch_path),
                'start_unix_time':launch['start_unix_time'],'observed_elapsed_seconds':now-launch['start_unix_time']})
            continue
        result=read_json(result_path)
        require(result['attempt_id']==attempt and result['snapshot_id']==launch['snapshot_id']
                and result['status']in ('passed','failed') and result['start_unix_time']==launch['start_unix_time']
                and result['start_unix_time']<=result['end_unix_time']<=now
                and result['reserved_chips']==launch['reserved_chips'],'Closed attempt identity or time differs')
        elapsed=result['end_unix_time']-result['start_unix_time']
        require(math.isclose(elapsed,result['elapsed_seconds'],rel_tol=0,abs_tol=1e-6)
                and math.isclose(elapsed*result['reserved_chips']/3600,result['reserved_chip_hours'],rel_tol=0,abs_tol=1e-9),'Attempt cost arithmetic differs')
        closed.append({'attempt':attempt,'reserved_attempt_chip_hours':result['reserved_chip_hours'],
            'result_sha256':sha256(result_path),'status':result['status']})
    by_name={r['attempt']:r for r in closed}
    require(all(by_name.get(r['attempt'])==r for r in previous['recorded_pod_attempts']),'Previously recorded attempt changed or disappeared')
    elapsed_hours=(now-previous['window_start_unix'])/3600
    history=path.parent/'ledger_history';history.mkdir(exist_ok=True)
    historical=history/f"ledger-{int(previous['window_end_unix'])}-{expected[:12]}.json"
    if historical.exists():require(sha256(historical)==expected,'Historical ledger path differs')
    else:
        with historical.open('xb')as f:
            f.write(path.read_bytes());f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
        sync_directory(history)
    current={**previous,'window_end_unix':now,'window_end_utc':datetime.fromtimestamp(now,timezone.utc).isoformat(),
        'elapsed_hours':elapsed_hours,'known_window_chip_hours':elapsed_hours*previous['reserved_chips'],
        'recorded_pod_attempts':closed,'observed_open_attempts':open_attempts,
        'summed_pod_attempt_window_chip_hours':sum(r['reserved_attempt_chip_hours']for r in closed),
        'operator_snapshot':SOURCE.name,'previous_ledger_sha256':expected,'previous_ledger':str(historical.relative_to(root))}
    verify(SOURCE);require(sha256(path)==expected,'Ledger changed before publication')
    fd,name=tempfile.mkstemp(prefix='.ledger-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb')as f:
            f.write(canonical_json(current));f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
        os.replace(name,path);sync_directory(path.parent)
    finally:
        if os.path.exists(name):os.unlink(name)
    print(canonical_json({'path':str(path.relative_to(root)),'sha256':sha256(path),
        'closed_attempts':len(closed),'open_attempts':len(open_attempts),
        'known_window_chip_hours':current['known_window_chip_hours'],
        'recorded_attempt_chip_hours':current['summed_pod_attempt_window_chip_hours']}).decode().strip())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--ledger',default='research/studies/runtime_qualification/reservation_ledger.json')
    p.add_argument('--expected-previous-sha256',required=True);a=p.parse_args();verify(SOURCE)
    root=a.workspace_root.resolve();path=artifact(root,a.ledger)
    with (path.parent/'.ledger-update.lock').open('a+b')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        update(root,path,a.expected_previous_sha256)


if __name__=='__main__':main()
