"""One host supervisor: pinned workers, restart backoff, explicit stop and status."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..runtime import pin
from ..storage import _process_identity
from .corpus import atomic_json
from . import handover


def clean_engines(output):
    path = output / 'engine-processes.json'
    if not path.exists():
        return
    for record in json.loads(path.read_text()):
        if _process_identity(record['pid']) == record['identity']:
            try:
                os.killpg(record['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


def serve(config_path: Path, entry: Path):
    config = json.loads(config_path.read_text())
    previous=config.get('handover')
    if (config['run_id']!='expert19-v2' or not previous or previous['previous_run']!='expert19-v1'
            or previous['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or set(previous['workers'])!=set(map(str,range(len(config['workers']))))):
        raise ValueError('The v2 supervisor requires an explicit predecessor handover')
    previous_stop=Path(config['storage_root'])/'runs'/previous['previous_run']/'stop'
    if not previous_stop.is_file():raise ValueError('The predecessor is still admitting games')
    run = Path(config['storage_root']) / 'runs' / config['run_id']
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pin(config['research_cpus'][:2])
    atomic_json(run / 'supervisor.json', dict(pid=os.getpid(), started=time.time(), state='starting'))
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
        if not (run/'stop').exists():atomic_json(run/'stop',dict(reason='supervisor signal',unix=time.time()))
        (run / 'stop').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    workers = {}
    failures = {i: 0 for i in range(len(config['workers']))}
    retry_after = {i: 0 for i in failures}
    try:
        while not stopping and not (run / 'stop').exists():
            if not previous_stop.is_file():raise ValueError('Predecessor drain request disappeared')
            waiting=[]
            for index, worker in enumerate(config['workers']):
                if index in workers:
                    process, log = workers[index]
                    if process.poll() is None:
                        continue
                    log.close()
                    clean_engines(run / f'worker-{index:02d}')
                    failures[index] += 1
                    retry_after[index] = time.time() + min(300, 5 * 2**min(failures[index], 6))
                    with (run / 'failures.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(worker=index, returncode=process.returncode, unix=time.time(),
                                                     failures=failures[index])) + '\n')
                    del workers[index]
                if time.time() < retry_after[index] or failures[index] >= 5:
                    continue
                if not handover.lane_clear(previous,index):
                    waiting.append(index)
                    continue
                output = run / f'worker-{index:02d}'
                output.mkdir(parents=True, exist_ok=True)
                log = (output / 'worker.log').open('a')
                # The supervisor's small affinity must not become the workers' allocation.
                argv = ['taskset', '-c', ','.join(map(str, worker['cpus'])), sys.executable,
                        str(entry), 'worker', str(config_path), str(index)]
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                    start_new_session=True)
                workers[index] = process, log
            released=handover.release_previous(previous,run/'previous-supervisor-released.json')
            atomic_json(run / 'supervisor.json', dict(pid=os.getpid(), updated=time.time(), state='running',
                        workers={i:p.pid for i,(p,_) in workers.items()}, failures=failures,
                        waiting_previous=waiting,previous_supervisor_released=released,
                        quarantined=[i for i,n in failures.items() if n >= 5]))
            time.sleep(5)
    finally:
        # A stop file drains in-flight batches. If explicitly terminated, bound shutdown.
        deadline = time.time() + config.get('drain_seconds', 240)
        if not (run/'stop').exists():atomic_json(run/'stop',dict(reason='supervisor exit',unix=time.time()))
        while (any(p.poll() is None for p,_ in workers.values()) or not handover.all_clear(previous)) and time.time() < deadline:
            handover.release_previous(previous,run/'previous-supervisor-released.json')
            time.sleep(2)
        for index, (process, log) in workers.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
            clean_engines(run / f'worker-{index:02d}')
        released=handover.release_previous(previous,run/'previous-supervisor-released.json')
        atomic_json(run / 'supervisor.json', dict(pid=os.getpid(), updated=time.time(), state='stopped',
                    previous_supervisor_released=released))
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve', 'worker'])
    parser.add_argument('config', type=Path)
    parser.add_argument('worker', type=int, nargs='?')
    args = parser.parse_args()
    if args.mode == 'worker':
        from .generate import run_worker
        run_worker(args.config, args.worker)
    else:
        serve(args.config, Path(sys.argv[0]).resolve())


if __name__ == '__main__':
    main()
