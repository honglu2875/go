"""Verify one host, then drain and transfer only its owned generation processes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def record(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rpartition(')')[2].split()
    return dict(pid=pid, identity=fields[19], state=fields[0], ppid=int(fields[1]))


def command(pid):
    return Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')[:-1]


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    path.chmod(0o444)


GUARD = '''import json,os,signal,sys,time
from pathlib import Path
config=Path(sys.argv[1]);c=json.loads(config.read_text());h=c['handover']
sys.path.insert(0,str(Path(c['producer_environment'])/'site-packages'))
from flygo.data.handover import all_clear,active
assert h['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
os.sched_setaffinity(0,{0,1})
run=config.parent
while not all_clear(h):
 if (run/'handover-aborted.json').exists():sys.exit(0)
 time.sleep(5)
if active(h['supervisor']):os.kill(h['supervisor']['pid'],signal.SIGCONT)
with (run/'drain-guard-released.json').open('x') as f:
 json.dump(dict(supervisor=h['supervisor'],released=time.time()),f)
'''


def main():
    payload = json.loads(sys.argv[1]); c = payload['config']
    if payload['mode'] not in ('verify', 'handover'):
        raise ValueError('Unknown operation')
    root = Path(c['storage_root']); env = Path(c['producer_environment'])
    if root != Path('/dev/shm/go-corpus19') or env != root/'environments'/c['producer_snapshot']:
        raise ValueError('Unexpected storage namespace')
    assert c['run_id'] == 'expert19-v2' and c['host_index'] in range(4)
    assert c['research_cpus'] == [0, 1] and c['drain_seconds'] == 86400
    assert c['storage_limits'] == dict(files_cap=24*(1 << 30), free_files_floor=64*(1 << 30),
                                      available_memory_floor=96*(1 << 30))
    snapshot = json.loads((env/'snapshot.json').read_text())
    assert snapshot['snapshot'] == c['producer_snapshot']
    assert snapshot['source_sha256'] == payload['qualified_sources']
    for name, digest in snapshot['source_sha256'].items():
        assert sha(env/'site-packages/flygo'/name) == digest, name
    native = next((env/'site-packages/flygo').glob('_native*.so'))
    assert sha(native) == snapshot['native_sha256'] == payload['native_sha256']
    sys.path.insert(0, str(env/'site-packages'))
    from flygo.data.handover import active
    from flygo.data.corpus import identity
    assert identity(c['contract']) == payload['contract_id']
    for model in [c['contract']['teacher'], *c['contract']['opponents']]:
        suffix = 'txt.gz' if model.get('url', '').endswith('.txt.gz') else 'bin.gz'
        assert sha(root/'artifacts'/model['sha256']/('model.'+suffix)) == model['sha256']
    binary = root/'artifacts'/c['contract']['engine_sha256']/'katago'
    assert sha(binary) == c['contract']['engine_sha256']
    s = os.statvfs(root)
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    resources = dict(shm_free=s.f_bavail*s.f_frsize, memory_available=memory['MemAvailable'])
    assert resources['shm_free'] > 68*(1 << 30) and resources['memory_available'] > 200*(1 << 30)
    old_run = root/'runs/expert19-v1'
    old_config_path = old_run/'config.json'
    old = json.loads(old_config_path.read_text())
    assert old['run_id'] == 'expert19-v1' and old['host_index'] == c['host_index']
    assert identity(old['contract']) == '60bf72d3961a31ee3f9bee0bada77aab6965cf1adcf03f1d825e1a16c964d545'
    old_env = root/'environments'/old['producer_snapshot']
    expected_supervisor = [str(old_env/'entry.py').encode(), b'serve', str(old_config_path).encode()]
    status = json.loads((old_run/'supervisor.json').read_text())
    supervisor = record(status['pid'])
    assert command(supervisor['pid'])[-3:] == expected_supervisor
    assert status['state'] == 'running' and all(v == 0 for v in status['failures'].values())
    assert not (old_run/'stop').exists(), 'An earlier handover needs inspection, not blind replay'
    run = root/'runs'/c['run_id']; run.mkdir(parents=True, exist_ok=True)
    assert not (run/'config.json').exists() and not (run/'supervisor.json').exists()
    assert not (run/'stop').exists()
    contract_path = run/'contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == c['contract']
    else:
        save(contract_path, c['contract'])
    result = dict(status='verified', host=c['host_index'], resources=resources,
                  old_supervisor=supervisor, producer_snapshot=c['producer_snapshot'],
                  verified_unix=time.time(), qualified_receipt_sha256=payload['qualification_sha256'])
    if payload['mode'] == 'verify':
        print(json.dumps(result)); return
    with (run/'deployment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert record(supervisor['pid'])['identity'] == supervisor['identity']
        paused = committed = False
        try:
            # Pause only the old supervisor. Its workers and their independent
            # engine process groups keep running and finish their existing batch.
            os.kill(supervisor['pid'], signal.SIGSTOP); paused = True
            for _ in range(100):
                if record(supervisor['pid'])['state'] == 'T': break
                time.sleep(.01)
            else: raise RuntimeError('Supervisor did not stop')
            status = json.loads((old_run/'supervisor.json').read_text())
            assert set(status['workers']) == set(map(str, range(8)))
            children = set(map(int, Path('/proc', str(supervisor['pid']), 'task',
                                         str(supervisor['pid']), 'children').read_text().split()))
            assert children == set(status['workers'].values()), 'Caught an unstable worker list; retry before draining'
            lanes = {}
            for i, start in enumerate((8, 16, 24, 32, 40, 48, 56, 64)):
                worker = c['workers'][i]
                assert worker['cpus'] == old['workers'][i]['cpus'] == list(range(start, start+8))
                assert worker['opponent_index'] == old['workers'][i]['opponent_index'] == i
                assert (worker['teacher_threads'], worker['opponent_threads']) == ((4, 4) if i == 7 else (7, 1))
                pid = status['workers'][str(i)]; current = record(pid)
                assert active(current) and current['ppid'] == supervisor['pid']
                assert command(pid)[-4:] == [str(old_env/'entry.py').encode(), b'worker', str(old_config_path).encode(), str(i).encode()]
                assert os.sched_getaffinity(pid) == set(worker['cpus'])
                engines = json.loads((old_run/f'worker-{i:02d}'/'engine-processes.json').read_text())
                assert len(engines) == 2
                processes = [current]
                for engine in engines:
                    proc = record(engine['pid'])
                    assert active(engine) and proc['ppid'] == pid
                    assert bytes(str(binary), 'utf-8') in command(engine['pid'])
                    assert os.sched_getaffinity(engine['pid']) == set(worker['cpus'])
                    processes.append(proc)
                lanes[str(i)] = dict(cpus=worker['cpus'], processes=processes)
            c['handover'] = dict(previous_run='expert19-v1',
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                supervisor=supervisor, workers=lanes, captured_unix=time.time(),
                original_supervisor_timeout_seconds=old['drain_seconds'])
            config_path = run/'config.json'; save(config_path, c)
            guard_path = run/'drain_guard.py'
            with guard_path.open('x') as f: f.write(GUARD)
            guard_path.chmod(0o444)
            with (run/'drain_guard.log').open('xb') as log:
                guard = subprocess.Popen([sys.executable, '-B', str(guard_path), str(config_path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # Persist the guard identity before requesting the irreversible drain
            # boundary, so recovery can identify every launched owned process.
            save(run/'drain-guard-launch.json', dict(**record(guard.pid), script_sha256=sha(guard_path), started=time.time()))
            save(old_run/'stop', dict(reason='continuous-v2 handover; preserve all in-flight games',
                                     successor_snapshot=c['producer_snapshot'], requested_unix=time.time()))
            committed = True
            with (run/'supervisor.log').open('xb') as log:
                process = subprocess.Popen([sys.executable, '-B', str(env/'entry.py'), 'serve', str(config_path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            result.update(status='handover_started', supervisor=record(process.pid),
                          guard=record(guard.pid), config=str(config_path), config_sha256=sha(config_path),
                          old_workers=len(lanes), started_unix=time.time())
            save(run/'launch.json', result)
            time.sleep(.5)
            assert process.poll() is None, 'New supervisor exited; inspect its log and recover using the persisted handover'
            print(json.dumps(result), flush=True)
        except BaseException:
            if paused and not committed:
                if active(supervisor): os.kill(supervisor['pid'], signal.SIGCONT)
                if not (run/'handover-aborted.json').exists():
                    save(run/'handover-aborted.json', dict(aborted_unix=time.time(), predecessor_resumed=True))
            # After the stop boundary, the independent guard still protects the
            # old games and releases their supervisor when its workers finish.
            raise


if __name__ == '__main__':
    main()
