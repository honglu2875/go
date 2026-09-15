"""Bounded asynchronous KataGo analysis transport and causal request contract."""
from __future__ import annotations

from concurrent.futures import Future
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from ..gtp import action_to_vertex

RULES = dict(ko='POSITIONAL', scoring='AREA', suicide=True, tax='NONE', hasButton=False,
             whiteHandicapBonus='0', friendlyPassOk=False)


def model_path(root: Path, record: dict) -> Path:
    # KataGo selects the parser by suffix; historical text weights retain it.
    suffix = 'txt.gz' if record.get('url', record.get('source_url', '')).endswith('.txt.gz') else 'bin.gz'
    return root / 'artifacts' / record['sha256'] / ('model.' + suffix)


def query(actions, *, visits: int, size: int = 9, komi: float = 7.5) -> dict:
    if visits < 1:
        raise ValueError('At least one root visit is required')
    return dict(moves=[['B' if i % 2 == 0 else 'W', action_to_vertex(int(a), size)]
                       for i, a in enumerate(actions)], initialPlayer='B', rules=RULES,
                komi=komi, boardXSize=size, boardYSize=size, maxVisits=visits,
                includePolicy=True, includeNoResultValue=True)


def config_text(*, concurrency=4, eigen_threads=4, seed='go-corpus19-v1', size=19) -> str:
    if size not in (9, 19):
        raise ValueError('The corpus producer supports 9x9 and 19x19')
    values = dict(numAnalysisThreads=concurrency, numSearchThreadsPerAnalysisThread=1,
                  numEigenThreadsPerModel=eigen_threads, numNNServerThreadsPerModel=1,
                  nnMaxBatchSize=concurrency, nnCacheSizePowerOfTwo=16,
                  nnMutexPoolSizePowerOfTwo=12, maxBoardXSizeForNNBuffer=size,
                  maxBoardYSizeForNNBuffer=size, requireMaxBoardSize=True,
                  nnRandomize=False, reportAnalysisWinratesAs='WHITE',
                  ignorePreRootHistory=False, ignoreAllHistory=False,
                  rootSymmetryPruning=False, rootNumSymmetriesToSample=1,
                  rootNoiseEnabled=False, chosenMoveTemperature=0,
                  chosenMoveTemperatureEarly=0, conservativePass=False,
                  preventCleanupPhase=False, playoutDoublingAdvantage=0,
                  nnRandSeed=seed, logToStderr=True)
    return ''.join(f'{key} = {str(value).lower() if isinstance(value, bool) else value}\n'
                   for key, value in values.items())


class AnalysisClient:
    def __init__(self, binary: Path, model: Path, output: Path, cpus, *, concurrency=4,
                 eigen_threads=4, max_pending=16, seed='go-corpus19-v1', size=19):
        output.mkdir(parents=True, exist_ok=True)
        config = output / 'analysis.cfg'
        config.write_text(config_text(concurrency=concurrency, eigen_threads=eigen_threads, seed=seed, size=size))
        self._stderr = (output / 'stderr.log').open('a')
        self._pending = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_pending)
        self._next_id = 0
        self._closed = False
        argv = ['taskset', '-c', ','.join(map(str, cpus)), str(binary), 'analysis',
                '-config', str(config), '-model', str(model)]
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._stderr, text=True, bufsize=1,
                                        start_new_session=True)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _fail_all(self, error):
        with self._lock:
            self._closed = True
            futures = list(self._pending.values())
            self._pending.clear()
        for future in futures:
            future.set_exception(error)
            self._slots.release()

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line) > 4 * 1024 * 1024:
                    raise RuntimeError('KataGo response exceeds size bound')
                data = json.loads(line)
                if data.get('isDuringSearch', False):
                    continue
                with self._lock:
                    future = self._pending.pop(str(data.get('id')), None)
                if future is None:
                    raise RuntimeError('Unexpected KataGo response ID')
                self._slots.release()
                if 'error' in data or 'warning' in data:
                    future.set_exception(RuntimeError('KataGo rejected query: ' + json.dumps(data)))
                else:
                    future.set_result(data)
        except Exception as error:
            self._fail_all(error)
        finally:
            self._fail_all(RuntimeError('KataGo analysis connection closed'))

    def submit(self, request: dict, timeout: float = 120) -> Future:
        if not self._slots.acquire(timeout=timeout):
            raise TimeoutError('KataGo request queue is full')
        future = Future()
        with self._lock:
            if self._closed:
                self._slots.release()
                raise RuntimeError('KataGo analysis connection is closed')
            identifier = str(self._next_id)
            self._next_id += 1
            self._pending[identifier] = future
        try:
            with self._write_lock:
                self.process.stdin.write(json.dumps({**request, 'id': identifier}, allow_nan=False) + '\n')
                self.process.stdin.flush()
        except Exception as error:
            self._fail_all(error)
            raise
        return future

    def request(self, request: dict, timeout: float = 120):
        return self.submit(request, timeout=timeout).result(timeout=timeout)

    def close(self):
        self._fail_all(RuntimeError('KataGo client closed'))
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self._reader.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
