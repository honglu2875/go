"""Numbered, deadline-bounded GTP subprocess transport with retained transcripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import subprocess
import threading
import time


class GTPError(RuntimeError):
    pass


class GTPCommandError(GTPError):
    """The engine returned a well-formed rejection; the connection is usable."""


class GTPClient:
    def __init__(self, argv: list[str], output: Path, *, cwd: Path | None = None, environment=None):
        output.mkdir(parents=True, exist_ok=False)
        self._stderr = (output / 'stderr.log').open('w')
        self._transcript = (output / 'gtp.jsonl').open('w')
        self._closed = threading.Event()
        self._lines = queue.Queue(maxsize=1024)
        self._lock = threading.Lock()
        self._next_id = 1
        self._broken = False
        try:
            self.process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._stderr, text=True, encoding='utf-8', errors='replace',
                bufsize=1, start_new_session=True)
        except Exception:
            self._stderr.close()
            self._transcript.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        self._record({'kind': 'start', 'argv': argv, 'pid': self.process.pid})
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _record(self, record):
        self._transcript.write(json.dumps({'unix_time': time.time(), **record}, sort_keys=True) + '\n')
        self._transcript.flush()

    def _enqueue(self, line):
        while not self._closed.is_set():
            try:
                self._lines.put(line, timeout=0.1)
                return
            except queue.Full:
                pass

    def _read_stdout(self):
        try:
            while not self._closed.is_set():
                line = self.process.stdout.readline(65537)
                if not line:
                    break
                if len(line) > 65536:
                    self._enqueue(GTPError('Engine stdout line exceeded 64 KiB'))
                    return
                self._enqueue(line.rstrip('\r\n'))
        finally:
            self._enqueue(None)

    def _line(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('GTP response deadline expired')
        try:
            value = self._lines.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError('GTP response deadline expired') from error
        if value is None:
            raise GTPError('Engine stdout closed before the response completed')
        if isinstance(value, Exception):
            raise value
        return value

    def command(self, text: str, *, timeout: float = 30.0) -> str:
        if not text.strip() or any(c in text for c in ('\n', '\r', '\0')) or timeout <= 0:
            raise ValueError('GTP commands must be one nonempty line with a positive timeout')
        with self._lock:
            if self._closed.is_set() or self._broken:
                raise GTPError('GTP connection is closed or unusable')
            identifier = self._next_id
            self._next_id += 1
            start = time.monotonic()
            self._record({'kind': 'command', 'id': identifier, 'text': text})
            try:
                deadline = start + timeout
                pending = memoryview(('%d %s\n' % (identifier, text)).encode('utf-8'))
                descriptor = self.process.stdin.fileno()
                while pending:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('GTP command write deadline expired')
                    if not select.select([], [descriptor], [], remaining)[1]:
                        raise TimeoutError('GTP command write deadline expired')
                    try:
                        pending = pending[os.write(descriptor, pending):]
                    except BlockingIOError:
                        continue
                header = self._line(deadline)
                while not header.strip() or header.lstrip().startswith('#'):
                    header = self._line(deadline)
                match = re.fullmatch(r'([=?])\s*(\d+)(?:[ \t]+(.*))?', header)
                if not match or int(match.group(2)) != identifier:
                    raise GTPError('Malformed or mismatched GTP response header: %r' % header)
                lines = [match.group(3)] if match.group(3) else []
                while True:
                    line = self._line(deadline)
                    if not line.strip():
                        break
                    lines.append(line)
                    if len(lines) > 4096:
                        raise GTPError('GTP response exceeded 4096 lines')
                response = '\n'.join(lines)
                self._record({'kind': 'response', 'id': identifier, 'success': match.group(1) == '=',
                              'text': response, 'elapsed_seconds': time.monotonic()-start})
                if match.group(1) == '?':
                    raise GTPCommandError(response)
                return response
            except GTPCommandError:
                raise
            except Exception as error:
                self._broken = True
                self._record({'kind': 'transport_error', 'id': identifier, 'error': repr(error)})
                raise

    def close(self):
        if self._closed.is_set():
            return
        if not self._broken and self.process.poll() is None:
            try:
                self.command('quit', timeout=2)
            except Exception:
                pass
        self._closed.set()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
        self._reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()
        self._record({'kind': 'exit', 'returncode': self.process.returncode})
        self._stderr.close()
        self._transcript.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
