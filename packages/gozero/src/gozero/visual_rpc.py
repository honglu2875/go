"""Bounded local transport for native GTP clients and one batched JAX owner.

Sockets are private Unix sockets. A connection owns one cache slot and may have
only one outstanding request. Model/config identity is checked before allocation.
The main thread alone advances model state; socket threads never call JAX.
"""
import json
import os
from pathlib import Path
import queue
import socket
import socketserver
import struct
import threading
import time


LIMIT = 65536


def receive(stream):
    def exact(n):
        chunks = []
        while n:
            data = stream.recv(n)
            if not data:
                raise EOFError('Inference connection closed')
            chunks.append(data); n -= len(data)
        return b''.join(chunks)
    n = struct.unpack('!I', exact(4))[0]
    if not 0 < n <= LIMIT:
        raise ValueError('Inference message exceeds bound')
    def invalid(value):
        raise ValueError('Nonfinite RPC JSON: ' + value)
    return json.loads(exact(n), parse_constant=invalid)


def send(stream, value):
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if not 0 < len(raw) <= LIMIT:
        raise ValueError('Inference response exceeds bound')
    stream.sendall(struct.pack('!I', len(raw)) + raw)


class Client:
    def __init__(self, path, identity, *, timeout=60.):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout); self.counter = 0
        try:
            self.socket.connect(str(path)); send(self.socket, {'hello': identity})
            hello = receive(self.socket)
            if hello.get('identity') != identity or hello.get('status') != 'ready':
                raise ValueError('Inference owner rejected identity: ' + repr(hello))
            self.slot = hello['slot']
        except BaseException:
            self.close(); raise

    def predict(self, history, observation_sha256):
        self.counter += 1
        send(self.socket, {'id': self.counter, 'history': history, 'observation_sha256': observation_sha256})
        response = receive(self.socket)
        if response.get('id') != self.counter or response.get('status') != 'ok':
            raise RuntimeError('Inference failed: ' + repr(response))
        return response

    def close(self):
        self.socket.close()


class Server:
    def __init__(self, path, identity, runner, *, gather_seconds=.001, request_timeout=60.):
        path = Path(path)
        if path.exists() or path.is_symlink() or len(os.fsencode(path)) >= 104:
            raise ValueError('Inference socket path exists or is too long')
        if not 0 <= gather_seconds <= .1 or not 0 < request_timeout <= 300:
            raise ValueError('Invalid inference service latency bounds')
        self.runner, self.identity, self.path = runner, identity, path
        self.gather_seconds, self.timeout = gather_seconds, request_timeout
        self.pending = queue.Queue(maxsize=runner.slots); self.free = queue.Queue()
        for slot in range(runner.slots):
            self.free.put(slot)
        self.stopping = threading.Event(); self.records = []
        owner = self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(owner.timeout); slot = None
                try:
                    if receive(self.request) != {'hello': owner.identity}:
                        send(self.request, {'status': 'error', 'error': 'Model or inference identity differs'}); return
                    slot = owner.free.get_nowait()
                    send(self.request, {'status': 'ready', 'identity': owner.identity, 'slot': slot})
                    expected = 1
                    while not owner.stopping.is_set():
                        request = receive(self.request)
                        if (set(request) != {'id', 'history', 'observation_sha256'}
                                or type(request['id']) is not int or request['id'] != expected):
                            raise ValueError('Invalid request sequence or fields')
                        expected += 1
                        task = {'slot': slot, 'request': request, 'ready': threading.Event(), 'enqueued': time.perf_counter()}
                        owner.pending.put_nowait(task)
                        # Do not return the slot to another connection while its
                        # earlier task might still mutate the associated cache.
                        while not task['ready'].wait(.1):
                            if owner.stopping.is_set():
                                return
                        send(self.request, task['response'])
                except (EOFError, OSError):
                    pass
                except Exception as error:
                    try:
                        send(self.request, {'status': 'error', 'error': repr(error)})
                    except OSError:
                        pass
                finally:
                    if slot is not None:
                        owner.free.put_nowait(slot)
        class UnixServer(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
            block_on_close = False
        self.server = UnixServer(str(path), Handler); path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        self.thread.start()

    def pump(self, wait_seconds=.02):
        try:
            first = self.pending.get(timeout=wait_seconds)
        except queue.Empty:
            return 0
        tasks = [first]; deadline = time.perf_counter() + self.gather_seconds
        while len(tasks) < self.runner.slots:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                tasks.append(self.pending.get(timeout=remaining))
            except queue.Empty:
                break
        started = time.perf_counter()
        if len({task['slot'] for task in tasks}) != len(tasks):
            raise RuntimeError('Concurrent requests reused a cache slot')
        try:
            result = self.runner.score({task['slot']: (task['request']['history'], task['request']['observation_sha256']) for task in tasks})
            finished = time.perf_counter()
            for task in tasks:
                prediction = result[task['slot']]
                task['response'] = {'id': task['request']['id'], 'status': 'ok',
                    'expert_logits': prediction['expert_logits'].tolist(),
                    'behavior_logits': prediction['behavior_logits'].tolist(), 'value': float(prediction['value'])}
            if len(self.records) >= 1000000:
                raise RuntimeError('Inference telemetry bound reached')
            self.records.append({'batch': len(tasks), 'service_seconds': finished - started,
                                 'queue_seconds': [started - t['enqueued'] for t in tasks]})
        except BaseException as error:
            for task in tasks:
                task['response'] = {'id': task['request']['id'], 'status': 'error', 'error': repr(error)}
            raise
        finally:
            for task in tasks:
                task['ready'].set()
        return len(tasks)

    def close(self):
        self.stopping.set(); self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=1)
        self.path.unlink(missing_ok=True)
