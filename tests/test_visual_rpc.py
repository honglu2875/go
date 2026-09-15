"""Concurrent request ownership, identity rejection and bounded RPC framing."""
from concurrent.futures import ThreadPoolExecutor
import socket
import struct
import tempfile
from pathlib import Path
import threading
import time
import unittest
import numpy as np
from gozero.visual_rpc import Client, Server, receive


class Runner:
    slots = 2
    def __init__(self):
        self.seen = []
    def score(self, requests):
        self.seen.append(requests)
        return {slot: {'expert_logits': np.array([len(h), slot], np.float32),
                       'behavior_logits': np.array([slot, len(h)], np.float32), 'value': 0.}
                for slot, (h, _) in requests.items()}


class RpcTests(unittest.TestCase):
    def test_concurrent_clients_retain_slots_and_order(self):
        with tempfile.TemporaryDirectory(prefix='visual-rpc-') as directory:
            runner = Runner(); identity = {'model': 'pinned'}
            server = Server(Path(directory) / 's', identity, runner, gather_seconds=.01)
            try:
                with self.assertRaises(ValueError):
                    Client(server.path, {'model': 'wrong'})
                barrier = threading.Barrier(2)
                def work(length):
                    client = Client(server.path, identity)
                    try:
                        barrier.wait()
                        first = client.predict([0] * length, '0' * 64)
                        second = client.predict([], '0' * 64)
                        return client.slot, first, second
                    finally:
                        client.close()
                with ThreadPoolExecutor(2) as pool:
                    futures = [pool.submit(work, n) for n in (1, 2)]
                    deadline = time.monotonic() + 5
                    while not all(f.done() for f in futures):
                        self.assertLess(time.monotonic(), deadline)
                        server.pump()
                    results = [f.result() for f in futures]
                self.assertEqual({r[0] for r in results}, {0, 1})
                for n, (slot, first, second) in enumerate(results, 1):
                    self.assertEqual(first['expert_logits'], [n, slot])
                    self.assertEqual(second['expert_logits'], [0, slot])
                    self.assertEqual((first['id'], second['id']), (1, 2))
                self.assertTrue(any(len(batch) == 2 for batch in runner.seen))
            finally:
                server.close()
            self.assertFalse(server.path.exists())

    def test_oversize_frame_is_rejected_before_payload(self):
        a, b = socket.socketpair()
        try:
            a.sendall(struct.pack('!I', 65537))
            with self.assertRaises(ValueError):
                receive(b)
        finally:
            a.close(); b.close()


if __name__ == '__main__':
    unittest.main()
