from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from gozero.joint_rpc import Client,Server
from gozero.visual_rpc import Client as LegacyClient


class Runner:
    slots=2
    size=1
    def __init__(self):self.requests=[]
    def score(self,requests):
        self.requests.append(requests)
        return {slot:dict(policy=np.array([len(h),slot],np.float32),value_logits=np.zeros(3,np.float32),value=np.float32(0))
                for slot,(h,_) in requests.items()}


class JointRpcTests(unittest.TestCase):
    def test_concurrent_single_policy_clients_preserve_state_digests_and_slots(self):
        with tempfile.TemporaryDirectory(prefix='joint-rpc-') as directory:
            runner=Runner();identity={'candidate':'pinned'}
            server=Server(Path(directory)/'s',identity,runner,gather_seconds=.01)
            try:
                with self.assertRaises(ValueError):Client(server.path,{'candidate':'wrong'})
                with self.assertRaises(ValueError):LegacyClient(server.path,identity)
                barrier=threading.Barrier(2)
                def work(n):
                    client=Client(server.path,identity)
                    try:
                        barrier.wait();first=client.predict([0]*n,str(n)*64);second=client.predict([],'0'*64)
                        return client.slot,first,second
                    finally:client.close()
                with ThreadPoolExecutor(2) as pool:
                    futures=[pool.submit(work,n) for n in (1,2)]
                    deadline=time.monotonic()+5
                    while not all(f.done() for f in futures):
                        self.assertLess(time.monotonic(),deadline);server.pump()
                    results=[f.result() for f in futures]
                self.assertEqual({r[0] for r in results},{0,1})
                for n,(slot,first,second) in enumerate(results,1):
                    self.assertEqual(set(first),{'id','status','policy','value_logits','value'})
                    self.assertEqual(first['policy'],[n,slot]);self.assertEqual(second['policy'],[0,slot])
                    self.assertEqual((first['id'],second['id']),(1,2))
                self.assertTrue(any(len(requests)==2 for requests in runner.requests))
                for requests in runner.requests:
                    for history,digest in requests.values():
                        self.assertEqual(digest,str(len(history))*64)
            finally:server.close()

    def test_shape_extra_head_and_value_perspective_are_checked(self):
        with tempfile.TemporaryDirectory(prefix='joint-rpc-') as directory:
            server=Server(Path(directory)/'s',{'candidate':'pinned'},Runner())
            try:
                good=dict(policy=np.zeros(2),value_logits=np.zeros(3),value=np.float32(0))
                server.prediction(good)
                bad=[{**good,'policy':np.zeros(3)},{**good,'behavior_logits':np.zeros(2)},
                     {**good,'value':np.float32(.5)},{**good,'value_logits':np.full(3,np.nan)}]
                for value in bad:
                    with self.assertRaises(ValueError):server.prediction(value)
            finally:server.close()


if __name__=='__main__':unittest.main()
