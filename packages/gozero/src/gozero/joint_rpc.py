"""Single-policy/value transport over the qualified bounded local RPC queue.

Connection/slot ownership and framing reuse visual_rpc. Its historical digest
field carries the canonical V7 native state digest here. A separate handshake
protocol prevents a legacy two-policy client from using this endpoint.
"""
import queue
import time

import numpy as np

from . import visual_rpc

PROTOCOL='joint-v7-policy-value-rpc-v1'


def bound_identity(identity):
    if not isinstance(identity,dict) or not identity:
        raise ValueError('An explicit joint inference identity is required')
    return dict(protocol=PROTOCOL,identity=identity)


class Client(visual_rpc.Client):
    def __init__(self,path,identity,*,timeout=60.):
        super().__init__(path,bound_identity(identity),timeout=timeout)

    def predict(self,history,state_digest):
        return super().predict(history,state_digest)


class Server(visual_rpc.Server):
    def __init__(self,path,identity,runner,**kwargs):
        if type(runner.size) is not int or not 1<=runner.size<=25:
            raise ValueError('Invalid joint RPC board size')
        super().__init__(path,bound_identity(identity),runner,**kwargs)

    def prediction(self,value):
        if set(value)!={'policy','value_logits','value'}:
            raise ValueError('Joint RPC requires exactly one policy and its value outputs')
        policy=np.asarray(value['policy']);logits=np.asarray(value['value_logits']);signed=np.asarray(value['value'])
        if (policy.shape!=(self.runner.size**2+1,) or logits.shape!=(3,) or signed.shape!=()
                or not all(np.isfinite(x).all() for x in (policy,logits,signed)) or abs(float(signed))>1.000001):
            raise ValueError('Invalid joint prediction payload')
        weights=np.exp(logits.astype(np.float64)-float(logits.max()));weights/=weights.sum()
        if abs(float(signed)-(weights[0]-weights[1]))>1e-6:
            raise ValueError('Value output disagrees with player-to-move logits')
        return dict(policy=policy.tolist(),value_logits=logits.tolist(),value=float(signed))

    def pump(self,wait_seconds=.02):
        try:first=self.pending.get(timeout=wait_seconds)
        except queue.Empty:return 0
        tasks=[first];deadline=time.perf_counter()+self.gather_seconds
        while len(tasks)<self.runner.slots:
            remaining=deadline-time.perf_counter()
            if remaining<=0:break
            try:tasks.append(self.pending.get(timeout=remaining))
            except queue.Empty:break
        started=time.perf_counter()
        if len({task['slot'] for task in tasks})!=len(tasks):
            raise RuntimeError('Concurrent joint requests reused a cache slot')
        try:
            result=self.runner.score({task['slot']:(task['request']['history'],task['request']['observation_sha256']) for task in tasks})
            finished=time.perf_counter()
            if set(result)!={task['slot'] for task in tasks}:
                raise ValueError('Joint prediction batch coverage differs')
            for task in tasks:
                task['response']=dict(id=task['request']['id'],status='ok',**self.prediction(result[task['slot']]))
            if len(self.records)>=1000000:raise RuntimeError('Inference telemetry bound reached')
            self.records.append(dict(batch=len(tasks),service_seconds=finished-started,
                                     queue_seconds=[started-task['enqueued'] for task in tasks]))
        except BaseException as error:
            for task in tasks:task['response']=dict(id=task['request']['id'],status='error',error=repr(error))
            raise
        finally:
            for task in tasks:task['ready'].set()
        return len(tasks)
