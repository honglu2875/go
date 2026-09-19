"""One local JAX owner for batched native V7 leaf evaluation.

Rust owns search and complete histories. The C++ V7 worker validates tapes and
encodes suffixes. The temporal owner retains one exact path per slot, rewinds to
common prefixes, and appends bounded blocks in one graph. This is evaluation
reuse on known legal paths, not acceptance of predicted speculative branches.
"""
from collections import OrderedDict
import re
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
import numpy as np

from gozero.v7_replay import Replay,MAX_OUTPUT
from gozero.v7_state import validate_history,row_digest
import causal
import joint


def frozen_prediction(prediction):
    result={k:np.asarray(v).copy() for k,v in prediction.items()}
    if not all(np.isfinite(v).all() for v in result.values()) or abs(float(result['value']))>1.000001:
        raise FloatingPointError('Invalid model prediction')
    for value in result.values():value.setflags(write=False)
    return result


def publish(prediction):return {k:v.copy() for k,v in prediction.items()}


class Base:
    def __init__(self,params,c,replay,*,slots,network_version,max_block=16):
        if (type(slots) is not int or not 1<=slots<=128 or slots%len(jax.local_devices()) or
                type(network_version) is not int or not 0<=network_version<2**32 or
                type(max_block) is not int or max_block not in (1,2,4,8,16,32)):
            raise ValueError('Invalid inference slot/version/block dimensions')
        if not isinstance(replay,Replay) or replay.size!=c['max_board_size']:raise ValueError('Replay and model board size differ')
        self.c=dict(c);self.replay=replay;self.slots=slots;self.version=network_version
        self.capacity=c['max_positions'];self.size=replay.size;self.max_block=max_block;self.failed=False
        self.mesh=Mesh(np.asarray(jax.local_devices()),('data',))
        self.batched=NamedSharding(self.mesh,P('data'));self.replicated=NamedSharding(self.mesh,P())
        self.params=jax.device_put(params,self.replicated);self.compiled={};self.compilation={}
        self.stats=dict(requests=0,prefix_hits=0,dispatches=0,appended_positions=0,padded_position_slots=0,
                        replay_calls=0,native_encoded_positions=0,native_validated_history_moves=0,
                        replay_seconds=0.,dispatch_seconds=0.)

    def put(self,x):return jax.device_put(x,self.batched)

    def validate(self,requests):
        if self.failed:raise RuntimeError('Inference owner failed; construct a fresh owner before reuse')
        if not isinstance(requests,dict) or len(requests)>self.slots:raise ValueError('Invalid request batch')
        out={}
        for slot,value in requests.items():
            if type(slot) is not int or not 0<=slot<self.slots or not isinstance(value,(tuple,list)) or len(value)!=2:
                raise ValueError('Invalid inference slot/request')
            history,digest=value
            tape=validate_history(history,size=self.size,capacity=self.capacity)
            if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None:raise ValueError('Expected canonical native leaf digest')
            out[slot]=(tape,digest)
        return out

    def replay_suffixes(self,tapes,starts):
        # A cold long-history batch can exceed the worker's framed output
        # bound. Subdivide native replay before allocating/issuing requests.
        rows_per_call=MAX_OUTPUT//(self.size*self.size*24+77)
        result=[];group=[];offsets=[];rows=0;started=time.perf_counter()
        def flush():
            nonlocal rows
            if not group:return
            result.extend(self.replay(group,offsets));self.stats['replay_calls']+=1
            self.stats['native_encoded_positions']+=rows
            self.stats['native_validated_history_moves']+=sum(map(len,group))
            group.clear();offsets.clear();rows=0
        for tape,start in zip(tapes,starts):
            count=len(tape)+1-start
            if count>rows_per_call:raise ValueError('One suffix exceeds native output bound')
            if rows+count>rows_per_call or len(group)==128:flush()
            group.append(tape);offsets.append(start);rows+=count
        flush();self.stats['replay_seconds']+=time.perf_counter()-started
        return result

    def close(self):self.replay.close()


class TemporalRunner(Base):
    def __init__(self,params,c,replay,**kwargs):
        if c['architecture']!='causal_visual_policy':raise ValueError('Expected temporal architecture')
        causal.validate(c);super().__init__(params,c,replay,**kwargs)
        self.cache_specs=dict(keys=P('data'),values=P('data'),lengths=P('data'),valid=P('data'),network_version=P())
        root=self.replay([[]])[0]
        spatial=np.broadcast_to(root['spatial'][0],(self.slots,self.size,self.size,22)).copy().astype(np.float32)
        glob=np.broadcast_to(root['global_features'][0],(self.slots,19)).copy()
        fn=jax.shard_map(lambda p,s,g:joint.first_move(p,s,g,self.c,network_version=self.version),mesh=self.mesh,
                         in_specs=(P(),P('data'),P('data')),out_specs=(P('data'),self.cache_specs),check_vma=False)
        started=time.perf_counter();prediction,self.cache=jax.jit(fn)(self.params,self.put(spatial),self.put(glob))
        prediction=jax.device_get(prediction);jax.block_until_ready(self.cache)
        self.compilation['root_seconds']=time.perf_counter()-started
        self.paths=[() for _ in range(self.slots)]
        self.predictions=[[frozen_prediction({k:v[i] for k,v in prediction.items()})] for i in range(self.slots)]
        digest=row_digest(root,[],size=self.size,komi=self.replay.komi)
        self.hashes=[[digest] for _ in range(self.slots)]

    def executable(self,horizon,extent,inputs):
        key=(horizon,extent)
        if key not in self.compiled:
            def append(p,cache,s,g,actions,counts,rewind):
                cache={**cache,'lengths':jnp.where(rewind>=0,rewind,cache['lengths'])}
                def step(cache,i):
                    out,cache=joint.append_move(p,cache,actions[:,i],s[:,i],g[:,i],self.c,
                        attention_positions=extent,active=i<counts,network_version=self.version)
                    return cache,out
                cache,out=jax.lax.scan(step,cache,jnp.arange(horizon))
                return jax.tree.map(lambda x:jnp.swapaxes(x,0,1),out),cache
            f=jax.shard_map(append,mesh=self.mesh,
                in_specs=(P(),self.cache_specs,*[P('data')]*5),out_specs=(P('data'),self.cache_specs),check_vma=False)
            start=time.perf_counter();exe=jax.jit(f,donate_argnums=(1,)).lower(*inputs).compile()
            mem=exe.memory_analysis();self.compiled[key]=exe
            self.compilation[str(key)]=dict(seconds=time.perf_counter()-start,
                memory_bytes={k:int(getattr(mem,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
        return self.compiled[key]

    def score(self,requests):
        requests=self.validate(requests);changed={};results={}
        for slot,(tape,digest) in requests.items():
            old=self.paths[slot];common=0
            while common<min(len(old),len(tape)) and old[common]==tape[common]:common+=1
            if common==len(tape):
                if self.hashes[slot][common]!=digest:raise ValueError('Cached prefix differs from native leaf state')
                results[slot]=publish(self.predictions[slot][common])
            else:changed[slot]=[tape,common,digest]
        exact=self.replay_suffixes([x[0] for x in changed.values()],[x[1] for x in changed.values()])
        # Validate every requested leaf and shared prefix before cache donation.
        for (slot,(tape,common,digest)),rows in zip(changed.items(),exact):
            if row_digest(rows,tape,size=self.size,komi=self.replay.komi)!=digest or row_digest(rows,tape[:common],size=self.size,komi=self.replay.komi,index=0)!=self.hashes[slot][common]:
                raise ValueError('Native replay, shared prefix and requested leaf disagree')
            changed[slot].extend((rows,common))
        self.stats['requests']+=len(requests);self.stats['prefix_hits']+=len(results)
        try:
            while changed:
                remaining=max(len(x[0])-x[1] for x in changed.values())
                horizon=min(self.max_block,1<<(remaining-1).bit_length())
                s=np.zeros((self.slots,horizon,self.size,self.size,22),np.float32)
                g=np.zeros((self.slots,horizon,19),np.float32);actions=np.zeros((self.slots,horizon),np.int32)
                counts=np.zeros(self.slots,np.int32);rewind=np.full(self.slots,-1,np.int32)
                for slot,(tape,common,_,rows,begin) in changed.items():
                    n=min(horizon,len(tape)-common);counts[slot]=n;rewind[slot]=2*common+1
                    s[slot,:n]=rows['spatial'][common+1-begin:common+n+1-begin]
                    g[slot,:n]=rows['global_features'][common+1-begin:common+n+1-begin]
                    actions[slot,:n]=tape[common:common+n]
                needed=max((rewind[i]+2*counts[i]+1)//2 for i in changed)
                extent=min(self.capacity,1<<(int(needed)-1).bit_length())
                inputs=(self.params,self.cache,*map(self.put,(s,g,actions,counts,rewind)))
                exe=self.executable(horizon,extent,inputs);started=time.perf_counter()
                prediction,self.cache=exe(*inputs)
                prediction,valid,lengths=jax.device_get((prediction,self.cache['valid'],self.cache['lengths']))
                self.stats['dispatch_seconds']+=time.perf_counter()-started
                if not valid.all() or any(lengths[i]!=rewind[i]+2*counts[i] for i in changed):
                    raise RuntimeError('Cache rejected a validated exact continuation')
                self.stats['dispatches']+=1;self.stats['appended_positions']+=int(counts.sum())
                self.stats['padded_position_slots']+=self.slots*horizon
                for slot in list(changed):
                    tape,common,digest,rows,begin=changed[slot];n=int(counts[slot])
                    values=[frozen_prediction({k:v[slot,i] for k,v in prediction.items()}) for i in range(n)]
                    hashes=[row_digest(rows,tape[:position+1],size=self.size,komi=self.replay.komi,index=position+1-begin)
                            for position in range(common,common+n)]
                    self.predictions[slot][common+1:]=values;self.hashes[slot][common+1:]=hashes;self.paths[slot]=tape[:common+n]
                    if common+n==len(tape):results[slot]=publish(values[-1]);del changed[slot]
                    else:changed[slot][1]+=n
        except BaseException:
            self.failed=True;raise
        return results


class CNNRunner(Base):
    def __init__(self,params,c,replay,**kwargs):
        if c['architecture']!='katago_nested_policy':raise ValueError('Expected CNN architecture')
        super().__init__(params,c,replay,**kwargs);self.cache=[OrderedDict() for _ in range(self.slots)]
        def run(p,s,g,counts):
            return joint.forward(p,dict(spatial=s[:,None],global_features=g[:,None],
                actions=jnp.zeros((s.shape[0],1),jnp.int32),counts=counts),self.c)
        fn=jax.shard_map(run,mesh=self.mesh,in_specs=(P(),*[P('data')]*3),out_specs=P('data'),check_vma=False)
        self.executable=jax.jit(fn)

    def score(self,requests):
        requests=self.validate(requests);pending={};results={}
        for slot,(tape,digest) in requests.items():
            found=self.cache[slot].get(tape)
            if found is not None:
                if found[0]!=digest:raise ValueError('Cached CNN leaf differs from native state')
                results[slot]=publish(found[1])
            else:pending[slot]=(tape,digest)
        exact=self.replay_suffixes([x[0] for x in pending.values()],[len(x[0]) for x in pending.values()])
        s=np.zeros((self.slots,self.size,self.size,22),np.float32);g=np.zeros((self.slots,19),np.float32)
        counts=np.zeros(self.slots,np.int32)
        for (slot,(tape,digest)),rows in zip(pending.items(),exact):
            if row_digest(rows,tape,size=self.size,komi=self.replay.komi)!=digest:raise ValueError('Replayed CNN leaf differs from native state')
            s[slot]=rows['spatial'][-1];g[slot]=rows['global_features'][-1];counts[slot]=1
        self.stats['requests']+=len(requests);self.stats['prefix_hits']+=len(results)
        if pending:
            try:
                started=time.perf_counter();pred=jax.device_get(self.executable(self.params,self.put(s),self.put(g),self.put(counts)))
                self.stats['dispatch_seconds']+=time.perf_counter()-started
                self.stats['dispatches']+=1;self.stats['appended_positions']+=len(pending);self.stats['padded_position_slots']+=self.slots
                for slot,(tape,digest) in pending.items():
                    value=frozen_prediction({k:v[slot,0] for k,v in pred.items()})
                    if len(self.cache[slot])>=2*self.capacity:self.cache[slot].popitem(last=False)
                    self.cache[slot][tape]=(digest,value);results[slot]=publish(value)
            except BaseException:
                self.failed=True;raise
        return results


def Runner(params,c,replay,**kwargs):
    return (CNNRunner if c['architecture']=='katago_nested_policy' else TemporalRunner)(params,c,replay,**kwargs)
