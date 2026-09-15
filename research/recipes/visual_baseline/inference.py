"""Common RPC owner; CNN scores one exact history window per pending leaf."""
import hashlib
import time
import jax
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from gozero.visual_history import Replay,observation_sha256
if __package__:
    from . import cnn,transformer_inference
else:
    import cnn,transformer_inference


def Runner(params,c,native,rules,**kwargs):
    cls=CNNRunner if c.get('architecture')=='residual_cnn' else transformer_inference.Runner
    return cls(params,c,native,rules,**kwargs)


class CNNRunner:
    def __init__(self,params,c,native,rules,*,slots,cache_positions,network_version,
                 exit_depth=None,max_block=16,suffix_replay=False):
        if slots<1 or slots%len(jax.local_devices()) or not 1<=cache_positions<=c['max_positions']:
            raise ValueError('Invalid CNN inference dimensions')
        if exit_depth is not None: raise ValueError('CNN baseline has no trained draft exit')
        self.c,self.slots,self.capacity,self.version=c,slots,cache_positions,network_version
        self.depth=exit_depth;self.size=rules['size'];self.suffix_replay=suffix_replay
        self.replay=Replay(native,rules,cache_positions)
        self.mesh=Mesh(np.asarray(jax.local_devices()),('data',))
        self.replicated=NamedSharding(self.mesh,P());self.batched=NamedSharding(self.mesh,P('data'))
        self.params=jax.device_put(params,self.replicated);self.cache=[{} for _ in range(slots)]
        self.stats={k:0 for k in ('requests','prefix_hits','dispatches','appended_positions','padded_position_slots','native_encoded_positions','native_validated_history_moves')}
        self.stats.update(replay_seconds=0.,dispatch_seconds=0.,total_seconds=0.)
        shape=(slots,self.size,self.size,c['history']*9+1)
        fn=jax.shard_map(lambda p,x:cnn.frames(p,x,c),mesh=self.mesh,in_specs=(P(),P('data')),out_specs=P('data'),check_vma=False)
        started=time.perf_counter();lowered=jax.jit(fn).lower(self.params,self.put(np.zeros(shape,np.float32)))
        self.executable=lowered.compile();jax.block_until_ready(self.executable(self.params,self.put(np.zeros(shape,np.float32))))
        self.compilation={'frames':{'seconds':time.perf_counter()-started,'hlo_sha256':hashlib.sha256(lowered.compiler_ir('hlo').as_hlo_text().encode()).hexdigest(),'compiler_cost_estimate':self.executable.cost_analysis()}}
    def put(self,x):return jax.device_put(x,self.batched)
    def warmup(self):pass
    def reset(self):self.cache=[{} for _ in range(self.slots)]
    def score(self,requests):
        started=time.perf_counter();results={};pending={}
        for slot,(history,digest) in requests.items():
            if type(slot) is not int or not 0<=slot<self.slots:raise ValueError('Invalid CNN slot')
            tape=self.replay.validate(history)
            if not isinstance(digest,str) or len(digest)!=64:raise ValueError('Missing exact leaf hash')
            found=self.cache[slot].get(tape)
            if found is not None:
                if found[0]!=digest:raise ValueError('Cached CNN history and leaf differ')
                results[slot]=found[1]
            else:pending[slot]=(tape,digest)
        start_replay=time.perf_counter();tapes=[list(x[0]) for x in pending.values()]
        starts=[max(0,len(tape)-self.c['history']+1) if self.suffix_replay else 0 for tape in tapes]
        exact=self.replay.suffix(tapes,starts) if self.suffix_replay else self.replay(tapes)
        inputs=np.zeros((self.slots,self.size,self.size,self.c['history']*9+1),np.float32)
        for (slot,(tape,digest)),boards,begin in zip(pending.items(),exact,starts):
            if observation_sha256(boards[-1])!=digest:raise ValueError('Replayed CNN leaf hash differs')
            planes=[]
            for lag in range(self.c['history']):
                source=len(tape)-lag
                obs=boards[source-begin] if source>=0 else np.zeros((self.size,self.size,6),np.float32)
                present=np.full((self.size,self.size,1),float(source>=0),np.float32)
                move=np.zeros_like(present);passed=np.zeros_like(present)
                if source>0:
                    action=tape[source-1]
                    if action==self.size*self.size:passed.fill(1.)
                    else:move[action//self.size,action%self.size,0]=1.
                planes.extend([obs,present,move,passed])
            planes.append(np.ones((self.size,self.size,1),np.float32));inputs[slot]=np.concatenate(planes,-1)
        self.stats['replay_seconds']+=time.perf_counter()-start_replay
        self.stats['native_encoded_positions']+=sum(len(x) for x in exact)
        self.stats['native_validated_history_moves']+=sum(map(len,tapes))
        self.stats['requests']+=len(requests);self.stats['prefix_hits']+=len(results)
        if pending:
            begin=time.perf_counter();predictions=jax.device_get(self.executable(self.params,self.put(inputs)))
            self.stats['dispatch_seconds']+=time.perf_counter()-begin;self.stats['dispatches']+=1
            self.stats['appended_positions']+=len(pending);self.stats['padded_position_slots']+=self.slots
            for slot,(tape,digest) in pending.items():
                prediction={k:np.asarray(v[slot]).copy() for k,v in predictions.items()}
                if not all(np.isfinite(v).all() for v in prediction.values()):raise FloatingPointError('Nonfinite CNN prediction')
                if len(self.cache[slot])>=self.capacity*2:self.cache[slot].clear()
                self.cache[slot][tape]=(digest,prediction);results[slot]=prediction
        self.stats['total_seconds']+=time.perf_counter()-started
        return results
