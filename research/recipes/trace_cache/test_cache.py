"""Native game transitions plus independent full-prefix checks of retained KV."""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

os.environ['JAX_PLATFORMS']='cpu'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
import jax
import jax.numpy as jnp
import numpy as np
from gozero.native import load_library
import model
from decode import decode
from cached_decode import decode_cached,empty_cache,rebuild_cache
from cache_protocol import CacheCoordinator,CacheRebuildRequired


class RetainedCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt=Path(os.environ['GOZERO_NATIVE_RECEIPT']);r=json.loads(receipt.read_text())
        cls.native=load_library(receipt.parent/r['filename'],r['binary_sha256'])
        assert cls.native.BOUNDED_STATE_TRACE_ABI_VERSION==1
        jax.config.update('jax_default_matmul_precision','highest')
        cls.c={'size':3,'komi':.5,'width':16,'heads':4,'blocks':1,'max_tokens':13,'dtype':'float32',
            'expert_temperature':1.,'behavior_temperature':1.,'board_mode':'exact','board_width':8,'board_blocks':1}
        cls.actor={'size':3,'komi':.5,'history':2,'scoring':'pass_alive_area','root_legal_mask':True,'max_game_moves':8}
        cls.p=model.initialize(271,cls.c);cls.key=jax.random.key(51);cls.compiled={}

    def make(self,k=3,models=(11,11)):
        return (self.native.BoundedTraceActors(json.dumps(self.actor),2),
                CacheCoordinator(2,4,k,10,13,models),empty_cache(self.c,2,k))

    def functions(self,k):
        if k not in self.compiled:
            settings={'horizon':4,'samples':k,'coupling':'shared'if k==1 else 'independent','oracle_behavior':k==1}
            c=self.c
            self.compiled[k]=(jax.jit(lambda p,t,l,e,key,s,legal:decode(p,t,l,e,key,s,c,root_legal=legal,**settings)),
                jax.jit(lambda p,carry,control,l,e,key,s,legal:decode_cached(p,carry,control,l,e,key,s,c,root_legal=legal,**settings)))
        return self.compiled[k]

    def frame(self,engine,active=(True,True),models=(11,11)):
        encoded,tokens,lengths,legal,stones=engine.start(list(models),[0,0],4,list(active))
        tickets=json.loads(encoded);episodes=np.asarray([t['episode']if t is not None else 0 for t in tickets],np.uint32)
        return encoded,tickets,tokens,lengths,legal,stones,episodes

    def predict(self,frame,coordinator,carry,p=None):
        _,tickets,tokens,lengths,legal,stones,episodes=frame;p=self.p if p is None else p
        control=coordinator.prepare(tickets,tokens,lengths);full,cached=self.functions(coordinator.samples)
        expected=jax.device_get(full(p,tokens,lengths,episodes,self.key,stones,legal))
        result=cached(p,carry,control,lengths,episodes,self.key,stones,legal)
        arrays=tuple(np.array(x,copy=True,order='C')for x in jax.device_get(result[:4]))
        for game,ticket in enumerate(tickets):
            if ticket is None:continue
            np.testing.assert_array_equal(arrays[0][game],expected[0][game])
            np.testing.assert_allclose(arrays[1][game],expected[1][game],rtol=2e-5,atol=2e-5)
            np.testing.assert_array_equal(arrays[2][game],expected[2][game]);np.testing.assert_array_equal(arrays[3][game],expected[3][game])
            counts=np.zeros((2,10),np.float32)
            for ply,action in enumerate(tokens[game,1:int(lengths[game])+1]):counts[ply%2,action]+=1
            np.testing.assert_array_equal(np.asarray(result[4]['counts'])[game],counts)
        for old,new in zip(jax.tree.leaves(carry),jax.tree.leaves(result[4])):
            for game,ticket in enumerate(tickets):
                if ticket is None:np.testing.assert_array_equal(np.asarray(old)[game],np.asarray(new)[game])
        return arrays,result[4],control

    def commit(self,engine,coordinator,frame,arrays,allowances):
        resolved=json.loads(engine.resolve(frame[0],*arrays,list(allowances)))
        certificates=coordinator.commit(resolved,arrays[0])
        return resolved,certificates

    def force_root(self,frame,arrays,desired):
        for game,ticket in enumerate(frame[1]):
            if ticket is None:continue
            action=desired[game];self.assertTrue(frame[4][game,action])
            arrays[2][game,0,action]=1e6
            arrays[0][game,ticket['move_number']%2,:,0]=action

    def test_long_native_prefixes_counts_and_inactive_carry_match_full_reconstruction(self):
        engine,coordinator,carry=self.make();episodes=set();depths=set()
        for step in range(32):
            active=(step%5!=1,step%7!=2);frame=self.frame(engine,active)
            arrays,carry,_=self.predict(frame,coordinator,carry)
            resolved,_=self.commit(engine,coordinator,frame,arrays,
                [1+step%4 if active[0]else 0,1+(step*3)%4 if active[1]else 0])
            for ticket,row in zip(frame[1],resolved):
                if ticket is not None:episodes.add(ticket['episode']);depths.add(len(row['actions']))
        self.assertGreater(max(episodes),0);self.assertGreater(max(depths),1)

    def test_terminal_capture_cap_and_inactive_finished_games(self):
        engine,coordinator,carry=self.make(k=1);sequence=[1,0,3,8,2,7,4,6]
        frozen=None
        for step,action in enumerate(sequence):
            active=(step<2,True);frame=self.frame(engine,active);arrays,carry,control=self.predict(frame,coordinator,carry)
            self.force_root(frame,arrays,[9,action]);resolved,_=self.commit(engine,coordinator,frame,arrays,[1 if active[0]else 0,1])
            states=json.loads(engine.inspect())
            if step==1:
                self.assertEqual(resolved[0]['stop'],'terminal');self.assertIsNotNone(resolved[0]['terminal_white_score']);frozen=states[0]
            if step>=2:self.assertEqual(states[0],frozen)
            if step==2:self.assertEqual(states[1]['stones'][0],0);self.assertEqual(states[1]['moves'],[1,0,3])
            if step==7:self.assertEqual(resolved[1]['stop'],'move_limit');self.assertIsNone(resolved[1]['terminal_white_score'])
        frame=self.frame(engine);arrays,carry,control=self.predict(frame,coordinator,carry)
        self.assertTrue((control[:,1]==1).all());self.assertTrue((frame[3]==0).all())
        self.assertEqual([t['episode']for t in frame[1]],[1,1]);self.commit(engine,coordinator,frame,arrays,[1,1])

    def test_legality_corrected_last_move_replaces_speculative_future_kv(self):
        engine,coordinator,carry=self.make(k=1);frame=self.frame(engine);arrays,carry,_=self.predict(frame,coordinator,carry)
        desired=[]
        for game in range(2):
            black=int(arrays[0][game,0,0,0]);original_white=int(arrays[0][game,1,0,1])
            # Choose an empty point different from the predicted second action.
            self.assertLess(black,9)
            other=next(a for a in range(9)if a not in (black,original_white));desired.append(other)
            arrays[2][game,1,black]=1e6;arrays[2][game,1,other]=1e5
            arrays[0][game,1,0,1]=black
        rows,certificates=self.commit(engine,coordinator,frame,arrays,[2,2])
        for row,certificate,other in zip(rows,certificates,desired):
            self.assertEqual(row['legality_corrections'],1);self.assertEqual(row['actions'][-1],other)
            self.assertEqual(certificate['valid_through'],1);self.assertEqual(certificate['last_actual_action'],other)
        frame=self.frame(engine);arrays,carry,control=self.predict(frame,coordinator,carry)
        np.testing.assert_array_equal(control[:,3],desired);self.commit(engine,coordinator,frame,arrays,[1,1])

    def test_prefix_and_generation_tampering_and_commit_atomicity(self):
        engine,coordinator,carry=self.make(k=1);frame=self.frame(engine);arrays,carry,_=self.predict(frame,coordinator,carry)
        rows=json.loads(engine.resolve(frame[0],*arrays,[2,2]));self.assertTrue(all(len(r['actions'])==2 for r in rows))
        broken=arrays[0].copy();view=(frame[1][1]['move_number']+1)%2;broken[1,view,rows[1]['selected_samples'][-1],0]=(rows[1]['actions'][0]+1)%10
        with self.assertRaisesRegex(ValueError,'prior actions'):coordinator.commit(rows,broken)
        self.assertTrue(all(e is None for e in coordinator.entries));self.assertIsNotNone(coordinator.pending)
        coordinator.commit(rows,arrays[0]);frame=self.frame(engine)
        bad=copy.deepcopy(frame[1]);bad[0]['generation']+=1
        with self.assertRaisesRegex(ValueError,'generation'):coordinator.prepare(bad,frame[2],frame[3])
        corrupt=frame[2].copy();corrupt[0,1]=(int(corrupt[0,1])+1)%10
        with self.assertRaisesRegex(ValueError,'canonical tape'):coordinator.prepare(frame[1],corrupt,frame[3])
        self.assertIsNone(coordinator.pending)
        coordinator.prepare(frame[1],frame[2],frame[3]);engine.cancel();coordinator.invalidate()
        frame=self.frame(engine)
        with self.assertRaises(CacheRebuildRequired):coordinator.prepare(frame[1],frame[2],frame[3])
        engine.cancel()

    def test_nonempty_model_change_and_fresh_coordinator_reconstruction(self):
        engine,coordinator,carry=self.make(k=1)
        for action in (1,0,3):
            frame=self.frame(engine);arrays,carry,_=self.predict(frame,coordinator,carry)
            self.force_root(frame,arrays,[action,action]);self.commit(engine,coordinator,frame,arrays,[1,1])
        changed=(12,12);frame=self.frame(engine,models=changed)
        with self.assertRaisesRegex(ValueError,'model/context'):coordinator.prepare(frame[1],frame[2],frame[3])
        coordinator.invalidate(models=changed)
        with self.assertRaises(CacheRebuildRequired):coordinator.prepare(frame[1],frame[2],frame[3])
        p=model.initialize(272,self.c)
        carry=jax.jit(lambda p,t,l:rebuild_cache(p,t,l,self.c,samples=1))(p,frame[2],frame[3])
        jax.block_until_ready(carry);coordinator.adopt_rebuilt(frame[1],frame[2],frame[3])
        arrays,carry,control=self.predict(frame,coordinator,carry,p=p)
        self.assertTrue((control[:,1]==0).all());self.assertTrue((control[:,4]==0).all())
        self.commit(engine,coordinator,frame,arrays,[1,1])
        # A new coordinator can reconstruct the same retained contract.
        fresh=CacheCoordinator(2,4,1,10,13,changed);frame=self.frame(engine,models=changed)
        with self.assertRaises(CacheRebuildRequired):fresh.prepare(frame[1],frame[2],frame[3])
        carry=rebuild_cache(p,frame[2],frame[3],self.c,samples=1);jax.block_until_ready(carry)
        fresh.adopt_rebuilt(frame[1],frame[2],frame[3]);arrays,carry,_=self.predict(frame,fresh,carry,p=p)
        with tempfile.TemporaryDirectory(prefix='gozero-cache-rebuild-')as temp:
            fixture=Path(temp)/'fixture.json';output=Path(temp)/'output.npz'
            fixture.write_text(json.dumps({'model':self.c,'actor':self.actor,'models':changed,'parameter_seed':272,
                'key_seed':51,'moves':[frame[2][i,1:int(frame[3][i])+1].tolist()for i in range(2)],
                'tickets':frame[1],'stones':frame[5].tolist()}))
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--rebuild-worker',str(fixture),str(output)],
                capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr[-4000:])
            with np.load(output,allow_pickle=False)as loaded:
                for i,expected in enumerate(arrays):
                    if i==1:np.testing.assert_allclose(loaded['a'+str(i)],expected,rtol=2e-5,atol=2e-5)
                    else:np.testing.assert_array_equal(loaded['a'+str(i)],expected)
        self.commit(engine,fresh,frame,arrays,[1,1])


def rebuild_worker(fixture_path,output_path):
    d=json.loads(Path(fixture_path).read_text());c=d['model'];games=len(d['moves']);actions=c['size']**2+1
    rpath=Path(os.environ['GOZERO_NATIVE_RECEIPT']);r=json.loads(rpath.read_text())
    native=load_library(rpath.parent/r['filename'],r['binary_sha256'])
    engine=native.BoundedTraceActors(json.dumps(d['actor']),games)
    for ply in range(max(map(len,d['moves']))):
        active=[ply<len(moves)for moves in d['moves']]
        encoded,tokens,lengths,legal,stones=engine.start(d['models'],[0,0],4,active)
        proposals=np.zeros((games,2,1,1),np.int32);logits=np.zeros((games,2,1,1,actions),np.float32)
        noise=np.zeros((games,1,actions),np.float32)
        inputs=np.broadcast_to(stones[:,None,None,None],(games,2,1,1,actions-1)).copy()
        for game,enabled in enumerate(active):
            if enabled:
                action=d['moves'][game][ply];assert legal[game,action]
                proposals[game,...]=action;logits[game,...,action]=100.
        engine.resolve(encoded,proposals,logits,noise,inputs,[int(x)for x in active])
    encoded,tokens,lengths,legal,stones=engine.start(d['models'],[0,0],4,[True]*games);tickets=json.loads(encoded)
    assert tickets==d['tickets'];np.testing.assert_array_equal(stones,d['stones'])
    jax.config.update('jax_default_matmul_precision','highest');p=model.initialize(d['parameter_seed'],c)
    carry=jax.jit(lambda p,t,l:rebuild_cache(p,t,l,c,samples=1))(p,tokens,lengths);jax.block_until_ready(carry)
    coordinator=CacheCoordinator(games,4,1,actions,c['max_tokens'],d['models'])
    coordinator.adopt_rebuilt(tickets,tokens,lengths);control=coordinator.prepare(tickets,tokens,lengths)
    episodes=np.asarray([t['episode']for t in tickets],np.uint32);key=jax.random.key(d['key_seed'])
    outputs=jax.jit(lambda p,state,ct,l,e,k,s,legal:decode_cached(p,state,ct,l,e,k,s,c,horizon=4,samples=1,
        coupling='shared',oracle_behavior=True,root_legal=legal))(p,carry,control,lengths,episodes,key,stones,legal)
    arrays=jax.device_get(outputs[:4]);np.savez(output_path,**{'a'+str(i):a for i,a in enumerate(arrays)})
    resolved=json.loads(engine.resolve(encoded,*[np.ascontiguousarray(a)for a in arrays],[1]*games))
    coordinator.commit(resolved,arrays[0])


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--rebuild-worker':rebuild_worker(sys.argv[2],sys.argv[3])
    else:unittest.main()
