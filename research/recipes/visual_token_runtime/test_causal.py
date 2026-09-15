"""Causal token alignment, patch coverage, cache extents and independent attention."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import causal
import policy_model

C=dict(architecture='causal_visual_policy',width=32,layers=2,mlp_hidden=48,heads=4,kv_heads=2,
       max_board_size=9,max_positions=16,dtype='float32',rematerialize=True,norm_epsilon=1e-6,
       rope_theta=10000.,attention_backend='xla',encoder='overlap_linear')

def fixture():
    r=np.random.default_rng(813);s=r.integers(0,2,(2,5,3,3,22)).astype(np.float32);s[...,0]=1
    return s,r.normal(size=(2,5,19)).astype(np.float32)*.1,r.integers(0,10,(2,5),dtype=np.int32),np.asarray([5,4],np.int32)

class CausalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=jax.jit(lambda:causal.initialize(91,C))();cls.s,cls.g,cls.a,cls.n=map(jnp.asarray,fixture())

    def test_gqa_against_numpy(self):
        r=np.random.default_rng(7);q=r.normal(size=(2,4,4,8)).astype(np.float32);k=r.normal(size=(2,9,2,8)).astype(np.float32);v=r.normal(size=k.shape).astype(np.float32)
        positions=np.asarray([[2,3,4,5],[0,1,2,3]]);lengths=np.asarray([6,4]);kk=np.repeat(k,2,axis=2);vv=np.repeat(v,2,axis=2)
        logits=np.einsum('bthd,bshd->bhts',q,kk)/np.sqrt(8)
        mask=(np.arange(9)[None,None,:]<=positions[:,:,None])&(np.arange(9)[None,None,:]<lengths[:,None,None])
        logits=np.where(mask[:,None],logits,-1e30);weights=np.exp(logits-logits.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
        expected=np.einsum('bhts,bshd->bthd',weights,vv)
        np.testing.assert_allclose(causal.dense_attention(jnp.asarray(q),jnp.asarray(k),jnp.asarray(v),jnp.asarray(positions),jnp.asarray(lengths)),expected,atol=2e-6,rtol=2e-6)

    def test_patch_projection_covers_board_and_matches_numpy(self):
        for size in range(1,53):
            covered=np.zeros((size,size),bool)
            for y in causal.anchors(size):
                for x in causal.anchors(size):covered[y:y+2,x:x+2]=True
            self.assertTrue(covered.all())
            self.assertEqual(set(map(int,causal.anchors(size))),set(int(max(0,size-2)-v) for v in causal.anchors(size)))
        got=causal.observation_tokens(self.p,self.s,self.g,C);expected=np.empty(got.shape,np.float32)
        for y,sy in enumerate(causal.anchors(3)):
            for x,sx in enumerate(causal.anchors(3)):
                patch=np.asarray(self.s[:,:,sy:sy+2,sx:sx+2]).reshape(2,5,88)
                expected[:,:,y*2+x]=(patch@np.asarray(self.p['encoder.patch.weight'])+np.asarray(self.p['encoder.patch.bias'])
                    +np.asarray(self.p['encoder.row.weight'][sy])+np.asarray(self.p['encoder.col.weight'][sx])
                    +np.asarray(self.p['token_types.weight'][0])+np.asarray(self.g)@np.asarray(self.p['encoder.global.weight']))
        np.testing.assert_allclose(got,expected,atol=2e-6,rtol=2e-6)

    def test_current_action_future_board_and_other_sequence_do_not_leak(self):
        run=jax.jit(lambda s,g,a:causal.forward(self.p,s,g,a,self.n,C))
        base=run(self.s,self.g,self.a)
        changed=run(self.s.at[0,3:].add(.3).at[1].set(0),self.g.at[0,3:].add(1).at[1].set(0),self.a.at[0,2:].set(0).at[1].set(9))
        np.testing.assert_array_equal(base[0,:3],changed[0,:3])
        now=run(self.s.at[0,2,1,1,1].add(1.),self.g,self.a)
        self.assertGreater(float(jnp.max(jnp.abs(base[0,2]-now[0,2]))),1e-6)

    def test_ragged_cached_move_matches_full_with_larger_allocation(self):
        counts=jnp.asarray([2,3]);_,cache=causal.forward(self.p,self.s[:,:3],self.g[:,:3],self.a[:,:3],counts,C,with_cache=True,cache_positions=16,network_version=71)
        s=jnp.stack([self.s[0,2],self.s[1,3]]);g=jnp.stack([self.g[0,2],self.g[1,3]]);a=jnp.asarray([self.a[0,1],self.a[1,2]])
        run=jax.jit(lambda ca:causal.append_move(self.p,ca,a,s,g,C,attention_positions=4,network_version=71))
        got,updated=run(cache)
        full,ref=causal.forward(self.p,self.s[:,:4],self.g[:,:4],self.a[:,:4],counts+1,C,with_cache=True,cache_positions=16,network_version=71)
        np.testing.assert_allclose(got,jnp.stack([full[0,2],full[1,3]]),atol=2e-5,rtol=2e-5)
        for key in ref:np.testing.assert_allclose(updated[key],ref[key],atol=2e-5,rtol=2e-5,err_msg=key)
        # Reading a larger zero-padded extent cannot alter a live prediction.
        other,_=causal.append_move(self.p,cache,a,s,g,C,attention_positions=16,network_version=71)
        np.testing.assert_allclose(got,other,atol=2e-5,rtol=2e-5)
        for extent,version,bad_action in [(3,71,a),(4,72,a),(4,71,jnp.asarray([-1,10]))]:
            _,bad=causal.append_move(self.p,cache,bad_action,s,g,C,attention_positions=extent,network_version=version)
            invalid=np.asarray(~bad['valid'])
            for key in ('keys','values','lengths'):np.testing.assert_array_equal(bad[key][invalid],cache[key][invalid])

    def test_opening_has_no_fictitious_action(self):
        got,cache=jax.jit(lambda p,s,g:causal.first_move(p,s,g,C))(self.p,self.s[:,0],self.g[:,0])
        expected,ref=causal.forward(self.p,self.s[:,:1],self.g[:,:1],self.a[:,:1],jnp.ones((2,),jnp.int32),C,with_cache=True)
        np.testing.assert_allclose(got,expected[:,0],atol=2e-5,rtol=2e-5)
        for key in ref:np.testing.assert_allclose(cache[key],ref[key],atol=2e-5,rtol=2e-5,err_msg=key)

    def test_carried_cache_matches_reference_and_rejects_bad_pointer(self):
        counts=jnp.asarray([2,3]);_,cache=causal.forward(self.p,self.s[:,:3],self.g[:,:3],self.a[:,:3],counts,C,with_cache=True)
        s=jnp.stack([self.s[0,2],self.s[1,3]]);g=jnp.stack([self.g[0,2],self.g[1,3]]);a=jnp.asarray([self.a[0,1],self.a[1,2]])
        for active in [jnp.asarray([True,True]),jnp.asarray([False,True]),jnp.asarray([False,False])]:
            actual=causal.append_move(self.p,cache,a,s,g,C,attention_positions=4,active=active)
            expected=causal.append_move_reference(self.p,cache,a,s,g,C,attention_positions=4,active=active)
            for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_allclose(x,y,atol=2e-5,rtol=2e-5)
        stride=causal.layout(3,C)['stride']
        for pointer in [-1,0,999999,2**31-1,((2**31-1)//stride)*stride-1]:
            broken={**cache,'lengths':jnp.asarray([pointer,pointer],jnp.int32)}
            logits,rejected=jax.jit(lambda ca:causal.append_move(self.p,ca,a,s,g,C,attention_positions=4))(broken)
            self.assertFalse(np.asarray(rejected['valid']).any());np.testing.assert_array_equal(logits,0.)
            for key in ('keys','values','lengths'):np.testing.assert_array_equal(rejected[key],broken[key])

    def test_one_policy_loss_and_encoder_gradient(self):
        policies=jnp.full((2,5,10),.1);b=dict(spatial=self.s,global_features=self.g,actions=self.a,counts=self.n,
            policies=policies,legal=jnp.ones((2,5,10),bool))
        (loss,metrics),grad=jax.jit(jax.value_and_grad(lambda p:policy_model.losses(p,b,C),has_aux=True))(self.p)
        self.assertTrue(np.isfinite(float(loss)));self.assertNotIn('helper_ce',metrics)
        self.assertGreater(float(jnp.linalg.norm(grad['encoder.patch.weight'])),1e-6)
        self.assertFalse(any('behavior' in k or 'value' in k or 'intermediate' in k for k in self.p))

    def test_global_weighted_gradient(self):
        from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
        self.assertEqual(len(jax.devices()),4)
        s=jnp.concatenate([self.s,self.s],0);g=jnp.concatenate([self.g,self.g],0);a=jnp.concatenate([self.a,self.a],0)
        b=dict(spatial=s,global_features=g,actions=a,counts=jnp.asarray([1,3,5,0]),policies=jnp.full((4,5,10),.1),legal=jnp.ones((4,5,10),bool))
        mesh=Mesh(np.asarray(jax.devices()),('data',));specs=jax.tree.map(lambda _:P('data'),b)
        mapped=jax.shard_map(lambda p,b:policy_model.losses(p,b,C,axis_name='data')[0],mesh=mesh,in_specs=(P(),specs),out_specs=P(),check_vma=False)
        p=jax.device_put(self.p,NamedSharding(mesh,P()));batch=jax.device_put(b,NamedSharding(mesh,P('data')))
        got=jax.jit(jax.value_and_grad(mapped))(p,batch)
        expected=jax.jit(jax.value_and_grad(lambda p,b:policy_model.losses(p,b,C)[0]))(p,b)
        for x,y in zip(jax.tree.leaves(got),jax.tree.leaves(expected)):np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-4)

if __name__=='__main__':unittest.main()
