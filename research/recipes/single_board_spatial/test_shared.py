"""Causality, shared-weight derivatives, rematerialization and cache semantics."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import causal
import encoder


def config(**kw):
    return dict(architecture='causal_visual_policy',width=16,layers=2,mlp_hidden=24,
        heads=2,kv_heads=1,max_board_size=3,max_positions=8,dtype='float32',rematerialize=True,
        norm_epsilon=1e-6,rope_theta=10000.,attention_backend='xla',encoder_width=16,
        encoder_blocks=2,encoder_passes=2,encoder_expansion=2,connector_channels=4,
        encoder_rematerialize=True,encoder_layer_scale=.1,policy_spatial_bias=True,**kw)


class SharedBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=config();cls.p=causal.initialize(17,cls.c)
        # Exercise every inference path with a learned, nonzero correction.
        cls.p['head.local.weight']=jnp.linspace(-.1,.1,cls.c['encoder_width'])[:,None]
        cls.p['head.local.bias']=jnp.asarray(.02)
        rng=np.random.default_rng(811)
        cls.s=jnp.asarray(rng.normal(size=(2,4,3,3,22)),jnp.float32)
        cls.g=jnp.asarray(rng.normal(size=(2,4,19)),jnp.float32)
        cls.a=jnp.asarray([[0,1,2,3],[4,5,6,7]],jnp.int32);cls.n=jnp.asarray([4,3])

    def test_causal_future_and_current_action(self):
        p,s,g,a,n,c=self.p,self.s,self.g,self.a,self.n,self.c
        fn=jax.jit(lambda s,g,a:causal.forward(p,s,g,a,n,c))
        base=fn(s,g,a)
        altered=fn(s.at[:,2:].set(99),g.at[:,2:].set(-50),a.at[:,1:].set(9))
        np.testing.assert_array_equal(base[:,:2],altered[:,:2])
        self.assertGreater(float(jnp.max(jnp.abs(base[:,2]-altered[:,2]))),1e-6)
        np.testing.assert_array_equal(base[1,3],0)

    def test_incremental_equals_full(self):
        p,s,g,a,n,c=self.p,self.s,self.g,self.a,self.n,self.c
        full=causal.forward(p,s,g,a,jnp.full((2,),4),c)
        first,cache=causal.first_move(p,s[:,0],g[:,0],c,network_version=3)
        np.testing.assert_allclose(first,full[:,0],atol=2e-5,rtol=2e-5)
        for t in range(1,4):
            out,cache=causal.append_move(p,cache,a[:,t-1],s[:,t],g[:,t],c,attention_positions=t+1,network_version=3)
            np.testing.assert_allclose(out,full[:,t],atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(cache['lengths'],2*t+1)
        pref,ca=causal.forward(p,s[:,:3],g[:,:3],a[:,:3],jnp.full((2,),3),c,with_cache=True,network_version=3)
        out,ca=causal.append_move(p,ca,a[:,2],s[:,3],g[:,3],c,attention_positions=4,network_version=3)
        np.testing.assert_allclose(out,full[:,3],atol=2e-5,rtol=2e-5)

    def test_cache_guards(self):
        p,s,g,a,c=self.p,self.s,self.g,self.a,self.c
        _,cache=causal.first_move(p,s[:,0],g[:,0],c,network_version=3)
        for active,version in [(jnp.asarray([False,False]),3),(jnp.asarray([True,True]),4)]:
            out,ca=causal.append_move(p,cache,a[:,0],s[:,1],g[:,1],c,attention_positions=2,active=active,network_version=version)
            np.testing.assert_array_equal(out,0)
            for k in ('keys','values','lengths'):np.testing.assert_array_equal(ca[k],cache[k])
            np.testing.assert_array_equal(ca['valid'],False if version==4 else True)

    def test_rematerialization_gradients(self):
        def loss(p,c):return jnp.mean(causal.forward(p,self.s,self.g,self.a,self.n,c)**2)
        vg=jax.jit(jax.value_and_grad(lambda p:loss(p,self.c)))(self.p)
        plain={**self.c,'encoder_rematerialize':False,'rematerialize':False}
        other=jax.jit(jax.value_and_grad(lambda p:loss(p,plain)))(self.p)
        for x,y in zip(jax.tree.leaves(vg),jax.tree.leaves(other)):
            np.testing.assert_allclose(x,y,atol=2e-6,rtol=2e-4)

    def test_shared_gradient_is_sum_of_pass_gradients(self):
        c=self.c;w=c['encoder_width'];p=encoder.initialize(7,c)
        params={k[7:]:v for k,v in p.items() if k.startswith('blocks.')}
        x=jnp.arange(2*3*3*w,dtype=jnp.float32).reshape(2,3,3,w)/100
        def repeated(stacks):
            y=x
            for stack in stacks:
                y,_=jax.lax.scan(lambda z,b:(encoder.block(z,b,c),None),y,stack)
            return jnp.mean(y*y)
        independent=jax.grad(repeated)((params,params))
        shared=jax.grad(lambda q:repeated((q,q)))(params)
        for name in params:
            np.testing.assert_allclose(shared[name],independent[0][name]+independent[1][name],rtol=2e-5,atol=2e-6)
        self.assertGreater(float(jnp.linalg.norm(independent[0]['up.weight'])),1e-6)
        self.assertGreater(float(jnp.linalg.norm(independent[1]['up.weight'])),1e-6)
        # Increasing pass count does not add trainable parameters.
        self.assertEqual(causal.parameter_schema(c),causal.parameter_schema({**c,'encoder_passes':3}))

    def test_parameter_count_and_dtype(self):
        shapes=causal.parameter_schema(self.c)
        self.assertFalse(any('readout' in x['path'] for x in shapes))
        bf={**self.c,'dtype':'bfloat16'}
        q={k[8:]:v for k,v in self.p.items() if k.startswith('encoder.')}
        self.assertEqual(encoder.spatial(q,self.s,bf).dtype,jnp.bfloat16)
        with self.assertRaises(ValueError):causal.layout(9,self.c)

    def test_zero_correction_preserves_control(self):
        c=self.c;off={**c,'policy_spatial_bias':False}
        candidate=causal.initialize(17,c);control=causal.initialize(17,off)
        self.assertEqual(set(candidate)-set(control),{'head.local.weight','head.local.bias'})
        for key in control:np.testing.assert_array_equal(candidate[key],control[key])
        x=causal.forward(candidate,self.s,self.g,self.a,self.n,c)
        y=causal.forward(control,self.s,self.g,self.a,self.n,off)
        np.testing.assert_array_equal(x,y)
        added=sum(v.size for k,v in candidate.items() if k not in control)
        self.assertEqual(added,c['encoder_width']+1)

    def test_spatial_path_learns_before_encoder_changes(self):
        c=self.c;p=causal.initialize(17,c)
        def objective(p):
            logits=causal.forward(p,self.s,self.g,self.a,self.n,c)
            # A location-dependent target exercises the spatial projection.
            return -jax.nn.log_softmax(logits,axis=-1)[0,0,2]
        g=jax.jit(jax.grad(objective))(p)
        self.assertGreater(float(jnp.linalg.norm(g['head.local.weight'])),1e-6)
        # The encoder still receives its original gradient at zero correction.
        off={**c,'policy_spatial_bias':False};control=causal.initialize(17,off)
        cg=jax.grad(lambda q:-jax.nn.log_softmax(causal.forward(q,self.s,self.g,self.a,self.n,off),axis=-1)[0,0,2])(control)
        for name in control:np.testing.assert_allclose(g[name],cg[name],atol=2e-6,rtol=2e-4)

    def test_readout_spatial_permutation_and_pass(self):
        c=self.c;p=self.p
        h=jnp.ones((2,c['width']));f=jnp.arange(2*3*3*c['encoder_width'],dtype=jnp.float32).reshape(2,3,3,-1)/100
        base=causal.policy(p,h,3,{**c,'policy_spatial_bias':False})
        delta=causal.policy(p,h,3,c,f)-base
        swapped=causal.policy(p,h,3,c,jnp.flip(f,axis=1))-base
        np.testing.assert_allclose(swapped[:,:9].reshape(2,3,3),jnp.flip(delta[:,:9].reshape(2,3,3),axis=1),atol=1e-6)
        np.testing.assert_array_equal(delta[:,-1],0.)
        with self.assertRaises(ValueError):causal.policy(p,h,3,c,None)


if __name__=='__main__':unittest.main()
