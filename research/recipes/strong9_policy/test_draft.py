"""Verify draft visibility, shared gradients and exact full-history cache semantics."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import causal
import encoder
import draft_model
import draft_mask
import policy_model
from test_shared import config


class DraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c={**config(),'first_pass_aux_weight':.25}
        cls.p=causal.initialize(5,cls.c)
        cls.p['head.local.weight']=jnp.linspace(-.1,.1,16)[:,None]
        cls.p['head.context.q.weight']=jax.random.normal(jax.random.key(19),(16,4))*.1
        cls.s=jax.random.normal(jax.random.key(20),(2,4,3,3,22));cls.g=jax.random.normal(jax.random.key(21),(2,4,19))
        cls.a=jnp.asarray([[0,1,2,3],[4,5,6,7]]);cls.n=jnp.asarray([4,4])

    def test_mask_matches_independent_visibility_definition(self):
        mask=np.asarray(draft_mask.DraftMask((512,512),12)[:,:])
        expected=np.zeros((512,512),bool)
        for query in range(512):
            frame,kind=divmod(query,3)
            for key in range(12):
                other,what=divmod(key,3)
                expected[query,key]=(other<frame and what in (1,2)) or (other==frame and what in ((0,) if kind==0 else (1,) if kind==1 else (1,2)))
        np.testing.assert_array_equal(mask,expected)

    def test_main_matches_ordinary_and_draft_matches_full_history_cache(self):
        c,p,s,g,a,n=self.c,self.p,self.s,self.g,self.a,self.n
        main,draft=draft_model.forward(p,s,g,a,n,c)
        normal=causal.forward(p,s,g,a,n,c)
        np.testing.assert_allclose(main,normal,atol=3e-5,rtol=3e-5)
        cheap={**c,'encoder_passes':1,'first_pass_aux_weight':0}
        opening,_=causal.first_move(p,s[:,0],g[:,0],cheap)
        np.testing.assert_allclose(draft[:,0],opening,atol=3e-5,rtol=3e-5)
        for t in range(1,4):
            _,history=causal.forward(p,s[:,:t],g[:,:t],a[:,:t],jnp.full((2,),t),c,with_cache=True)
            predicted,_=causal.append_move(p,history,a[:,t-1],s[:,t],g[:,t],cheap,attention_positions=t+1)
            np.testing.assert_allclose(draft[:,t],predicted,atol=3e-5,rtol=3e-5)

    def test_no_draft_history_or_current_full_board_leakage(self):
        p,c=self.p,self.c;ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
        full,first=encoder.spatial(ep,self.s,c,with_first=True)
        ft=encoder.connect(ep,full,self.g,c);dt=encoder.connect(ep,first,self.g,c)
        def predictions(ft,dt):return draft_model.packed_forward(p,ft,dt,self.a,self.n,3,c,full,first)
        fg,dg=jax.grad(lambda ft,dt:jnp.mean(predictions(ft,dt)[1][0,2]**2),argnums=(0,1))(ft,dt)
        np.testing.assert_array_equal(fg[:,2:],0)
        np.testing.assert_array_equal(dg[:,:2],0)
        np.testing.assert_array_equal(dg[:,3:],0)
        self.assertGreater(float(jnp.linalg.norm(fg[0,:2])),1e-7)
        self.assertGreater(float(jnp.linalg.norm(dg[0,2])),1e-7)
        no_main_gradient=jax.grad(lambda dt:jnp.mean(predictions(ft,dt)[0]**2))(dt)
        np.testing.assert_array_equal(no_main_gradient,0)

    def test_first_pass_reuse_and_main_gradient_preservation(self):
        p,c=self.p,self.c;ep={k[8:]:v for k,v in p.items() if k.startswith('encoder.')}
        full,first=encoder.spatial(ep,self.s,c,with_first=True)
        np.testing.assert_array_equal(full,encoder.spatial(ep,self.s,c))
        np.testing.assert_array_equal(first,encoder.spatial(ep,self.s,{**c,'encoder_passes':1}))
        grad=jax.jit(jax.grad(lambda p:jnp.mean(draft_model.forward(p,self.s,self.g,self.a,self.n,c)[0]**2)))(p)
        normal=jax.jit(jax.grad(lambda p:jnp.mean(causal.forward(p,self.s,self.g,self.a,self.n,c)**2)))(p)
        for key in p:np.testing.assert_allclose(grad[key],normal[key],atol=3e-6,rtol=5e-4)

    def test_auxiliary_objective_is_one_shared_teacher_target(self):
        b={'spatial':self.s,'global_features':self.g,'actions':self.a,'counts':self.n,
           'policies':jax.nn.one_hot(self.a,10),'legal':jnp.ones((2,4,10),bool)}
        loss,metrics=policy_model.losses(self.p,b,self.c)
        main,draft=draft_model.forward(self.p,self.s,self.g,self.a,self.n,self.c)
        ce=lambda x:policy_model.averages(policy_model.total_metrics(x,b))['expert_ce']
        np.testing.assert_allclose(loss,.75*ce(main)+.25*ce(draft),rtol=1e-6)
        self.assertEqual(float(metrics['expert_positions']),8)
        self.assertEqual(causal.parameter_schema(self.c),causal.parameter_schema({**self.c,'first_pass_aux_weight':0}))
        gradient=jax.jit(jax.grad(lambda p:policy_model.losses(p,b,self.c)[0]))(self.p)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(gradient)))
        self.assertGreater(float(jnp.linalg.norm(gradient['encoder.blocks.up.weight'])),1e-7)

    @unittest.skipUnless(jax.device_count()>=4,'Requires four host CPU devices')
    def test_global_auxiliary_gradient_with_unequal_live_counts(self):
        from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
        mesh=Mesh(np.asarray(jax.devices()[:4]),('data',))
        b={'spatial':jnp.concatenate((self.s,self.s),0),'global_features':jnp.concatenate((self.g,self.g),0),
           'actions':jnp.concatenate((self.a,self.a),0),'counts':jnp.asarray([4,3,2,0]),
           'policies':jax.nn.one_hot(jnp.concatenate((self.a,self.a),0),10),'legal':jnp.ones((4,4,10),bool)}
        expected=jax.jit(jax.value_and_grad(lambda p:policy_model.losses(p,b,self.c)[0]))(self.p)
        shard=NamedSharding(mesh,P('data'));replicated=NamedSharding(mesh,P())
        batch=jax.tree.map(lambda x:jax.device_put(x,shard),b)
        params=jax.tree.map(lambda x:jax.device_put(x,replicated),self.p)
        objective=jax.shard_map(lambda p,b:policy_model.losses(p,b,self.c,axis_name='data')[0],
            mesh=mesh,in_specs=(P(),jax.tree.map(lambda _:P('data'),b)),out_specs=P(),check_vma=False)
        actual=jax.jit(jax.value_and_grad(objective))(params,batch)
        for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            np.testing.assert_allclose(a,e,atol=5e-6,rtol=8e-4)


if __name__=='__main__':unittest.main()
