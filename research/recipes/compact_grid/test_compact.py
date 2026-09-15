"""Compare against the untouched full-grid path with active spatial heads."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import causal
import compact
import draft_model
import policy_model
from test_shared import config


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c={**config(),'first_pass_aux_weight':.25}
        cls.p=causal.initialize(713,cls.c)
        cls.p['head.local.weight']=jnp.linspace(-.12,.08,16)[:,None]
        cls.p['head.local.bias']=jnp.asarray(.07)
        cls.p['head.context.q.weight']=jax.random.normal(jax.random.key(713),(16,4))*.15
        rng=np.random.default_rng(19813)
        cls.b=dict(spatial=jnp.asarray(rng.normal(size=(3,3,3,3,22)),jnp.float32),
            global_features=jnp.asarray(rng.normal(size=(3,3,19)),jnp.float32),
            actions=jnp.asarray([[0,1,9],[7,8,9],[3,5,9]],jnp.int32),counts=jnp.asarray([3,1,0],jnp.int32),
            policies=jax.nn.softmax(jnp.asarray(rng.normal(size=(3,3,10)),jnp.float32),axis=-1),
            legal=jnp.ones((3,3,10),bool))

    def test_outputs_match_with_single_partial_and_oversized_chunks(self):
        p,b,c=self.p,self.b,self.c
        reference=draft_model.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c)
        self.assertGreater(float(jnp.linalg.norm(reference[0]-reference[1])),1e-5)
        for chunk in (1,4,16):
            with self.subTest(chunk=chunk):
                actual=jax.jit(lambda p:compact.forward(p,b['spatial'],b['global_features'],b['actions'],b['counts'],c,chunk_frames=chunk))(p)
                for x,y in zip(actual,reference):np.testing.assert_allclose(x,y,atol=4e-6,rtol=4e-5)
                np.testing.assert_array_equal(actual[0][2],0.)
                np.testing.assert_array_equal(actual[1][1,1:],0.)

    def test_auxiliary_loss_and_every_parameter_gradient(self):
        p,b,c=self.p,self.b,self.c
        reference=jax.jit(jax.value_and_grad(lambda p:policy_model.losses(p,b,c)[0]))(p)
        for inner in (None,False):
            with self.subTest(inner_rematerialize=inner):
                actual=jax.jit(jax.value_and_grad(lambda p:compact.losses(p,b,c,chunk_frames=4,inner_rematerialize=inner)[0]))(p)
                np.testing.assert_allclose(actual[0],reference[0],atol=2e-6,rtol=2e-6)
                for name in p:
                    self.assertTrue(np.isfinite(actual[1][name]).all(),name)
                    np.testing.assert_allclose(actual[1][name],reference[1][name],atol=3e-6,rtol=8e-4,err_msg=name)
                for name in ('encoder.blocks.up.weight','encoder.attention.q.weight','head.context.k.weight','head.context.q.weight'):
                    self.assertGreater(float(jnp.linalg.norm(actual[1][name])),1e-7,name)

    def test_future_frames_actions_and_other_games_cannot_change_current_policy(self):
        p,b,c=self.p,self.b,self.c
        f=jax.jit(lambda s,g,a:compact.forward(p,s,g,a,b['counts'],c,chunk_frames=4))
        old=f(b['spatial'],b['global_features'],b['actions'])
        s=b['spatial'].at[0,2:].set(20).at[1:].set(-12)
        g=b['global_features'].at[0,2:].set(-7).at[1:].set(9)
        actions=b['actions'].at[0,1:].set(9).at[1:].set(0)
        changed=f(s,g,actions)
        for x,y in zip(old,changed):np.testing.assert_array_equal(x[0,:2],y[0,:2])

    def test_compact_outputs_do_not_retain_wide_spatial_features(self):
        c={**self.c,'dtype':'bfloat16','width':768,'encoder_width':768,'policy_context_dim':64}
        result=compact.retained_feature_bytes(batch=8,positions=1536,size=19,c=c)
        self.assertEqual(result['full_and_first_grid_bytes'],2*8*1536*361*768*2)
        self.assertEqual(result['projected_grid_bytes'],2*8*1536*361*65*4)
        self.assertLess(result['compact_outputs_bytes'],result['reference_outputs_bytes']/5)
        shapes=jax.eval_shape(lambda p:compact.encode(p,self.b['spatial'],self.b['global_features'],self.c,chunk_frames=4),self.p)
        self.assertEqual(shapes[2]['key'].shape,(3,3,3,3,4))
        self.assertEqual(shapes[2]['local'].shape,(3,3,3,3))
        self.assertEqual(shapes[2]['key'].dtype,jnp.float32)

    def test_current_full_pass_cannot_leak_into_first_pass_readout(self):
        p,b,c=self.p,self.b,self.c
        full,first,full_features,first_features=compact.encode(p,b['spatial'],b['global_features'],c,chunk_frames=4)
        run=jax.jit(lambda x:compact.packed_forward(p,x,first,b['actions'],b['counts'],3,c,full_features,first_features))
        before=run(full)
        changed=run(full.at[0,1].add(jnp.linspace(-2.,3.,c['width'])))
        np.testing.assert_array_equal(before[1][0,:2],changed[1][0,:2])
        self.assertGreater(float(jnp.linalg.norm(before[0][0,1]-changed[0][0,1])),1e-6)
        self.assertGreater(float(jnp.linalg.norm(before[1][0,2]-changed[1][0,2])),1e-6)
        # The earlier completed full-pass token is allowed into later history.
        self.assertGreater(float(jnp.linalg.norm(jax.grad(lambda x:run(x)[1][0,2,0])(full)[0,1])),1e-8)
        np.testing.assert_array_equal(jax.grad(lambda x:run(x)[1][0,1,0])(full)[0,1:],0.)

    def test_unsupported_architectures_fail_explicitly(self):
        for change in ({'policy_refinement_dim':8},{'policy_readout_kind':'additive'},
                       {'policy_spatial_bias':False},{'first_pass_aux_weight':0}):
            with self.subTest(change=change),self.assertRaises(ValueError):compact.validate({**self.c,**change},4)
        for chunk in (0,1.5,513):
            with self.assertRaises(ValueError):compact.validate(self.c,chunk)
        with self.assertRaises(ValueError):compact.validate(self.c,4,'false')


if __name__=='__main__':unittest.main()
