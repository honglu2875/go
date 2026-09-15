import os
os.environ['JAX_PLATFORMS']='cpu'
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config={'width':8,'blocks':1,'groups':2,'value_hidden':8,'dtype':'float32'}
        self.params=model.initialize(27,8,self.config)
        self.x=np.random.default_rng(3).normal(size=(2,3,3,8)).astype(np.float32)
        self.x[...,-1]=1

    def test_padding_other_examples_does_not_change_an_example(self):
        single=model.apply(self.params,jnp.asarray(self.x[:1]),self.config)
        padded=model.apply(self.params,jnp.concatenate([self.x[:1],jnp.zeros((3,3,3,8))]),self.config)
        for a,b in zip(single,padded):np.testing.assert_allclose(a,b[:1],rtol=1e-5,atol=1e-6)
        other=model.apply(self.params,jnp.ones((2,5,5,8)),self.config)
        self.assertEqual(other[0].shape,(2,26));self.assertEqual(other[1].shape,(2,))

    def test_all_symmetries_transform_spatial_targets_and_preserve_pass(self):
        features=np.arange(9*8,dtype=np.float32).reshape((3,3,8))
        policy=np.arange(10,dtype=np.float32)
        xs,ps=model.augment(jnp.asarray(np.stack([features]*8)),jnp.asarray(np.stack([policy]*8)),jnp.arange(8))
        for s in range(8):
            expected_x=np.rot90(features,s%4,axes=(0,1));expected_p=np.rot90(policy[:-1].reshape(3,3),s%4)
            if s>=4:expected_x=np.flip(expected_x,axis=1);expected_p=np.flip(expected_p,axis=1)
            np.testing.assert_array_equal(xs[s],expected_x)
            np.testing.assert_array_equal(ps[s,:-1],expected_p.reshape(-1));self.assertEqual(float(ps[s,-1]),9.0)

    def test_loss_gradient_matches_finite_difference(self):
        targets=jax.nn.one_hot(jnp.array([0,9]),10);z=jnp.array([-1.,1.])
        cfg={'l2':0.0001}
        def loss(bias):return model.losses({**self.params,'policy_bias':bias},self.x,targets,z,self.config,cfg)[0]
        analytic=float(jax.grad(loss)(self.params['policy_bias']))
        epsilon=0.01
        numeric=float((loss(self.params['policy_bias']+epsilon)-loss(self.params['policy_bias']-epsilon))/(2*epsilon))
        self.assertAlmostEqual(analytic,numeric,delta=0.0002)


if __name__=='__main__':unittest.main()
