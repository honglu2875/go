import os
os.environ['JAX_PLATFORMS']='cpu'
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import model


class ModelTests(unittest.TestCase):
    def test_attention_matches_independent_per_head_reference(self):
        rng=np.random.default_rng(81)
        x=rng.normal(size=(2,5,8)).astype(np.float32)
        block={k:rng.normal(size=s).astype(np.float32)/3 for k,s in
               [('qkv',(8,24)),('projection',(8,8))]}
        bias=rng.normal(size=(2,5,5)).astype(np.float32)/4
        q,k,v=np.split(x@block['qkv'],3,axis=-1)
        expected=np.empty_like(x)
        for batch in range(2):
            for head in range(2):
                region=slice(head*4,(head+1)*4)
                scores=q[batch,:,region]@k[batch,:,region].T/2+bias[head]
                weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
                weights/=weights.sum(axis=-1,keepdims=True)
                expected[batch,:,region]=weights@v[batch,:,region]
        expected=expected@block['projection']
        actual=model.attention(jnp.asarray(x),block,jnp.asarray(bias),2,jnp.float32)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)

    def test_relative_offsets_are_d4_invariant_and_respect_distances(self):
        for size in (2,5,9):
            indices=np.asarray(model.relative_indices(size));grid=np.arange(size*size).reshape(size,size)
            self.assertEqual(indices[0,-1],size*(size+1)//2-1)
            for rotation in range(4):
                for flip in (False,True):
                    transformed=np.rot90(grid,rotation)
                    if flip:transformed=np.flip(transformed,axis=1)
                    perm=transformed.reshape(-1)
                    np.testing.assert_array_equal(indices,indices[np.ix_(perm,perm)])
        with self.assertRaises(ValueError):model.apply(self.params,jnp.ones((1,27,27,8)),self.config)

    def test_attention_projection_gradient_matches_finite_difference(self):
        pi=jax.nn.one_hot(jnp.array([0,9]),10);z=jnp.array([-1.,1.]);owner=jnp.ones((2,3,3))
        def loss(value):
            block={**self.params['blocks'][0]}
            block['qkv']=block['qkv'].at[2,3].set(value)
            params={**self.params,'blocks':[block]}
            return model.losses(params,self.x,pi,z,owner,self.config,{'l2':0.,'ownership_weight':1.5,'score_weight':1.})[0]
        value=self.params['blocks'][0]['qkv'][2,3];epsilon=0.01
        analytic=float(jax.grad(loss)(value));numeric=float((loss(value+epsilon)-loss(value-epsilon))/(2*epsilon))
        self.assertAlmostEqual(analytic,numeric,delta=0.0003)
        self.assertGreater(abs(analytic),1e-6)

    def test_score_labels_include_signed_komi_and_reverse_with_perspective(self):
        x=jnp.zeros((2,1,1,8)).at[:,0,0,-3].set(jnp.array([2.,-2.]))
        owners=jnp.zeros((2,1,1))
        np.testing.assert_allclose(model.score_targets(x,owners,2.),[0.5,-0.5],atol=1e-7,rtol=0)

    def test_score_head_gradient_and_utility_do_not_replace_win_target(self):
        pi=jax.nn.one_hot(jnp.array([0,9]),10);z=jnp.array([-1.,1.]);owner=jnp.ones((2,3,3))
        learner={'l2':0.,'ownership_weight':0.,'score_weight':1.}
        def loss(bias):return model.losses({**self.params,'score_bias':bias},self.x,pi,z,owner,self.config,learner)[0]
        bias=self.params['score_bias'];epsilon=0.01
        derivative=float(jax.grad(loss)(bias)[0])
        finite=float((loss(bias+epsilon)-loss(bias-epsilon))/(2*epsilon))
        self.assertAlmostEqual(derivative,finite,delta=0.0002)
        logits,win,_,score=model.apply(self.params,self.x,self.config,with_targets=True)
        pi_out,utility=model.apply(self.params,self.x,self.config)
        np.testing.assert_array_equal(logits,pi_out)
        np.testing.assert_allclose(utility,(win+0.3*score)/1.3,atol=1e-7,rtol=0)
        control={**self.config,'score_utility_factor':0.}
        np.testing.assert_array_equal(model.apply(self.params,self.x,control)[1],win)

    def setUp(self):
        self.config={'width':8,'blocks':1,'groups':2,'value_hidden':8,'dtype':'float32','score_utility_factor':0.3,'score_scale':2.0,'architecture':'attention','heads':2,'mlp_ratio':4,'max_board_size':26}
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
        owner=np.arange(9,dtype=np.float32).reshape(3,3)/4-1
        xs,ps,os=model.augment(jnp.asarray(np.stack([features]*8)),jnp.asarray(np.stack([policy]*8)),jnp.asarray(np.stack([owner]*8)),jnp.arange(8))
        for s in range(8):
            expected_x=np.rot90(features,s%4,axes=(0,1));expected_p=np.rot90(policy[:-1].reshape(3,3),s%4)
            if s>=4:expected_x=np.flip(expected_x,axis=1);expected_p=np.flip(expected_p,axis=1)
            np.testing.assert_array_equal(xs[s],expected_x)
            np.testing.assert_array_equal(ps[s,:-1],expected_p.reshape(-1));self.assertEqual(float(ps[s,-1]),9.0)
            expected_o=np.rot90(owner,s%4)
            if s>=4:expected_o=np.flip(expected_o,axis=1)
            np.testing.assert_array_equal(os[s],expected_o)

    def test_loss_gradient_matches_finite_difference(self):
        targets=jax.nn.one_hot(jnp.array([0,9]),10);z=jnp.array([-1.,1.])
        cfg={'l2':0.0001,'ownership_weight':1.5,'score_weight':0.}
        owner=jnp.ones((2,3,3))
        def loss(bias):return model.losses({**self.params,'policy_bias':bias},self.x,targets,z,owner,self.config,cfg)[0]
        analytic=float(jax.grad(loss)(self.params['policy_bias']))
        epsilon=0.01
        numeric=float((loss(self.params['policy_bias']+epsilon)-loss(self.params['policy_bias']-epsilon))/(2*epsilon))
        self.assertAlmostEqual(analytic,numeric,delta=0.0002)

    def test_ownership_gradient_and_zero_coefficient_control(self):
        pi=jax.nn.one_hot(jnp.array([0,9]),10);z=jnp.array([-1.,1.])
        owner=jnp.array([[[1.,0.,-1.]]*3,[[1.,1.,0.]]*3])
        def loss(bias,weight):
            return model.losses({**self.params,'ownership_bias':bias},self.x,pi,z,owner,self.config,{'l2':0.,'ownership_weight':weight,'score_weight':0.})[0]
        bias=self.params['ownership_bias'];epsilon=0.01
        analytic=float(jax.grad(loss)(bias,1.5))
        numeric=float((loss(bias+epsilon,1.5)-loss(bias-epsilon,1.5))/(2*epsilon))
        self.assertAlmostEqual(analytic,numeric,delta=0.0002)
        self.assertEqual(float(jax.grad(loss)(bias,0.)),0.)
        # Changing only labels cannot affect any parameter gradient at weight zero.
        def gradient(labels):return jax.grad(model.losses,has_aux=True)(self.params,self.x,pi,z,labels,self.config,{'l2':0.,'ownership_weight':0.,'score_weight':0.})[0]
        for a,b in zip(jax.tree.leaves(gradient(owner)),jax.tree.leaves(gradient(-owner))):np.testing.assert_array_equal(a,b)
        default=model.apply(self.params,self.x,self.config)
        auxiliary=model.apply(self.params,self.x,self.config,with_ownership=True)
        for a,b in zip(default,auxiliary):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
