"""Causality, KV arithmetic, objective separation and sampler qualification."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import model


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c={'size':3,'komi':0.5,'width':16,'heads':4,'blocks':2,
               'max_tokens':13,'dtype':'float32','expert_temperature':1.,'behavior_temperature':1.}
        cls.p=model.initialize(271,cls.c)
        cls.tokens=jnp.array([[10,0,1,2,3,9,6,11,11,11,11,11,11],
                              [10,8,2,4,1,9,0,11,11,11,11,11,11]],jnp.int32)
        cls.lengths=jnp.array([3,5],jnp.int32)

    def test_future_tokens_and_other_games_cannot_change_a_prefix(self):
        h,_=model.prefill(self.p,self.tokens,self.lengths,self.c)
        changed=self.tokens.at[:,6:].set(7)
        changed=changed.at[1,:].set(5)
        other,_=model.prefill(self.p,changed,jnp.array([12,12]),self.c)
        np.testing.assert_array_equal(h[0,:4],other[0,:4])
        counts=model.historical_counts(self.tokens,self.lengths,10)
        np.testing.assert_array_equal(counts[0,0],np.zeros((2,10)))
        expected=np.zeros((2,10));expected[0,0]=1;expected[1,1]=1;expected[0,2]=1
        np.testing.assert_array_equal(counts[0,3],expected)
        np.testing.assert_array_equal(counts[0,-1],expected)

    def test_cached_append_matches_fresh_causal_prefix_at_variable_lengths(self):
        for dtype,tolerance in [('float32',2e-6),('bfloat16',0.04)]:
            c={**self.c,'dtype':dtype};tokens=self.tokens;lengths=self.lengths
            _,cache=model.prefill(self.p,tokens,lengths,c)
            for moves in (jnp.array([7,6]),jnp.array([9,4])):
                lengths=lengths+1
                h,cache=model.append(self.p,cache,moves,lengths,c)
                tokens=tokens.at[jnp.arange(2),lengths].set(moves)
                all_h,full_cache=model.prefill(self.p,tokens,lengths,c)
                np.testing.assert_allclose(h,all_h[jnp.arange(2),lengths],atol=tolerance,rtol=tolerance)
                for layer,full in zip(cache,full_cache):
                    for a,b in zip(layer,full):
                        for game,length in enumerate(np.asarray(lengths)):
                            np.testing.assert_allclose(np.asarray(a[game,:length+1],np.float32),
                                np.asarray(b[game,:length+1],np.float32),atol=tolerance,rtol=tolerance)

    def test_behavior_targets_cannot_train_the_play_trunk(self):
        shape=self.tokens.shape
        policies=jax.nn.one_hot(jnp.zeros(shape,jnp.int32),10)
        moves=jnp.full(shape,9,jnp.int32);outcomes=jnp.zeros(shape)
        valid=(jnp.arange(shape[1])[None,:]<=self.lengths[:,None]).astype(jnp.float32)
        def component(p,name):return model.losses(p,self.tokens,self.lengths,policies,moves,outcomes,valid,self.c)[1][name]
        gradient=jax.grad(lambda p:component(p,'behavior_loss'))(self.p)
        for key in ('tokens','positions','player','global','blocks','norm','play','value'):
            self.assertTrue(all(np.count_nonzero(x)==0 for x in jax.tree.leaves(gradient[key])),key)
        self.assertGreater(float(jnp.linalg.norm(gradient['behavior_out'])),0.)
        self.assertGreater(float(jnp.linalg.norm(gradient['behavior_style'])),0.)
        expert=jax.grad(lambda p:component(p,'play_loss'))(self.p)
        self.assertGreater(float(jnp.linalg.norm(expert['play'])),0.)
        self.assertGreater(float(jnp.linalg.norm(expert['tokens'])),0.)
        for key in ('behavior_hidden','behavior_style','behavior_bias','behavior_out'):
            self.assertEqual(int(jnp.count_nonzero(expert[key])),0)

    def test_one_compiled_scan_couples_roles_and_declares_own_sampler(self):
        for coupling,oracle in [('independent',False),('shared',True)]:
            run=jax.jit(lambda p,t,l,e,k:model.decode(p,t,l,e,k,self.c,horizon=4,samples=3,
                                                        coupling=coupling,oracle_behavior=oracle))
            actions,logits,noise=jax.device_get(run(self.p,self.tokens,self.lengths,jnp.array([0,2],jnp.uint32),jax.random.key(51)))
            self.assertEqual(actions.shape,(2,2,3,4));self.assertEqual(logits.shape,(2,2,3,4,10))
            for game in range(2):
                for view in range(2):
                    for depth in range(4):
                        if (int(self.lengths[game])+depth)%2==view:
                            np.testing.assert_array_equal(actions[game,view,:,depth],
                                np.argmax(logits[game,view,:,depth]+noise[game,depth],-1))
                if oracle:
                    np.testing.assert_array_equal(actions[game],np.broadcast_to(actions[game,0,0],(2,3,4)))

    def test_own_noise_uses_episode_and_absolute_ply_not_packet_boundaries(self):
        key=jax.random.key(73);episodes=jnp.array([0,1],jnp.uint32)
        def noise(lengths,episodes,horizon):
            return model.decode(self.p,self.tokens,lengths,episodes,key,self.c,horizon=horizon,samples=1)[2]
        long=noise(self.lengths,episodes,4)
        one=noise(self.lengths+2,episodes,1)
        np.testing.assert_array_equal(long[:,2],one[:,0])
        other=noise(self.lengths,episodes+1,1)
        self.assertFalse(np.array_equal(long[:,0],other[:,0]))

    def test_root_mask_applies_to_both_views_without_hiding_raw_own_logits(self):
        legal=jnp.zeros((2,10),jnp.bool_).at[:,4].set(True)
        run=jax.jit(lambda p,t,l:model.decode(p,t,l,jnp.array([0,0],jnp.uint32),jax.random.key(91),self.c,
            horizon=4,samples=3,root_legal=legal))
        actions,logits,noise=jax.device_get(run(self.p,self.tokens,self.lengths))
        np.testing.assert_array_equal(actions[...,0],np.full((2,2,3),4,np.int32))
        self.assertTrue(np.isfinite(logits).all());self.assertTrue(np.isfinite(noise).all())


if __name__=='__main__':unittest.main()
