"""Exercise new readouts after learning, including cached and draft semantics."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import causal
import draft_model
from test_shared import config


class ReadoutTests(unittest.TestCase):
    def test_learned_readouts_are_causal_and_cache_equivalent(self):
        rng=np.random.default_rng(531)
        s=jnp.asarray(rng.normal(size=(1,3,3,3,22)),jnp.float32)
        g=jnp.asarray(rng.normal(size=(1,3,19)),jnp.float32)
        a=jnp.asarray([[0,2,4]]);n=jnp.asarray([3])
        for change in ({'policy_context_dim':8},
                       {'policy_readout_kind':'additive'},
                       {'policy_refinement_dim':8}):
            with self.subTest(change=change):
                c={**config(),**change,'first_pass_aux_weight':.25}
                p=causal.initialize(13,c)
                initial=causal.forward(p,s,g,a,n,c)
                parent_c={**config(),'first_pass_aux_weight':.25}
                parent=causal.initialize(13,parent_c)
                np.testing.assert_array_equal(initial,causal.forward(parent,s,g,a,n,parent_c))
                def loss(p):return -jax.nn.log_softmax(causal.forward(p,s,g,a,n,c),-1)[0,1,7]
                step=jax.jit(jax.grad(loss))
                for _ in range(2):
                    grad=step(p)
                    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(grad)))
                    p={k:v-.03*grad[k] for k,v in p.items()}
                main,draft=draft_model.forward(p,s,g,a,n,c)
                full=causal.forward(p,s,g,a,n,c)
                np.testing.assert_allclose(main,full,atol=3e-5,rtol=3e-5)
                changed=causal.forward(p,s.at[:,2:].set(15),g.at[:,2:].set(-9),a.at[:,1:].set(9),n,c)
                np.testing.assert_array_equal(full[:,:2],changed[:,:2])
                first,cache=causal.first_move(p,s[:,0],g[:,0],c)
                np.testing.assert_allclose(first,full[:,0],atol=3e-5,rtol=3e-5)
                for t in (1,2):
                    cheap={**c,'encoder_passes':1,'first_pass_aux_weight':0}
                    guess,_=causal.append_move(p,cache,a[:,t-1],s[:,t],g[:,t],cheap,attention_positions=t+1)
                    np.testing.assert_allclose(guess,draft[:,t],atol=4e-5,rtol=4e-5)
                    actual,cache=causal.append_move(p,cache,a[:,t-1],s[:,t],g[:,t],c,attention_positions=t+1)
                    np.testing.assert_allclose(actual,full[:,t],atol=3e-5,rtol=3e-5)


if __name__=='__main__':unittest.main()
