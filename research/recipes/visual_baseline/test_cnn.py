import copy
import json
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import cnn

class CNNQualification(unittest.TestCase):
    def test_causality_history_and_microbatch(self):
        c=json.loads(Path(__file__).with_name('cpu_cnn.json').read_text())['model']
        p=cnn.initialize(7,c); random=np.random.default_rng(11)
        o=random.normal(size=(2,8,3,3,6)).astype(np.float32)
        a=random.integers(0,10,size=(2,8),dtype=np.int32); n=np.array([8,7],np.int32)
        run=jax.jit(lambda p,o,a:cnn.forward(p,o,a,n,c))
        out=run(p,o,a)
        altered=o.copy(); altered[:,5:]+=2
        actions=a.copy(); actions[:,4:]=(actions[:,4:]+1)%10
        changed=run(p,altered,actions)
        for k in out: np.testing.assert_array_equal(out[k][:,:5],changed[k][:,:5])
        # Earliest history and its action fall outside the last three frames.
        altered=o.copy(); altered[:,:3]+=2
        actions=a.copy(); actions[:,:2]=(actions[:,:2]+1)%10
        changed=run(p,altered,actions)
        for k in out: np.testing.assert_array_equal(out[k][:,6:],changed[k][:,6:])
        partition=jax.jit(lambda:cnn.forward(p,o,a,n,{**c,'microbatch':2}))()
        for k in out: np.testing.assert_allclose(out[k],partition[k],atol=2e-6,rtol=2e-6)
        # Current action cannot enter its own prediction; previous action does.
        f=np.asarray(cnn.history_features(jnp.asarray(o),jnp.asarray(a),c))
        aa=a.copy(); aa[:,3]=(aa[:,3]+1)%10
        ff=np.asarray(cnn.history_features(jnp.asarray(o),jnp.asarray(aa),c))
        np.testing.assert_array_equal(f[:,:4],ff[:,:4]);self.assertGreater(np.abs(f[:,4]-ff[:,4]).sum(),0)
        self.assertTrue(all(np.all(np.asarray(v)[1,7]==0) for v in out.values()))
        # Independent final-frame path used by an inference owner must agree.
        final=cnn.frames(p,jnp.asarray(f[:,6]),c)
        for k in out: np.testing.assert_allclose(final[k],out[k][:,6],atol=2e-6,rtol=2e-6)
    def test_gradient(self):
        c=json.loads(Path(__file__).with_name('cpu_cnn.json').read_text())['model']
        p=cnn.initialize(9,c); o=jnp.ones((1,4,3,3,6));a=jnp.zeros((1,4),jnp.int32);n=jnp.array([4])
        def loss(p):
            r=cnn.forward(p,o,a,n,c)
            return jnp.mean(r['expert_logits']**2)+jnp.mean(r['behavior_logits']**2)+jnp.mean(r['value']**2)
        g=jax.jit(jax.grad(loss))(p)
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g)))
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in jax.tree.leaves(g)),0)

if __name__=='__main__':unittest.main()
