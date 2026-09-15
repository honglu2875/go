"""Compare the complete attention update with and without data-parallel partitioning."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import model


@unittest.skipUnless(jax.default_backend()=='cpu' and jax.device_count()>=2,
                     'Use CPU with XLA_FLAGS=--xla_force_host_platform_device_count=4')
class ShardingTests(unittest.TestCase):
    def test_global_attention_loss_and_momentum_update_match_one_device(self):
        config={'width':16,'blocks':2,'groups':4,'value_hidden':16,'dtype':'float32','score_utility_factor':0.3,'score_scale':2.0,'architecture':'attention','heads':2,'mlp_ratio':4,'max_board_size':26}
        learner={'learning_rate':0.02,'momentum':0.9,'l2':0.0001,'max_grad_norm':5.0,'augment_symmetries':True,'ownership_weight':1.5,'score_weight':1.0}
        rng=np.random.default_rng(27)
        x=rng.normal(size=(8,3,3,8)).astype(np.float32);x[...,-1]=(rng.uniform(size=(8,3,3))>0.3)
        pi=rng.uniform(size=(8,10)).astype(np.float32);pi[:,:9]*=x[...,-1].reshape(8,9);pi/=pi.sum(-1,keepdims=True)
        z=np.asarray([-1,1]*4,np.float32)
        owner=rng.choice([-1.,0.,1.],size=(8,3,3)).astype(np.float32)
        p=model.initialize(27,8,config);v=jax.tree.map(jnp.zeros_like,p)
        mesh=Mesh(np.array(jax.devices()),('data',));rep=NamedSharding(mesh,P());batch=NamedSharding(mesh,P('data'))
        dp=jax.device_put(jax.tree.map(np.asarray,p),rep);dv=jax.device_put(jax.tree.map(np.asarray,v),rep)
        bx,bpi,bz,bo=(jax.device_put(a,batch) for a in (x,pi,z,owner))
        single=jax.jit(lambda p,v,x,pi,z,o,key:model.update(p,v,x,pi,z,o,key,config,learner))
        distributed=jax.jit(lambda p,v,x,pi,z,o,key:model.update(p,v,x,pi,z,o,key,config,learner),
                            in_shardings=(rep,rep,batch,batch,batch,batch,rep),out_shardings=(rep,rep,rep))
        for step in range(3):
            key=jax.random.key(step)
            dk=jax.random.wrap_key_data(jax.device_put(np.asarray(jax.random.key_data(key)),rep))
            p,v,metrics=single(p,v,x,pi,z,owner,key)
            dp,dv,dmetrics=distributed(dp,dv,bx,bpi,bz,bo,dk)
            for a,b in zip(jax.tree.leaves((p,v,metrics)),jax.tree.leaves((dp,dv,dmetrics))):
                np.testing.assert_allclose(a,b,rtol=1e-5,atol=2e-6)
