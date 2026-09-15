"""Meaningful distributed-objective, normalization and optimizer qualification."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
import katago
import policy_model as model
import policy_optimizer as optimizer

C=dict(architecture='katago_nested_policy',width=16,mid_width=8,gpool_width=2,policy_width=4,layers=4,
       dtype='float32',rematerialize=True,microbatch=2,max_board_size=5,max_positions=512,norm_epsilon=1e-4)

class Qualification(unittest.TestCase):
    def test_main_only_objective_has_no_helper_gradient_or_batch_dependency(self):
        rng=np.random.default_rng(1729)
        spatial=rng.normal(size=(2,1,3,3,22)).astype(np.float32)*.1
        spatial[...,0]=1.
        b=dict(spatial=spatial,global_features=np.zeros((2,1,19),np.float32),
               counts=np.ones(2,np.int32),actions=np.zeros((2,1),np.int32),
               policies=np.full((2,1,10),.1,np.float32),legal=np.ones((2,1,10),bool))
        params=jax.jit(lambda:katago.initialize(49,C))()
        main=model.logits(params,b,C,training=False)
        expected=model.averages(model.total_metrics(main,b))['expert_ce']
        value,gradient=jax.jit(jax.value_and_grad(lambda p:model.losses(p,b,C)[0]))(params)
        np.testing.assert_allclose(value,expected,rtol=1e-6,atol=1e-6)
        names=[k for k in params if k.startswith(('intermediate_','norm_intermediate_'))]
        self.assertTrue(names)
        for name in names:np.testing.assert_array_equal(gradient[name],jnp.zeros_like(gradient[name]))
        self.assertGreater(float(jnp.linalg.norm(gradient['policy_head.conv2p.weight'])),0.)
        perturbed={**params,**{k:jnp.full_like(params[k],100.) for k in names}}
        changed={**b,'spatial':b['spatial'].copy()};changed['spatial'][1,:,1:,1:,:]=5.
        again=model.logits(perturbed,changed,C,training=False)
        np.testing.assert_allclose(main[0],again[0],rtol=1e-6,atol=1e-6)

    def test_phase_metrics_partition_live_positions(self):
        rng=np.random.default_rng(519);counts=np.asarray([7,33,131,258],np.int32)
        policies=np.full((4,300,5),.2,np.float32)
        b=dict(actions=np.zeros((4,300),np.int32),counts=counts,policies=policies,legal=np.ones((4,300,5),bool))
        raw=model.total_metrics(jnp.asarray(rng.normal(size=(4,300,5)),jnp.float32),b,stratify=True)
        n=sum(int(raw[f'phase_{lo}_{hi}_count']) for lo,hi in model.PHASES)
        self.assertEqual(n,int(raw['expert_count']));self.assertEqual(n,int(counts.sum()))
        for metric in ('ce','target_entropy'):
            total=sum(float(raw[f'phase_{lo}_{hi}_{metric}']) for lo,hi in model.PHASES)
            np.testing.assert_allclose(total,float(raw['expert_'+metric]),rtol=2e-6)
        self.assertTrue(all(np.isfinite(float(v)) for v in model.averages(raw).values()))

    def test_global_objective_gradient_and_optimizer(self):
        self.assertEqual(len(jax.devices()),4,'Run with four virtual CPU devices')
        rng=np.random.default_rng(719)
        counts=np.asarray([2,1,2,0,1,2,2,1],np.int32); live=np.arange(2)[None,:]<counts[:,None]
        spatial=rng.integers(0,2,(8,2,3,3,22)).astype(np.float32)*live[:,:,None,None,None]
        spatial[...,0]=live[:,:,None,None]
        glob=rng.normal(size=(8,2,19)).astype(np.float32)*.1*live[:,:,None]
        target=rng.uniform(.1,1,(8,2,10)).astype(np.float32);target/=target.sum(-1,keepdims=True)
        b=dict(spatial=spatial,global_features=glob,counts=counts,actions=np.zeros((8,2),np.int32),
               policies=target,legal=np.ones((8,2,10),bool))
        params=jax.jit(lambda:katago.initialize(49,C))()
        whole=jax.jit(jax.value_and_grad(lambda p,b:model.losses(p,b,C)[0]))
        value,gradient=whole(params,b)
        mesh=Mesh(np.asarray(jax.devices()),('data',)); rep=NamedSharding(mesh,P()); sh=NamedSharding(mesh,P('data'))
        objective=jax.shard_map(lambda p,b:model.losses(p,b,C,axis_name='data')[0],mesh=mesh,
            in_specs=(P(),jax.tree.map(lambda _:P('data'),b)),out_specs=P(),check_vma=False)
        val2,g2=jax.jit(jax.value_and_grad(objective))(jax.device_put(params,rep),jax.device_put(b,sh))
        np.testing.assert_allclose(value,val2,atol=2e-6,rtol=2e-5)
        for key in gradient: np.testing.assert_allclose(gradient[key],g2[key],atol=2e-6,rtol=3e-4,err_msg=key)
        state=optimizer.initialize(params)
        step=jax.jit(lambda p,s,g:optimizer.apply_gradient(p,s,g,value,learning_rate=1e-4))
        p1,s1,m1=step(params,state,gradient); p2,s2,m2=step(params,state,g2)
        # Tiny near-zero Adam gradients can magnify reduction differences;
        # test moments tightly and updates with an explicit absolute bound.
        for key in params:
            np.testing.assert_allclose(s1['first'][key],s2['first'][key],atol=2e-7,rtol=5e-4)
            np.testing.assert_allclose(p1[key],p2[key],atol=2e-5,rtol=2e-5)
        self.assertTrue(bool(m1['accepted'])); self.assertTrue(bool(m2['accepted']))
        poisoned={**gradient,next(iter(gradient)):jnp.full_like(next(iter(gradient.values())),jnp.nan)}
        rejected,rs,metrics=step(params,state,poisoned)
        self.assertFalse(bool(metrics['accepted']))
        for a,b in zip(jax.tree.leaves((params,state)),jax.tree.leaves((rejected,rs))): np.testing.assert_array_equal(a,b)

    def test_decay_and_repvgg_cover_stacked_parameters(self):
        schema=katago.parameter_schema(C)
        for s in schema:
            if s['path'].endswith(('.gamma','.beta','.bias')): self.assertFalse(optimizer.decay(s['path']))
        p=katago.initialize(1,C); g=jax.tree.map(jnp.ones_like,p); scaled=katago.repvgg_gradient(g)
        mapping=katago.reference_names(p,C)
        for key,items in mapping.items():
            name=items[0][0]; a=np.asarray(scaled[key])
            selected='normactconv' in name and '.conv.weight' in name and a.shape[-4:-2]==(3,3)
            if selected:
                self.assertTrue(np.all(a[...,1,1,:,:]==2))
                self.assertTrue(np.all(a[...,0,0,:,:]==1))
            else: self.assertTrue(np.all(a==1))

if __name__=='__main__': unittest.main()
