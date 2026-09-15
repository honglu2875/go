import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh,PartitionSpec as P
import game_metrics
import evaluate_games
import policy_model

class GameChecks(unittest.TestCase):
    def test_metrics_against_numpy_and_registered_aggregation(self):
        rng=np.random.default_rng(72);z=rng.normal(size=(4,137,10)).astype(np.float32)
        legal=rng.random(z.shape)>.2;legal[...,9]=True
        targets=rng.random(z.shape).astype(np.float32)*legal;targets/=targets.sum(-1,keepdims=True)
        b={'legal':jnp.asarray(legal),'policies':jnp.asarray(targets),'actions':jnp.zeros((4,137),jnp.int32),'counts':jnp.asarray([0,7,70,137])}
        got=jax.jit(game_metrics.totals)(jnp.asarray(z),b)
        for i,n in enumerate([0,7,70,137]):
            scores=np.where(legal[i,:n],z[i,:n],-1e9);scores-=scores.max(-1,keepdims=True)
            lp=scores-np.log(np.exp(scores).sum(-1,keepdims=True))
            self.assertEqual(float(got['count'][i]),n)
            np.testing.assert_allclose(got['ce'][i],-(targets[i,:n]*lp).sum(),atol=1e-4,rtol=2e-6)
            self.assertEqual(float(got['top1'][i]),float((np.argmax(lp,-1)==np.argmax(targets[i,:n],-1)).sum()))
        aggregate=policy_model.total_metrics(jnp.asarray(z),b,stratify=True)
        for k in got:
            expected=aggregate['expert_'+k] if not k.startswith('phase_') else aggregate[k]
            np.testing.assert_allclose(got[k].sum(),expected,atol=1e-4,rtol=2e-6)

    def test_sharded_game_order_for_all_three_model_adapters(self):
        rng=np.random.default_rng(91);b={'spatial':jnp.asarray(rng.normal(size=(4,3,3,3,22)).astype(np.float32)),
            'global_features':jnp.asarray(rng.normal(size=(4,3,19)).astype(np.float32)),
            'actions':jnp.zeros((4,3),jnp.int32),'counts':jnp.asarray([1,3,0,2]),
            'policies':jnp.full((4,3,10),.1),'legal':jnp.ones((4,3,10),bool)}
        base=dict(architecture='causal_visual_policy',width=32,layers=2,mlp_hidden=48,heads=4,kv_heads=2,
            max_board_size=9,max_positions=16,dtype='float32',rematerialize=True,norm_epsilon=1e-6,rope_theta=10000.,attention_backend='xla')
        cnn=dict(architecture='katago_nested_policy',dtype='float32',gpool_width=2,layers=4,max_board_size=19,
            max_positions=512,microbatch=32,mid_width=8,norm_epsilon=.0001,policy_width=4,rematerialize=True,width=16)
        b['spatial']=b['spatial'].at[...,0].set(1.)
        mesh=Mesh(np.asarray(jax.devices()),('data',));self.assertEqual(len(jax.devices()),4)
        for c in [cnn,{**base,'encoder':'overlap_linear'},{**base,'encoder':'overlap_conv64'}]:
            p=jax.jit(lambda:evaluate_games.implementation(c).initialize(91,c))()
            fn=lambda p,b:game_metrics.totals(evaluate_games.logits(p,b,c),b)
            mapped=jax.jit(jax.shard_map(fn,mesh=mesh,in_specs=(P(),{k:P('data') for k in b}),out_specs=P('data'),check_vma=False))
            got=mapped(p,b);expected=jax.jit(fn)(p,b)
            for k in got:
                local=np.concatenate([np.asarray(s.data) for s in sorted(got[k].addressable_shards,key=lambda s:s.index[0].start)],axis=0)
                self.assertTrue(np.isfinite(local).all());self.assertTrue(np.isfinite(np.asarray(expected[k])).all())
                np.testing.assert_allclose(local,expected[k],atol=1e-4,rtol=1e-5)

if __name__=='__main__':unittest.main()
