import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,PartitionSpec as P
import policy_model
import audit_population
from types import SimpleNamespace
import copy


class PopulationMetricsTests(unittest.TestCase):
    def batch(self):
        prediction=jnp.asarray([[[2.,0.],[1.,0.]],[[0.,2.],[0.,1.]],[[0.,0.],[12.,-12.]],[[9.,-9.],[9.,-9.]]])
        batch=dict(actions=jnp.zeros((4,2),jnp.int32),counts=jnp.asarray([2,2,1,0]),
            policies=jnp.broadcast_to(jnp.asarray([.75,.25]),(4,2,2)),legal=jnp.ones((4,2,2),bool),
            opponent=jnp.asarray([0,0,1,-1]),family_weights=jnp.asarray([.25,.25,1.,0.]))
        return prediction,batch

    def test_equal_family_mass_and_exact_population_counts(self):
        prediction,batch=self.batch()
        totals=policy_model.total_metrics(prediction,batch,stratify=True)
        metrics=policy_model.averages(totals)
        logp=np.asarray(jax.nn.log_softmax(prediction,-1));ce=-np.sum(np.asarray(batch['policies'])*logp,-1)
        self.assertEqual(float(metrics['expert_count']),5)
        self.assertEqual(float(metrics['family_count']),2)
        self.assertEqual(float(metrics['opponent_0_count']),4)
        self.assertEqual(float(metrics['opponent_1_count']),1)
        self.assertEqual(float(metrics['opponent_7_count']),0)
        self.assertAlmostEqual(float(metrics['expert_ce']),(ce[:2].sum()+ce[2,0])/5,places=6)
        self.assertAlmostEqual(float(metrics['family_ce']),(ce[:2].sum()/4+ce[2,0])/2,places=6)

    def test_split_host_reduction_matches_unsplit_totals(self):
        if len(jax.devices())!=4:self.skipTest('Requires four simulated CPU devices')
        prediction,batch=self.batch();mesh=Mesh(np.asarray(jax.devices()),('data',))
        fn=jax.shard_map(lambda x,b:policy_model.total_metrics(x,b,axis_name='data',stratify=True),
            mesh=mesh,in_specs=(P('data'),jax.tree.map(lambda _:P('data'),batch)),out_specs=P(),check_vma=False)
        actual=jax.jit(fn)(prediction,batch);expected=policy_model.total_metrics(prediction,batch,stratify=True)
        for name in expected:np.testing.assert_allclose(actual[name],expected[name],rtol=2e-6,atol=1e-6,err_msg=name)

    def test_independent_target_audit_rejects_population_and_entropy_changes(self):
        prediction,batch=self.batch()
        games=[dict(split=1,opening_family=b'a',length=2,opponent=0),
               dict(split=1,opening_family=b'a',length=2,opponent=0),
               dict(split=1,opening_family=b'b',length=1,opponent=1)]
        data=SimpleNamespace(shards=[dict(games=games,expert_offsets=np.asarray([0,2,4,5]),
             policies=np.tile(np.asarray([.75,.25],np.float32),(5,1)))])
        expected=audit_population.population(data,[('expert',0,i) for i in range(3)])
        totals=policy_model.total_metrics(prediction,batch,stratify=True)
        row=dict(split=1,episode_ids_sha256=expected['episode_ids_sha256'],
            raw_totals={k:float(v) for k,v in totals.items()},
            metrics={k:float(v) for k,v in policy_model.averages(totals).items()})
        audit_population.check(row,expected,split=1)
        for key in ('opponent_0_count','family_count','family_target_entropy'):
            corrupt=copy.deepcopy(row);corrupt['raw_totals'][key]+=1
            with self.assertRaises(ValueError):audit_population.check(corrupt,expected,split=1)
        data.shards[0]['games'][0]['split']=2
        with self.assertRaises(ValueError):audit_population.population(data,[('expert',0,0)])


if __name__=='__main__':unittest.main()
