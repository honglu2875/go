"""Independent population arithmetic and four-device gradient qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

import heads


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fixture():
    rng=np.random.default_rng(93147)
    prediction=rng.uniform(-.95,.95,(8,300)).astype(np.float32)
    targets=rng.uniform(-.9,.9,(8,300)).astype(np.float32)
    counts=np.asarray([270,100,17,1,0,0,0,0],np.int32)
    live=np.arange(300)[None,:]<counts[:,None]
    prediction[~live]=np.nan;targets[~live]=np.nan
    b=dict(values=targets,counts=counts,
           family_weights=np.asarray([1/370,1/370,1/17,1,np.nan,np.nan,np.nan,np.nan],np.float32),
           opponent=np.asarray([0,0,1,7,2,5,6,-1],np.int32))
    return prediction,b


def independent_reference(prediction,b):
    # Build explicit lists of real observations. No padded arithmetic, JAX
    # masking, or helper from the implementation enters the reference.
    groups={'value':[], 'value_family':[]}
    groups.update({f'value_opponent_{i}':[] for i in range(8)})
    phases=((0,16),(16,64),(64,128),(128,256),(256,2048))
    groups.update({f'value_phase_{lo}_{hi}':[] for lo,hi in phases})
    for game,count in enumerate(b['counts']):
        for position in range(count):
            p=float(prediction[game,position]);t=float(b['values'][game,position])
            groups['value'].append((p,t,1.))
            groups['value_family'].append((p,t,float(b['family_weights'][game])))
            groups[f"value_opponent_{b['opponent'][game]}"].append((p,t,1.))
            for lo,hi in phases:
                if lo<=position<hi:
                    groups[f'value_phase_{lo}_{hi}'].append((p,t,1.))
    totals={};averages={}
    for name,rows in groups.items():
        if rows:
            p,t,w=np.asarray(rows,np.float64).T
            count=w.sum();mean=np.average(t,weights=w)
            raw=(count,np.dot((p-t)**2,w),np.dot(np.abs(p-t),w),np.dot(t,w),np.dot(t*t,w),np.dot(p,w))
            avg=(count,np.average((p-t)**2,weights=w),np.average(np.abs(p-t),weights=w),mean,
                 np.average(p,weights=w),np.average(t*t,weights=w),np.average((t-mean)**2,weights=w))
        else:
            raw=(0.,)*6;avg=(0.,)*7
        for key,value in zip(('count','squared_error','absolute_error','target_sum','target_squared','prediction_sum'),raw):
            totals[name+'_'+key]=value
        for key,value in zip(('count','mse','mae','mean_target','mean_prediction','zero_predictor_mse','fitted_constant_mse'),avg):
            averages[name+'_'+key]=value
    return totals,averages


class Qualification(unittest.TestCase):
    def assert_tree_close(self,actual,expected):
        self.assertEqual(jax.tree.structure(actual),jax.tree.structure(expected))
        for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
            np.testing.assert_allclose(a,e,rtol=3e-5,atol=3e-6)

    def test_population_strata_and_fitted_baselines_against_numpy(self):
        prediction,b=fixture();raw,avg=independent_reference(prediction,b)
        actual=heads.value_totals(jnp.asarray(prediction),jax.tree.map(jnp.asarray,b),stratify=True)
        self.assert_tree_close(actual,raw)
        self.assert_tree_close(heads.value_averages(actual),avg)
        self.assertEqual(float(actual['value_count']),388.)
        self.assertAlmostEqual(float(actual['value_family_count']),3.,places=5)
        self.assertEqual(float(actual['value_phase_256_2048_count']),14.)

    def test_fractional_family_mass_is_normalized(self):
        prediction,b=fixture();b['family_weights'][:4]/=16
        raw,avg=independent_reference(prediction,b)
        actual=heads.value_averages(heads.value_totals(jnp.asarray(prediction),jax.tree.map(jnp.asarray,b),stratify=True))
        self.assertAlmostEqual(float(actual['value_family_count']),3/16,places=6)
        self.assert_tree_close(actual,avg)

    def test_nan_padding_has_zero_gradient_and_all_padding_is_finite(self):
        prediction,b=fixture();b=jax.tree.map(jnp.asarray,b)
        objective=lambda x:heads.value_averages(heads.value_totals(x,b))['value_mse']
        _,grad=jax.jit(jax.value_and_grad(objective))(jnp.asarray(prediction))
        expected=np.zeros_like(prediction)
        for i,count in enumerate(np.asarray(b['counts'])):
            expected[i,:count]=2*(prediction[i,:count]-np.asarray(b['values'])[i,:count])/388
        np.testing.assert_allclose(grad,expected,rtol=2e-6,atol=1e-8)
        b={**b,'counts':jnp.zeros(8,jnp.int32)}
        result=heads.value_averages(heads.value_totals(jnp.full((8,300),jnp.nan),b,stratify=True))
        self.assertTrue(all(float(x)==0. for x in result.values()))
        objective=lambda x:heads.value_averages(heads.value_totals(x,b))['value_mse']
        value,grad=jax.jit(jax.value_and_grad(objective))(jnp.full((8,300),jnp.nan))
        self.assertEqual(float(value),0.);np.testing.assert_array_equal(grad,np.zeros((8,300)))

    def test_four_device_raw_totals_with_two_empty_shards(self):
        self.assertEqual(jax.device_count(),4,'Must run with four simulated CPU devices')
        prediction,b=fixture();mesh=Mesh(np.asarray(jax.devices()),('data',))
        operation=jax.shard_map(lambda x,b:heads.value_totals(x,b,axis_name='data',stratify=True),
            mesh=mesh,in_specs=(P('data'),jax.tree.map(lambda _:P('data'),b)),out_specs=P(),check_vma=False)
        actual=jax.jit(operation)(jnp.asarray(prediction),jax.tree.map(jnp.asarray,b))
        raw,_=independent_reference(prediction,b);self.assert_tree_close(actual,raw)

    def test_replicated_parameter_gradient_matches_global_objective(self):
        self.assertEqual(jax.device_count(),4,'Must run with four simulated CPU devices')
        rng=np.random.default_rng(4012);_,b=fixture()
        b=jax.tree.map(jnp.asarray,b);x=jnp.asarray(rng.normal(size=(8,300,12)).astype(np.float32))
        params=heads.initialize(8294,kind='temporal',width=12,hidden=7)
        def objective(p,x,b,axis_name=None):
            prediction=heads.signed_value(heads.temporal(p,x))
            return heads.value_averages(heads.value_totals(prediction,b,axis_name=axis_name))['value_mse']
        expected=jax.jit(jax.value_and_grad(objective))(params,x,b)
        mesh=Mesh(np.asarray(jax.devices()),('data',))
        replicated=NamedSharding(mesh,P());sharded=NamedSharding(mesh,P('data'))
        operation=jax.shard_map(lambda p,x,b:objective(p,x,b,'data'),mesh=mesh,
            in_specs=(P(),P('data'),jax.tree.map(lambda _:P('data'),b)),out_specs=P(),check_vma=False)
        actual=jax.jit(jax.value_and_grad(operation))(
            jax.tree.map(lambda a:jax.device_put(a,replicated),params),jax.device_put(x,sharded),
            jax.tree.map(lambda a:jax.device_put(a,sharded),b))
        self.assert_tree_close(actual,expected)
        self.assertTrue(all(np.isfinite(g).all() for g in jax.tree.leaves(actual)))
        self.assertTrue(all(float(jnp.linalg.norm(g))>0. for g in jax.tree.leaves(actual[1])))

    def test_value_perspective_and_full_width_temporal_head(self):
        logits=jnp.asarray([[4.,-2.,1.],[-2.,4.,1.],[0.,0.,20.]])
        value=np.asarray(heads.signed_value(logits))
        self.assertGreater(value[0],.9);self.assertAlmostEqual(float(value[0]),-float(value[1]),places=7)
        self.assertAlmostEqual(float(value[2]),0.,places=7)
        params=jax.eval_shape(lambda:heads.initialize(0,kind='temporal'))
        self.assertEqual({k:v.shape for k,v in params.items()},
            {'linear2.weight':(768,256),'linear2.bias':(256,),
             'linear_valuehead.weight':(256,3),'linear_valuehead.bias':(3,)})
        self.assertEqual(sum(v.size for v in params.values()),197635)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification)
    names=[x.id() for x in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(kind='joint_value_metric_cpu_qualification',status='passed' if result.wasSuccessful() else 'failed',
        elapsed_seconds=time.monotonic()-start,tests_run=result.testsRun,test_names=names,
        failures=[(t.id(),detail) for t,detail in result.failures+result.errors],
        device_count=jax.device_count(),jax_version=jax.__version__,
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(heads.__file__))},
        scope='Independent population totals, fractional family mass, padding, four-device unequal counts and replicated value-head gradients; not joint backbone or TPU qualification')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status=report['status'],tests=report['tests_run'],seconds=report['elapsed_seconds'])),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
