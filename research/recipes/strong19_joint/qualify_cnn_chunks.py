"""Check bounded CNN helper statistics and every coupled parameter gradient."""
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

import cnn_chunks
import joint
import katago
from qualify_joint import configs,fixture


class Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=configs()['cnn'];cls.b=fixture()
        cls.p=joint.initialize(213,cls.c,dict(hidden=7,spatial_channels=5))
        cls.p['norm_intermediate_trunkfinal.gamma']=jnp.linspace(-.05,.05,16)
        cls.p['norm_intermediate_trunkfinal.beta']=jnp.linspace(-.02,.03,16)

    def assert_close(self,a,b,*,atol=8e-6,rtol=8e-4):
        self.assertEqual(jax.tree.structure(a),jax.tree.structure(b))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)

    def test_centered_moments_and_gradients_against_float64_reference(self):
        rng=np.random.default_rng(347);x=(10000+rng.normal(size=(12,3,3,7))).astype(np.float32)
        mask=np.ones((12,3,3,1),np.float32);mask[3:7]=0.;mask[10]=0.
        def stats(x):
            def one(carry,data):return cnn_chunks.merge_moments(carry,cnn_chunks.frame_moments(*data)),None
            start=(jnp.asarray(0.),jnp.zeros(7),jnp.zeros(7))
            result,_=jax.lax.scan(one,start,(x.reshape(6,2,3,3,7),jnp.asarray(mask).reshape(6,2,3,3,1)))
            return cnn_chunks.global_moments(result)
        result=stats(jnp.asarray(x));selected=x[np.broadcast_to(mask.astype(bool),x.shape)].reshape(-1,7).astype(np.float64)
        self.assertEqual(float(result['count']),len(selected))
        np.testing.assert_allclose(result['mean'],selected.mean(0),rtol=0,atol=.003)
        np.testing.assert_allclose(result['variance'],selected.var(0),rtol=0,atol=.001)
        grad=jax.jit(jax.grad(lambda x:jnp.sum(stats(x)['variance'])))(jnp.asarray(x))
        expected=2*(x.astype(np.float64)-selected.mean(0))*mask/len(selected)
        np.testing.assert_allclose(grad,expected,rtol=.003,atol=6e-5)

    def test_policy_value_and_global_statistics_match_with_partial_chunks(self):
        p,b,c=self.p,self.b,self.c
        expected=jax.jit(lambda p:joint.forward(p,b,c,training=True))(p)
        spatial=b['spatial'].reshape(-1,3,3,22);glob=b['global_features'].reshape(-1,19)
        x=katago.trunk_batched(p,spatial,glob,c);mask=spatial[...,:1]
        n=float(jnp.sum(mask));mean=jnp.sum(x*mask,axis=(0,1,2))/n
        var=jnp.sum(((x-mean)*mask)**2,axis=(0,1,2))/n
        for chunk in (1,3,32):
            with self.subTest(chunk_frames=chunk):
                actual,stats=jax.jit(lambda p:cnn_chunks.forward(p,b,c,chunk_frames=chunk,with_statistics=True))(p)
                self.assert_close(actual,expected)
                self.assertAlmostEqual(float(stats['count']),n)
                self.assert_close(stats['mean'],mean);self.assert_close(stats['variance'],var)

    def test_every_joint_gradient_matches_materialized_reference(self):
        p,b,c=self.p,self.b,self.c
        expected=jax.jit(jax.value_and_grad(lambda p:joint.losses(p,b,c,value_weight=.7)[0]))(p)
        for inner in (False,True):
            with self.subTest(inner_rematerialize=inner):
                actual=jax.jit(jax.value_and_grad(lambda p:cnn_chunks.losses(p,b,c,value_weight=.7,chunk_frames=3,
                                                                          inner_rematerialize=inner)[0]))(p)
                self.assert_close(actual,expected,atol=1e-5,rtol=1e-3)
                self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(actual)))

    def test_global_helper_and_gradient_match_with_two_empty_shards(self):
        self.assertEqual(jax.device_count(),4,'Must use four simulated CPU devices')
        p,c=self.p,self.c;counts=jnp.asarray([4,1,0,0]);live=jnp.arange(4)[None,:]<counts[:,None]
        b={**self.b,'counts':counts,'spatial':self.b['spatial']*live[...,None,None,None],
           'global_features':self.b['global_features']*live[...,None]}
        expected=jax.jit(jax.value_and_grad(lambda p:joint.losses(p,b,c,value_weight=.7)[0]))(p)
        mesh=Mesh(np.asarray(jax.devices()),('data',));rep=NamedSharding(mesh,P());data=NamedSharding(mesh,P('data'))
        operation=jax.shard_map(lambda p,b:cnn_chunks.losses(p,b,c,value_weight=.7,chunk_frames=3,axis_name='data')[0],
            mesh=mesh,in_specs=(P(),jax.tree.map(lambda _:P('data'),b)),out_specs=P(),check_vma=False)
        actual=jax.jit(jax.value_and_grad(operation))(jax.tree.map(lambda x:jax.device_put(x,rep),p),
            jax.tree.map(lambda x:jax.device_put(x,data),b))
        self.assert_close(actual,expected,atol=1e-5,rtol=1e-3)

    def test_empty_population_has_finite_zero_loss_and_parameter_gradient(self):
        b={**self.b,'counts':jnp.zeros(4,jnp.int32),'spatial':jnp.zeros_like(self.b['spatial']),
           'global_features':jnp.zeros_like(self.b['global_features']),'values':jnp.full((4,4),jnp.nan)}
        actual=jax.jit(jax.value_and_grad(lambda p:cnn_chunks.losses(p,b,self.c,value_weight=.7,chunk_frames=3)[0]))(self.p)
        for x in jax.tree.leaves(actual):np.testing.assert_array_equal(x,0.)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification)
    names=[t.id() for t in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(kind='bounded_cnn_helper_cpu_qualification',status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun,test_names=names,elapsed_seconds=time.monotonic()-start,
        failures=[(t.id(),message) for t,message in result.failures+result.errors],
        device_count=jax.device_count(),jax_version=jax.__version__,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        scope='Centered chunk moments, exact readout targets and all coupled backbone/head gradients within declared float32 tolerances, including global reductions and empty shards. Training computation increases; full-size TPU memory/timing remains unqualified.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status=report['status'],tests=report['tests_run'],seconds=report['elapsed_seconds'])),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
