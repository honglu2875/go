"""Check saturation recovery, masking, collectives and full-backbone gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
import heads
import joint
import policy_model


class Qualification(unittest.TestCase):
    def fixture(self):
        z=jnp.asarray([[[0,0,20],[1,2,3],[0,0,0]],[[1,0,-2],[2,1,0],[0,0,0]],
                       [[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],jnp.float32)
        b=dict(values=jnp.asarray([[-1,.4,np.nan],[1,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]]),
               counts=jnp.asarray([2,1,0,0],jnp.int32))
        return z,b

    def test_01_saturation_gradient_and_signed_target(self):
        z,b=self.fixture()
        g=jax.grad(lambda z:heads.signed_target_cross_entropy(z,b)['ce'])(z)
        self.assertGreater(float(jnp.linalg.norm(g[0,0])),.4)
        for target in [-.9,-.2,0,.4,.95]:
            prob=jnp.asarray([[(1+target)/2,(1-target)/2,1e-12]])
            logits=jnp.log(prob)[None,:]
            tb=dict(values=jnp.asarray([[target]]),counts=jnp.asarray([1]))
            self.assertLess(abs(float(heads.signed_value(logits)[0,0])-target),1e-6)
            gradient=jax.grad(lambda z:heads.signed_target_cross_entropy(z,tb)['ce'])(logits)
            self.assertLess(float(jnp.linalg.norm(gradient)),1e-6)

    def test_02_padding_nan_and_empty_population(self):
        z,b=self.fixture();live=jnp.arange(3)[None,:]<b['counts'][:,None]
        z=jnp.where(live[...,None],z,jnp.nan)
        actual=heads.signed_target_cross_entropy(z,b)
        self.assertTrue(all(np.isfinite(x) for x in actual.values()))
        gradient=jax.grad(lambda z:heads.signed_target_cross_entropy(z,b)['ce'])(z)
        np.testing.assert_array_equal(np.asarray(gradient)[~np.asarray(live)],0.)
        empty={**b,'counts':jnp.zeros(4,jnp.int32),'values':jnp.full((4,3),jnp.nan)}
        for value in heads.signed_target_cross_entropy(jnp.full((4,3,3),jnp.nan),empty).values():
            self.assertEqual(float(value),0.)

    def test_03_four_device_mean_uses_live_positions(self):
        self.assertEqual(jax.device_count(),4);z,b=self.fixture()
        mesh=Mesh(np.asarray(jax.devices()),('data',));sharding=NamedSharding(mesh,P('data'))
        f=jax.shard_map(lambda z,b:heads.signed_target_cross_entropy(z,b,axis_name='data'),mesh=mesh,
                        in_specs=(P('data'),P('data')),out_specs=P(),check_vma=False)
        actual=jax.jit(f)(jax.device_put(z,sharding),jax.device_put(b,sharding))
        expected=heads.signed_target_cross_entropy(z,b)
        for key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=1e-6,rtol=1e-6)

    def test_04_full_model_uses_ce_and_backpropagates(self):
        config=json.loads((Path(__file__).parent/'cpu_transformer_qualification.json').read_text())
        c=config['model'];p=joint.initialize(912,c,config['value_model']);n,t,size=1,2,9
        b=dict(spatial=jnp.ones((n,t,size,size,22),jnp.float32),global_features=jnp.zeros((n,t,19)),
            actions=jnp.asarray([[1,2]],jnp.int32),counts=jnp.asarray([2]),policies=jnp.full((n,t,82),1/82.),
            legal=jnp.ones((n,t,82),bool),values=jnp.asarray([[-.9,.9]]))
        objective=lambda p:joint.losses(p,b,c,value_weight=.7,chunk_frames=2,inner_rematerialize=False)
        (loss,metrics),g=jax.jit(jax.value_and_grad(objective,has_aux=True))(p)
        np.testing.assert_allclose(metrics['value_loss'],.75*metrics['main_value_ce']+.25*metrics['aux_value_ce'],rtol=1e-6)
        np.testing.assert_allclose(loss,metrics['policy_loss']+.7*metrics['value_loss'],rtol=1e-6)
        for prefix in ('value_head.','encoder.','layers.'):
            matched=[v for k,v in g.items() if k.startswith(prefix)]
            if prefix=='layers.':matched=[v for k,v in g.items() if k.startswith('blocks.')]
            self.assertTrue(matched,prefix)
            self.assertTrue(all(np.isfinite(x).all() for x in matched))
            self.assertGreater(sum(float(jnp.sum(x*x)) for x in matched),0.)


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    if jax.default_backend()!='cpu':raise ValueError('CPU qualification only')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Qualification))
    record=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    with args.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
