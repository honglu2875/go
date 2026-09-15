"""Joint-backbone gradients, causal value readout and incremental cache gates."""
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

import causal
import compact
import joint
import policy_model


def configs():
    cnn=dict(architecture='katago_nested_policy',width=16,mid_width=8,gpool_width=2,policy_width=4,
             layers=3,dtype='float32',rematerialize=True,microbatch=3,max_board_size=3,max_positions=8,norm_epsilon=1e-4)
    transformer=dict(architecture='causal_visual_policy',width=16,layers=2,mlp_hidden=24,
        heads=2,kv_heads=1,max_board_size=3,max_positions=8,dtype='float32',rematerialize=True,
        norm_epsilon=1e-6,rope_theta=10000.,attention_backend='xla',encoder_width=16,
        encoder_blocks=2,encoder_passes=2,encoder_expansion=2,connector_channels=4,
        encoder_rematerialize=True,encoder_layer_scale=.1,policy_spatial_bias=True,policy_context_dim=4,
        encoder_attention_blocks=1,encoder_attention_heads=2,encoder_attention_mlp_hidden=24,
        encoder_rope_theta=10000.,first_pass_aux_weight=.25)
    return dict(cnn=cnn,transformer=transformer)


def fixture():
    rng=np.random.default_rng(8231);counts=np.asarray([4,3,1,0],np.int32)
    live=np.arange(4)[None,:]<counts[:,None]
    s=rng.integers(0,2,size=(4,4,3,3,22)).astype(np.float32)
    s[...,:1]=1.;s*=live[...,None,None,None]
    values=rng.uniform(-.9,.9,(4,4)).astype(np.float32);values[~live]=np.nan
    return jax.tree.map(jnp.asarray,dict(spatial=s,
        global_features=rng.normal(size=(4,4,19)).astype(np.float32)*.2*live[...,None],
        actions=rng.integers(0,10,size=(4,4),dtype=np.int32),counts=counts,
        policies=rng.dirichlet(np.ones(10),size=(4,4)).astype(np.float32),
        legal=np.ones((4,4,10),bool),values=values))


class Qualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs=configs();cls.b=fixture();cls.v=dict(hidden=7,spatial_channels=5)
        cls.params={k:joint.initialize(213,c,cls.v) for k,c in cls.configs.items()}
        p=cls.params['transformer']
        p['head.local.weight']=jnp.linspace(-.1,.12,16)[:,None]
        p['head.context.q.weight']=jnp.linspace(-.2,.13,64).reshape(16,4)

    def assert_close(self,a,b,*,atol=8e-6,rtol=6e-4):
        self.assertEqual(jax.tree.structure(a),jax.tree.structure(b))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)

    def test_backbone_draws_are_unchanged(self):
        for name,c in self.configs.items():
            with self.subTest(architecture=name):
                old=policy_model.initialize(13,c);new=joint.initialize(13,c,self.v)
                for key in old:np.testing.assert_array_equal(new[key],old[key])
                self.assertTrue(all(key.startswith(('value_head.','intermediate_value_head.')) for key in set(new)-set(old)))

    def test_policy_outputs_and_all_gradients_match_at_zero_value_weight(self):
        for name,c in self.configs.items():
            with self.subTest(architecture=name):
                p=self.params[name];b=self.b
                old=jax.jit(jax.value_and_grad(lambda p:policy_model.losses(p,b,c)[0]))(p)
                new=jax.jit(jax.value_and_grad(lambda p:joint.losses(p,b,c,value_weight=0.,chunk_frames=3)[0]))(p)
                self.assert_close(new,old)
                expected=policy_model.logits(p,b,c)
                result=joint.forward(p,b,c)
                self.assert_close(result['policy'],expected)
                for key,value in new[1].items():
                    if key.startswith(('value_head.','intermediate_value_head.')):np.testing.assert_array_equal(value,0.)

    def test_value_objective_reaches_both_backbones_and_auxiliary_heads(self):
        for name,c in self.configs.items():
            with self.subTest(architecture=name):
                p=self.params[name];b=self.b
                objective=lambda p:joint.losses(p,b,c,value_weight=.7,chunk_frames=3)
                (loss,metrics),gradient=jax.jit(jax.value_and_grad(objective,has_aux=True))(p)
                self.assertAlmostEqual(float(loss),float(metrics['policy_loss']+.7*metrics['value_loss']),places=5)
                self.assertEqual(float(metrics['positions']),8.)
                self.assertTrue(all(np.isfinite(g).all() for g in gradient.values()))
                value_grad=jax.jit(jax.grad(lambda p:objective(p)[1]['value_loss']))(p)
                keys=(['conv_spatial.weight','cycles.g.normactconvp.conv.weight','intermediate_value_head.linear2.weight']
                      if name=='cnn' else ['encoder.blocks.up.weight','encoder.attention.q.weight','blocks.q.weight','head.norm.scale'])
                for key in keys+['value_head.linear2.weight','value_head.linear_valuehead.weight']:
                    self.assertGreater(float(jnp.linalg.norm(value_grad[key])),1e-9,key)
                if name=='transformer':
                    for key in ('head.context.k.weight','head.context.q.weight','head.local.weight'):
                        np.testing.assert_array_equal(value_grad[key],0.)

    def test_joint_gradient_matches_unsplit_with_unequal_rank_counts(self):
        self.assertEqual(jax.device_count(),4,'Must use four simulated CPU devices')
        mesh=Mesh(np.asarray(jax.devices()),('data',));rep=NamedSharding(mesh,P());data=NamedSharding(mesh,P('data'))
        for name,c in self.configs.items():
            with self.subTest(architecture=name):
                p=self.params[name];b=self.b
                expected=jax.jit(jax.value_and_grad(lambda p:joint.losses(p,b,c,value_weight=.7,chunk_frames=3)[0]))(p)
                f=jax.shard_map(lambda p,b:joint.losses(p,b,c,value_weight=.7,chunk_frames=3,axis_name='data')[0],
                    mesh=mesh,in_specs=(P(),jax.tree.map(lambda _:P('data'),b)),out_specs=P(),check_vma=False)
                actual=jax.jit(jax.value_and_grad(f))(jax.tree.map(lambda x:jax.device_put(x,rep),p),
                    jax.tree.map(lambda x:jax.device_put(x,data),b))
                self.assert_close(actual,expected,atol=1e-5,rtol=1e-3)

    def test_policy_and_value_current_outputs_have_no_future_or_other_game_leak(self):
        p=self.params['transformer'];c=self.configs['transformer'];b=self.b
        f=jax.jit(lambda b:joint.forward(p,b,c,training=True,chunk_frames=3))
        old=f(b)
        changed={**b,'spatial':b['spatial'].at[0,2:].set(9).at[1:].set(-6),
                 'global_features':b['global_features'].at[0,2:].set(7).at[1:].set(3),
                 'actions':b['actions'].at[0,1:].set(9).at[1:].set(0)}
        new=f(changed)
        for name in old:np.testing.assert_array_equal(old[name][0,:2],new[name][0,:2])
        for name in ('value','aux_value','value_logits','aux_value_logits'):
            np.testing.assert_array_equal(old[name][3],0.)

    def test_incremental_main_and_first_pass_values_match_full_history(self):
        p=self.params['transformer'];c=self.configs['transformer']
        b=jax.tree.map(lambda x:x[:1],self.b);b['counts']=jnp.asarray([4])
        whole=joint.forward(p,b,c,training=True,chunk_frames=3)
        out,cache=joint.first_move(p,b['spatial'][:,0],b['global_features'][:,0],c,network_version=3)
        cheap={**c,'encoder_passes':1,'first_pass_aux_weight':0}
        draft,_=joint.first_move(p,b['spatial'][:,0],b['global_features'][:,0],cheap,network_version=3)
        for key in ('policy','value','value_logits'):
            self.assert_close(out[key],whole[key][:,0]);self.assert_close(draft[key],whole['aux_'+key][:,0])
        for t in range(1,4):
            # Draft branches consume completed full-pass history; their cache
            # is discarded here and cannot replace the verified main history.
            draft,_=joint.append_move(p,cache,b['actions'][:,t-1],b['spatial'][:,t],b['global_features'][:,t],cheap,
                                     attention_positions=t+1,network_version=3)
            out,cache=joint.append_move(p,cache,b['actions'][:,t-1],b['spatial'][:,t],b['global_features'][:,t],c,
                                       attention_positions=t+1,network_version=3)
            for key in ('policy','value','value_logits'):
                self.assert_close(out[key],whole[key][:,t]);self.assert_close(draft[key],whole['aux_'+key][:,t])
        prefix={k:(v[:,:3] if v.ndim>=2 else jnp.asarray([3])) for k,v in b.items()}
        before,cache=joint.prefill(p,prefix,c,network_version=3)
        out,cache=joint.append_move(p,cache,b['actions'][:,2],b['spatial'][:,3],b['global_features'][:,3],c,
                                   attention_positions=4,network_version=3)
        for key in ('policy','value','value_logits'):
            self.assert_close(before[key],whole[key][:,:3]);self.assert_close(out[key],whole[key][:,3])

    def test_rejected_cache_appends_have_zero_values_and_no_writes(self):
        p=self.params['transformer'];c=self.configs['transformer'];b=self.b
        _,cache=joint.first_move(p,b['spatial'][:,0],b['global_features'][:,0],c,network_version=3)
        for active,version in ((jnp.zeros(4,bool),3),(jnp.ones(4,bool),4)):
            out,new=joint.append_move(p,cache,b['actions'][:,0],b['spatial'][:,1],b['global_features'][:,1],c,
                                     attention_positions=2,active=active,network_version=version)
            for key in out:np.testing.assert_array_equal(out[key],0.)
            for key in ('keys','values','lengths'):np.testing.assert_array_equal(cache[key],new[key])
            np.testing.assert_array_equal(new['valid'],version==3)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification)
    names=[t.id() for t in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(kind='joint_backbone_cpu_qualification',status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun,test_names=names,elapsed_seconds=time.monotonic()-start,
        failures=[(t.id(),message) for t,message in result.failures+result.errors],
        device_count=jax.device_count(),jax_version=jax.__version__,configs=configs(),value_config=dict(hidden=7,spatial_channels=5),
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        scope='Small complete backbones, all gradients, four-device global normalization, causal main/first value and cache semantics; not full-size TPU or a training experiment')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status=report['status'],tests=report['tests_run'],seconds=report['elapsed_seconds'])),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
