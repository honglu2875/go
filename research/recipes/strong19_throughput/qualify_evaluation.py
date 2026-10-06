"""Compare bounded main-only evaluation with the qualified complete models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.corpus_sequence_batches import Dataset
import evaluation
import heads
import joint
import policy_model


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}


class Qualification(unittest.TestCase):
    observations=[]

    @classmethod
    def setUpClass(cls):
        if jax.default_backend()!='cpu' or jax.device_count()!=4:
            raise ValueError('Use four simulated CPU devices')
        cls.models=cls.parent['models'];cls.value_config=cls.parent['value_config']
        cls.params={name:joint.initialize(213,c,cls.value_config) for name,c in cls.models.items()}
        p=cls.params['transformer'];p['head.local.weight']=jnp.linspace(-.1,.12,16)[:,None]
        p['head.context.q.weight']=jnp.linspace(-.2,.13,64).reshape(16,4)
        data=Dataset(cls.dataset,cls.parent['fixture']['manifest_sha256'],allow_partial=True)
        entries=[('expert',s,e) for s,e in data.indices['expert',0][:3]]
        full=data.batch(entries+[None],positions=data.time)
        b={k:(v[:,:4].copy() if v.ndim>=2 else v.copy()) for k,v in full.items()}
        b['counts']=np.asarray([4,3,1,0],np.int32)
        live=np.arange(4)[None,:]<b['counts'][:,None]
        b['spatial']*=live[...,None,None,None];b['global_features']*=live[...,None]
        b['values'][~live]=np.nan;b['family_weights']=np.asarray([.1,.05,.4,0.],np.float32)
        cls.b=jax.tree.map(jnp.asarray,b);cls.compiled={}
        cls.reference={name:jax.jit(lambda p,b:joint.forward(p,b,c))(cls.params[name],cls.b) for name,c in cls.models.items()}

    def function(self,name,chunk=3):
        if (name,chunk) not in self.compiled:
            c=self.models[name]
            self.compiled[name,chunk]=jax.jit(lambda p,b:evaluation.forward(p,b,c,chunk_frames=chunk))
        return self.compiled[name,chunk]

    def close(self,a,b):
        self.assertEqual(jax.tree.structure(a),jax.tree.structure(b));maximum=0.
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
            np.testing.assert_allclose(x,y,atol=self.plan['output_atol'],rtol=self.plan['output_rtol'])
            maximum=max(maximum,float(np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0.)))
        return maximum

    def test_01_complete_outputs_match_across_chunk_boundaries(self):
        for name,p in self.params.items():
            for chunk in (1,3,32):
                with self.subTest(model=name,chunk=chunk):
                    actual=self.function(name,chunk)(p,self.b)
                    error=self.close(actual,self.reference[name])
                    self.assertEqual(set(actual),{'policy','value','value_logits'})
                    self.observations.append(dict(check='main_outputs',model=name,chunk_frames=chunk,max_abs=error))

    def test_02_future_and_other_games_do_not_change_current_outputs(self):
        b=self.b
        changed={**b,'spatial':b['spatial'].at[0,2:].set(7).at[1:].set(9),
            'global_features':b['global_features'].at[0,2:].set(4).at[1:].set(5),
            'actions':b['actions'].at[0,1:].set(361).at[1:].set(0)}
        for name,p in self.params.items():
            before=self.function(name)(p,b);after=self.function(name)(p,changed)
            for key in before:np.testing.assert_array_equal(before[key][0,:2],after[key][0,:2])
            for key in ('value','value_logits'):np.testing.assert_array_equal(before[key][3],0.)

    def test_03_policy_value_means_match_independent_numpy(self):
        b=jax.tree.map(np.asarray,self.b);live=np.arange(4)[None,:]<b['counts'][:,None]
        weights={'':live,'_family':live*b['family_weights'][:,None]}
        for name,p in self.params.items():
            out=jax.tree.map(np.asarray,self.function(name)(p,self.b))
            totals=evaluation.totals(p,self.b,self.models[name],chunk_frames=3)
            means=evaluation.averages(totals)
            logp=np.where(b['legal'],out['policy'].astype(np.float64),-1e9)
            logp=logp-logp.max(-1,keepdims=True);logp=logp-np.log(np.exp(logp).sum(-1,keepdims=True))
            ce=-(b['policies']*logp).sum(-1)
            for suffix,w in weights.items():
                denom=w.sum();policy_key='family_ce' if suffix else 'expert_ce'
                self.assertAlmostEqual(float(means[policy_key]),float((ce*w).sum()/denom),places=5)
                error=np.where(live,out['value'],0.).astype(np.float64)-np.where(live,b['values'],0.)
                self.assertAlmostEqual(float(means['value'+suffix+'_mse']),float((error**2*w).sum()/denom),places=6)
                self.assertAlmostEqual(float(means['value'+suffix+'_count']),float(denom),places=6)
            self.assertLess(float(means['family_count']),1.)
            self.assertEqual(float(means['expert_count']),8.)
            self.assertEqual(float(means['value_count']),8.)
            # Phase/opponent subpopulations preserve total mass.
            self.assertEqual(sum(float(means[f'value_opponent_{i}_count']) for i in range(8)),8.)
            self.assertEqual(sum(float(means[f'value_phase_{a}_{z}_count']) for a,z in heads.PHASES),8.)

    def test_04_four_device_global_totals_match_unsplit(self):
        mesh=Mesh(np.asarray(jax.devices()),('data',));rep=NamedSharding(mesh,P());data=NamedSharding(mesh,P('data'))
        for name,p in self.params.items():
            c=self.models[name]
            reference=jax.jit(lambda p,b:evaluation.totals(p,b,c,chunk_frames=3))(p,self.b)
            f=jax.shard_map(lambda p,b:evaluation.totals(p,b,c,chunk_frames=3,axis_name='data'),
                mesh=mesh,in_specs=(P(),P('data')),out_specs=P(),check_vma=False)
            actual=jax.jit(f)(jax.tree.map(lambda x:jax.device_put(x,rep),p),jax.tree.map(lambda x:jax.device_put(x,data),self.b))
            error=self.close(actual,reference)
            self.observations.append(dict(check='global_totals',model=name,max_abs=error))

    def test_05_empty_global_population_has_finite_zero_totals(self):
        b={**self.b,'counts':jnp.zeros(4,jnp.int32),'spatial':jnp.zeros_like(self.b['spatial']),
            'global_features':jnp.zeros_like(self.b['global_features']),'values':jnp.full_like(self.b['values'],jnp.nan)}
        for name,p in self.params.items():
            c=self.models[name];totals=jax.jit(lambda p,b:evaluation.totals(p,b,c,chunk_frames=3))(p,b)
            means=evaluation.averages(totals)
            for value in jax.tree.leaves((totals,means)):np.testing.assert_array_equal(value,0.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Evaluation plan identity differs')
    plan=json.loads(a.plan.read_text());parent_path=ROOT/plan['parent_qualification']['path']
    if sha(parent_path)!=plan['parent_qualification']['sha256']:raise ValueError('Parent qualification changed')
    parent=json.loads(parent_path.read_text())
    if parent['status']!='passed':raise ValueError('Parent is not qualified')
    provenance_path=ROOT/plan['provenance']['path']
    if sha(provenance_path)!=plan['provenance']['sha256']:raise ValueError('Copied source provenance changed')
    provenance=json.loads(provenance_path.read_text())
    for name,record in provenance['files'].items():
        if sha(Path(__file__).parent/name)!=record['sha256']:raise ValueError('Qualified implementation changed: '+name)
    Qualification.plan=plan;Qualification.parent=parent;Qualification.dataset=a.dataset
    source=sources();start=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification);names=[test.id() for test in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite);stable=sources()==source
    report=dict(kind='joint_bounded_evaluation_cpu_qualification',status='passed' if result.wasSuccessful() and stable else 'failed',
        created=time.time(),seconds=time.monotonic()-start,tests_run=result.testsRun,test_names=names,
        failures=[(t.id(),m) for t,m in result.failures+result.errors],source_sha256=source,source_stable=stable,
        plan_sha256=a.plan_sha256,observations=Qualification.observations,jax_version=jax.__version__,device_count=jax.device_count(),
        scope='Small complete models on exact real 19x19 prefixes; float32 reference outputs, no future/game leakage, independent policy/value means and four-device global totals. Not full-size TPU BF16 equivalence or measured memory.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    a.output.chmod(0o444);print(json.dumps(dict(status=report['status'],seconds=report['seconds'],sha256=sha(a.output))),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
