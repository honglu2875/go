"""CPU optimizer qualification over exact V7 prefixes of real 19x19 games.

Short prefixes here are numerical fixtures, not a training data selection or
horizon. The optimizer checkpoint does not include a trainer's sampler state.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from gozero import checkpoints
from gozero.corpus_sequence_batches import Dataset
from gozero.katago_sequence_batches import augment
import adamw
import joint
import learner
import optimizer_io
from qualify_joint import configs


ROOT=Path(__file__).resolve().parents[3]
RECIPE=Path(__file__).resolve().parent
PLAN_SHA='626dee82dda80b1538e8e72f285ff4b9a8beb2e325eb37a8a4cdc4eb5c54b7aa'
DATA_SHA='2708fea4eca5f09be1a2f5231079943cf7c091e35fbb67183ddc339745f53869'
VALUE_CONFIG=dict(hidden=7,spatial_channels=5)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_files():
    files=list(RECIPE.glob('*.py'))+list((ROOT/'packages/gozero/src/gozero').glob('*.py'))
    files += [ROOT/'research/recipes/strong9_policy/policy_optimizer.py',ROOT/'uv.lock',ROOT/'pyproject.toml']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def model_config(name):
    return {**configs()[name],'max_board_size':19}


def parameters(name):
    p=joint.initialize(213,model_config(name),VALUE_CONFIG)
    if name=='transformer':
        p['head.local.weight']=jnp.linspace(-.1,.12,16)[:,None]
        p['head.context.q.weight']=jnp.linspace(-.2,.13,64).reshape(16,4)
    else:
        p['norm_intermediate_trunkfinal.gamma']=jnp.linspace(-.05,.05,16)
        p['norm_intermediate_trunkfinal.beta']=jnp.linspace(-.02,.03,16)
    return p


def fixture(directory):
    data=Dataset(directory,DATA_SHA,allow_partial=True)
    if data.size!=19:raise ValueError('Expected real 19x19 features')
    entries=[('expert',s,e) for s,e in data.indices['expert',0][:3]]
    if len(entries)!=3:raise ValueError('Missing fixture games')
    whole=data.batch(entries+[None],positions=data.time)
    b={k:(v[:,:4].copy() if v.ndim>=2 else v.copy()) for k,v in whole.items()}
    b['counts']=np.asarray([4,3,1,0],np.int32)
    live=np.arange(4)[None,:]<b['counts'][:,None]
    b['spatial']*=live[...,None,None,None]
    b['global_features']*=live[...,None]
    b['values'][~live]=np.nan
    description=dict(manifest_sha256=DATA_SHA,board_size=19,counts=b['counts'].tolist(),
        game_ids=[bytes(data.game_info(e)['game_id']).decode() for e in entries],
        complete_dataset_games=len(data.indices['expert',0]),complete_dataset_positions=data.manifest['populations']['train']['positions'],
        scope='First 4/3/1 positions of three real games and one empty lane, solely for numerical qualification')
    return b,description


def fixed_batches(b):
    rng=np.random.default_rng(1237)
    return [b,augment(b,rng.integers(0,8,4)),augment(b,rng.integers(0,8,4))]


def array_error(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.max(np.abs(a-b),initial=0.))


class Qualification(unittest.TestCase):
    plan=None
    dataset=None
    observations=[]

    @classmethod
    def setUpClass(cls):
        if jax.default_backend()!='cpu' or jax.device_count()!=4:
            raise ValueError('Use exactly four simulated CPU devices')
        raw,cls.data_record=fixture(cls.dataset)
        cls.batches=[jax.tree.map(jnp.asarray,b) for b in fixed_batches(raw)]
        cls.params={name:parameters(name) for name in configs()}
        cls.opt=cls.plan['fixture_optimizer'];cls.weight=cls.plan['fixture_value_coefficient']
        cls.gates=cls.plan['numerical_gates'];cls.compiled={};cls.trajectories={}
        cls.mesh=Mesh(np.asarray(jax.devices()),('data',))

    def close(self,a,b,*,kind):
        self.assertEqual(jax.tree.structure(a),jax.tree.structure(b))
        atol=self.gates[kind+'_atol'];rtol=self.gates[kind+'_rtol'];maximum=0.
        for (path,x),(_,y) in zip(jax.tree.flatten_with_path(a)[0],jax.tree.flatten_with_path(b)[0]):
            np.testing.assert_allclose(x,y,atol=atol,rtol=rtol,err_msg=str(path))
            maximum=max(maximum,array_error(x,y))
        return maximum

    def exact(self,a,b):
        self.assertEqual(jax.tree.structure(a),jax.tree.structure(b))
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def operation(self,name,path='bounded',distributed=False):
        key=name,path,distributed
        if key not in self.compiled:
            f=learner.step(model_config(name),self.opt,value_weight=self.weight,path=path,chunk_frames=3,
                           mesh=self.mesh if distributed else None)
            self.compiled[key]=jax.jit(f)
        return self.compiled[key]

    def trajectory(self,name,path):
        key=name,path
        if key not in self.trajectories:
            p=self.params[name];s=adamw.initialize(p);out=[]
            for b in self.batches:
                p,s,m=self.operation(name,path)(p,s,b)
                jax.block_until_ready((p,s,m))
                self.assertTrue(bool(m['accepted']));self.assertEqual(int(m['positions']),8)
                out.append((p,s,m))
            self.trajectories[key]=out
        return self.trajectories[key]

    def compare_update(self,actual,expected,before,label):
        p,s,m=actual;q,t,n=expected
        p_error=self.close(p,q,kind='parameter')
        m_error=self.close((s['first'],s['second']),(t['first'],t['second']),kind='gradient')
        self.exact(s['step'],t['step']);self.assertTrue(bool(m['accepted']))
        self.close(m['loss'],n['loss'],kind='loss')
        error=sum(float(np.sum((np.asarray(p[k],np.float64)-np.asarray(q[k],np.float64))**2)) for k in p)
        norm=sum(float(np.sum((np.asarray(q[k],np.float64)-np.asarray(before[k],np.float64))**2)) for k in p)
        ratio=float(np.sqrt(error/max(norm,1e-30)))
        self.assertLessEqual(ratio,self.gates['whole_update_relative_l2'],label)
        self.observations.append(dict(check=label,parameter_max_abs=p_error,moment_max_abs=m_error,
                                      whole_update_relative_l2=ratio,loss=float(m['loss']),step=int(s['step'])))

    def test_01_precomputed_gradient_matches_qualified_policy_optimizer(self):
        old_path=ROOT/'research/recipes/strong9_policy/policy_optimizer.py'
        self.assertEqual(sha(old_path),self.plan['parent_optimizer_sha256'])
        spec=importlib.util.spec_from_file_location('qualified_parent_adamw',old_path)
        parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
        rng=np.random.default_rng(711)
        for name,p in self.params.items():
            with self.subTest(architecture=name):
                q=p;s=adamw.initialize(p);t=parent.initialize(q)
                for i in range(3):
                    g={k:jnp.asarray(rng.normal(0,.05,v.shape),jnp.float32) for k,v in p.items()}
                    kwargs={k:self.opt[k] for k in ('beta1','beta2','epsilon','weight_decay','max_grad_norm')}
                    kwargs.update(learning_rate=learner.schedule(i+1,self.opt),architecture=model_config(name)['architecture'])
                    p,s,m=adamw.apply_gradient(p,s,g,jnp.asarray(3.),**kwargs)
                    q,t,n=parent.apply_gradient(q,t,g,jnp.asarray(3.),**kwargs)
                    self.exact((p,s),(q,t))
                    for key in ('grad_norm','raw_grad_norm','clip_scale','accepted'):self.exact(m[key],n[key])
        self.observations.append(dict(check='precomputed_gradient_parent',steps_per_model=3,bit_exact=True))

    def test_02_bounded_and_materialized_all_gradients(self):
        for name,p in self.params.items():
            with self.subTest(architecture=name):
                results=[]
                for path in ('bounded','materialized'):
                    f=lambda p:learner.loss(p,self.batches[0],model_config(name),value_weight=self.weight,
                                            path=path,chunk_frames=3)[0]
                    results.append(jax.jit(jax.value_and_grad(f))(p))
                loss_error=self.close(results[0][0],results[1][0],kind='loss')
                gradient_error=self.close(results[0][1],results[1][1],kind='gradient')
                self.observations.append(dict(check=name+'_all_gradients',loss_max_abs=loss_error,
                                              gradient_max_abs=gradient_error,parameter_leaves=len(p)))

    def test_03_complete_updates_match_materialized(self):
        for name in self.params:
            with self.subTest(architecture=name):
                actual=self.trajectory(name,'bounded');expected=self.trajectory(name,'materialized')
                previous=self.params[name]
                for i,(a,e) in enumerate(zip(actual,expected)):
                    self.compare_update(a,e,previous,f'{name}_bounded_materialized_{i+1}')
                    previous=e[0]

    def test_04_unequal_four_device_updates_match_unsplit(self):
        rep=NamedSharding(self.mesh,P());data=NamedSharding(self.mesh,P('data'))
        for name,p in self.params.items():
            with self.subTest(architecture=name):
                s=adamw.initialize(p);p,s=jax.tree.map(lambda x:jax.device_put(x,rep),(p,s))
                previous=self.params[name]
                for i,(b,expected) in enumerate(zip(self.batches,self.trajectory(name,'materialized'))):
                    db=jax.tree.map(lambda x:jax.device_put(x,data),b)
                    p,s,m=self.operation(name,distributed=True)(p,s,db)
                    self.compare_update((p,s,m),expected,previous,f'{name}_four_device_{i+1}')
                    previous=expected[0]

    def test_05_rejected_updates_preserve_parameters_and_entire_state(self):
        for name in self.params:
            with self.subTest(architecture=name):
                p,s,_=self.trajectory(name,'bounded')[0];b=self.batches[0]
                empty={**b,'spatial':jnp.zeros_like(b['spatial']),'global_features':jnp.zeros_like(b['global_features']),
                       'counts':jnp.zeros_like(b['counts']),'values':jnp.full_like(b['values'],jnp.nan)}
                bad={**b,'values':b['values'].at[0,0].set(jnp.nan)}
                for label,used_state,batch in [('empty',s,empty),('nonfinite_value',s,bad),
                    ('exhausted_horizon',{**s,'step':jnp.asarray(self.opt['horizon_steps'],jnp.int32)},b)]:
                    q,t,m=self.operation(name)(p,used_state,batch)
                    self.assertFalse(bool(m['accepted']),label);self.exact((p,used_state),(q,t))
                gradients=jax.tree.map(jnp.zeros_like,p)
                key=next(iter(gradients));gradients[key]=jnp.full_like(gradients[key],jnp.inf)
                q,t,m=learner.apply(p,s,gradients,jnp.asarray(1.),{'positions':jnp.asarray(8)},model_config(name),self.opt)
                self.assertFalse(bool(m['accepted']));self.exact((p,s),(q,t))
                # Empty global population exercises both collectives and the
                # optimizer gate, beyond merely having an empty local shard.
                rep=NamedSharding(self.mesh,P());data=NamedSharding(self.mesh,P('data'))
                q,t,m=self.operation(name,distributed=True)(
                    jax.tree.map(lambda x:jax.device_put(x,rep),p),
                    jax.tree.map(lambda x:jax.device_put(x,rep),s),
                    jax.tree.map(lambda x:jax.device_put(x,data),empty))
                self.assertFalse(bool(m['accepted']));self.exact((p,s),(q,t))
        self.observations.append(dict(check='rollback',cases_per_model=5,bit_exact=True))

    def test_06_checkpoint_and_fresh_process_continuation(self):
        for name in self.params:
            with self.subTest(architecture=name),tempfile.TemporaryDirectory(prefix='gozero-joint-optimizer-') as tmp:
                directory=Path(tmp);p,s,_=self.trajectory(name,'bounded')[0]
                np.savez(directory/'batches.npz',**{f'b{i}_{k}':np.asarray(v) for i,b in enumerate(self.batches[1:]) for k,v in b.items()})
                config=dict(name=name,model=model_config(name),value_config=VALUE_CONFIG,optimizer=self.opt,
                            value_weight=self.weight,path='bounded',chunk_frames=3,batch_sha256=sha(directory/'batches.npz'))
                config_sha=digest(config);source_sha=digest(source_files())
                metadata,arrays=optimizer_io.flatten(p,s,configuration_sha256=config_sha,source_sha256=source_sha)
                checkpoint_sha=checkpoints.write(directory/'checkpoint',state=metadata,arrays=arrays,actors='{}')
                saved,loaded,_=checkpoints.read(directory/'checkpoint',expected_manifest_sha256=checkpoint_sha)
                kwargs=dict(schema=metadata['parameter_schema'],configuration_sha256=config_sha,source_sha256=source_sha)
                self.exact((p,s),optimizer_io.restore(saved,loaded,**kwargs))
                instructions=dict(config=config,configuration_sha256=config_sha,source_sha256=source_sha,checkpoint_sha256=checkpoint_sha)
                (directory/'fixture.json').write_text(json.dumps(instructions,sort_keys=True))
                child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--resume-fixture',str(directory)],
                                     capture_output=True,text=True,timeout=600)
                self.assertEqual(child.returncode,0,child.stdout+'\n'+child.stderr)
                continuation=json.loads((directory/'continuation.json').read_text())
                saved,loaded,_=checkpoints.read(directory/'continued',expected_manifest_sha256=continuation['checkpoint_sha256'])
                continued=optimizer_io.restore(saved,loaded,**kwargs)
                expected=self.trajectory(name,'bounded')[-1]
                self.exact(continued,expected[:2])
                self.assertEqual(continuation['source_sha256'],source_sha)
                self.observations.append(dict(check=name+'_fresh_process',optimizer_step=int(continued[1]['step']),
                    bit_exact=True,checkpoint_sha256=checkpoint_sha,continued_checkpoint_sha256=continuation['checkpoint_sha256']))

    def test_07_checkpoint_rejects_wrong_source_configuration_schema_or_state(self):
        p,s,_=self.trajectory('cnn','bounded')[0]
        metadata,arrays=optimizer_io.flatten(p,s,configuration_sha256='a'*64,source_sha256='b'*64)
        args=dict(schema=metadata['parameter_schema'],configuration_sha256='a'*64,source_sha256='b'*64)
        for key in ('configuration_sha256','source_sha256'):
            with self.assertRaises(ValueError):optimizer_io.restore(metadata,arrays,**{**args,key:'c'*64})
        altered=copy.deepcopy(metadata);altered['parameter_schema'][0]['shape']=[1]
        with self.assertRaises(ValueError):optimizer_io.restore(altered,arrays,**args)
        missing={k:v for k,v in arrays.items() if k!='m_0000'}
        with self.assertRaises(ValueError):optimizer_io.restore(metadata,missing,**args)
        for key,value in [('step',np.asarray(0.,np.float32)),('v_0000',-np.ones_like(arrays['v_0000'])),
                          ('m_0000',np.ones((1,),np.float32)),('p_0000',np.full_like(arrays['p_0000'],np.nan))]:
            with self.assertRaises(ValueError):optimizer_io.restore(metadata,{**arrays,key:value},**args)
        self.observations.append(dict(check='checkpoint_rejections',cases=8))


def resume(directory):
    f=json.loads((directory/'fixture.json').read_text());c=f['config']
    if digest(c)!=f['configuration_sha256'] or digest(source_files())!=f['source_sha256']:
        raise ValueError('Fresh process source or configuration differs')
    if sha(directory/'batches.npz')!=c['batch_sha256']:raise ValueError('Fixture arrays differ')
    initial=parameters(c['name'])
    expected,_=optimizer_io.flatten(initial,adamw.initialize(initial),configuration_sha256=f['configuration_sha256'],source_sha256=f['source_sha256'])
    saved,arrays,_=checkpoints.read(directory/'checkpoint',expected_manifest_sha256=f['checkpoint_sha256'])
    p,s=optimizer_io.restore(saved,arrays,schema=expected['parameter_schema'],configuration_sha256=f['configuration_sha256'],source_sha256=f['source_sha256'])
    p,s=jax.tree.map(jnp.asarray,(p,s))
    operation=jax.jit(learner.step(c['model'],c['optimizer'],value_weight=c['value_weight'],path=c['path'],chunk_frames=c['chunk_frames']))
    with np.load(directory/'batches.npz',allow_pickle=False) as packed:
        for i in range(2):
            prefix=f'b{i}_';b={k[len(prefix):]:jnp.asarray(packed[k]) for k in packed.files if k.startswith(prefix)}
            p,s,m=operation(p,s,b)
            if not bool(m['accepted']):raise ValueError('Continuation rejected')
    saved,arrays=optimizer_io.flatten(p,s,configuration_sha256=f['configuration_sha256'],source_sha256=f['source_sha256'])
    identity=checkpoints.write(directory/'continued',state=saved,arrays=arrays,actors='{}')
    (directory/'continuation.json').write_text(json.dumps(dict(checkpoint_sha256=identity,source_sha256=digest(source_files()))))


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path);mode.add_argument('--resume-fixture',type=Path)
    parser.add_argument('--dataset',type=Path);parser.add_argument('--plan',type=Path)
    a=parser.parse_args()
    if a.resume_fixture:return resume(a.resume_fixture)
    if a.output.exists():raise FileExistsError(a.output)
    if a.plan is None or sha(a.plan)!=PLAN_SHA:raise ValueError('Unregistered qualification plan')
    if a.dataset is None:raise ValueError('Real feature fixture required')
    Qualification.plan=json.loads(a.plan.read_text());Qualification.dataset=a.dataset
    before=source_files();start=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification);names=[test.id() for test in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    stable=source_files()==before
    report=dict(kind='joint_optimizer_cpu_qualification',status='passed' if result.wasSuccessful() and stable else 'failed',
        created=time.time(),plan_sha256=PLAN_SHA,source_sha256=before,source_stable=stable,
        tests_run=result.testsRun,test_names=names,elapsed_seconds=time.monotonic()-start,
        failures=[(t.id(),message) for t,message in result.failures+result.errors],
        observations=Qualification.observations,fixture=getattr(Qualification,'data_record',None),
        models={k:model_config(k) for k in configs()},value_config=VALUE_CONFIG,
        jax_version=jax.__version__,device_count=jax.device_count(),
        scope='Small complete joint backbones on real 19x19 prefixes; optimizer fixture only, not full TPU memory or a registered learning experiment. Checkpoints cover parameters/moments/step, not trainer sampler state.')
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps(dict(status=report['status'],tests=report['tests_run'],seconds=report['elapsed_seconds'])),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
