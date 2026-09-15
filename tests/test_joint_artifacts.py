import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from gozero import checkpoints, joint_artifacts as artifacts
from gozero.snapshots import canonical_json, freeze


class JointArtifactsTests(unittest.TestCase):
    def fixture(self, root, *, bad_array=None):
        recipe = root / 'research/recipes/fixture'; recipe.mkdir(parents=True)
        for name in (*artifacts.MODEL_MODULES, 'train.py'):
            (recipe / name).write_text('"""Artifact protocol fixture; no model execution."""\n')
        (recipe / 'recipe.json').write_text('{"id":"fixture"}\n')
        (root / 'uv.lock').write_text('version = 1\n')
        dataset = root / 'data'; dataset.mkdir()
        (dataset / 'manifest.json').write_bytes(canonical_json(dict(
            kind='katago_raw_teacher_sequences', feature_version=7,
            feature_worker_binary_sha256='a'*64, size=19, komi=7.5,
            spatial_channels=22, global_channels=19, qualification_only=True)))
        data_sha = checkpoints.sha256(dataset / 'manifest.json')
        config = dict(kind='fixed_joint_learning', steps=4, expected_processes=1,
                      training={'purpose':'qualification'}, model={'max_board_size':19,'max_positions':64}, value_model={},
                      dataset=dict(path=str(dataset), manifest_sha256=data_sha))
        config_path = recipe / 'config.json'; config_path.write_bytes(canonical_json(config))
        source = freeze(root, recipe, config_path, root / '.gozero/snapshots')
        config_sha = checkpoints.sha256(source / 'resolved_config.json')
        schema = [dict(path=name, shape=shape, dtype='float32', elements=int(np.prod(shape)), inference=inference)
                  for name, shape, inference in [('intermediate_policy_head.weight',[2],False),
                                                 ('policy_head.weight',[3,2],True),
                                                 ('value_head.weight',[2,3],True)]]
        arrays = {f'{prefix}_{i:04d}': np.full(row['shape'], i+.125, np.float32)
                  for i, row in enumerate(schema) for prefix in ('p','m','v')}
        arrays['step'] = np.asarray(4, np.int32)
        if bad_array is not None:
            arrays['p_0001'] = bad_array
        owner = root / 'learning/checkpoints/turn-000000004'
        mapping = [dict(host=0, jax_rank=0)]; counters = dict(updates=4, expert_positions=42)
        state = dict(kind='visual_replicated_rank_state', snapshot_id=source.name,
                     config_sha256=config_sha, turn=4, host_rank=0, jax_rank=0,
                     owns_replicated_arrays=True, model_schema=schema, counters=counters,
                     dataset_manifest_sha256=data_sha)
        manifest_sha = checkpoints.write(owner, state=state, arrays=arrays, actors='{}')
        group = dict(kind='visual_replicated_checkpoint_group', snapshot_id=source.name,
                     config_sha256=config_sha, turn=4, owner_checkpoint_path=str(owner),
                     host_manifests={'0':manifest_sha}, host_jax_mapping=mapping,
                     replicated_arrays_elements_sha256='b'*64)
        group_path = owner.with_suffix('.group.json'); group_path.write_bytes(canonical_json(group))
        result_path = root / 'learning/result.json'
        result = dict(kind='fixed_joint_learning', status='passed', training_complete=True,
                      snapshot_id=source.name, host_rank=0, config_sha256=config_sha, turn=4,
                      counters=counters, model_schema=schema, training_purpose='qualification',
                      world_size=1, host_jax_mapping=mapping, dataset_manifest_sha256=data_sha,
                      latest_checkpoint=dict(owner_checkpoint_path=str(owner),
                          group_sha256=checkpoints.sha256(group_path), manifest_sha256=manifest_sha,
                          replicated_arrays_elements_sha256='b'*64))
        result_path.write_bytes(canonical_json(result))
        return dict(root=root, source=source, result_path=result_path,
                    result_sha256=checkpoints.sha256(result_path), output=root/'export', schema=schema)

    def test_weights_preserve_bytes_and_remain_portable_after_training_files_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary)); report = artifacts.export(**args)
            state, params = artifacts.load(args['output'], report['manifest_sha256'])
            self.assertEqual(report['arrays'],2); self.assertEqual(report['dropped_helper_arrays'],1)
            self.assertEqual(report['training_parameters'],14); self.assertEqual(report['inference_parameters'],12)
            self.assertEqual(set(params),{'policy_head.weight','value_head.weight'})
            np.testing.assert_array_equal(params['policy_head.weight'], np.full((3,2),1.125,np.float32))
            np.testing.assert_array_equal(params['value_head.weight'], np.full((2,3),2.125,np.float32))
            self.assertNotIn(str(args['root']), (args['output']/'state.json').read_text())
            artifacts.compatible(state,args['source']/'research/recipes/fixture',schema=args['schema'])
            shutil.rmtree(args['root']/'learning');shutil.rmtree(args['root']/'data')
            _, again = artifacts.load(args['output'],report['manifest_sha256'])
            for name in params:self.assertEqual(params[name].tobytes(),again[name].tobytes())

    def test_incomplete_and_inconsistent_training_results_are_rejected(self):
        for key,value in [('training_complete',False),('status','failed'),('turn',3),('config_sha256','c'*64)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temporary:
                args=self.fixture(Path(temporary));report=json.loads(args['result_path'].read_text());report[key]=value
                args['result_path'].write_bytes(canonical_json(report));args['result_sha256']=checkpoints.sha256(args['result_path'])
                with self.assertRaises(ValueError):artifacts.export(**args)
                self.assertFalse(args['output'].exists())

    def test_parameter_shape_dtype_and_finiteness_are_checked_before_publication(self):
        for value in [np.ones((2,3),np.float32),np.ones((3,2),np.float64),np.full((3,2),np.nan,np.float32)]:
            with self.subTest(shape=value.shape,dtype=value.dtype),tempfile.TemporaryDirectory() as temporary:
                args=self.fixture(Path(temporary),bad_array=value)
                with self.assertRaises(ValueError):artifacts.export(**args)
                self.assertFalse(args['output'].exists())

    def test_full_checkpoint_integrity_is_checked_even_for_omitted_optimizer_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary));p=args['root']/'learning/checkpoints/turn-000000004/arrays.npz'
            p.chmod(0o644)
            with p.open('ab') as stream:stream.write(b'changed optimizer container')
            with self.assertRaises(ValueError):artifacts.export(**args)

    def test_missing_policy_and_extra_optimizer_arrays_are_rejected_on_load(self):
        for mutation in ('missing','optimizer'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                args=self.fixture(Path(temporary));report=artifacts.export(**args)
                state,arrays,actors=checkpoints.read(args['output'],expected_manifest_sha256=report['manifest_sha256'])
                if mutation=='missing':del arrays['p_0001']
                else:arrays['m_0001']=arrays['p_0001'].copy()
                wrong=args['root']/'wrong';digest=checkpoints.write(wrong,state=state,arrays=arrays,actors=actors)
                with self.assertRaises(ValueError):artifacts.load(wrong,digest)

    def test_foreign_numerical_code_and_changed_shape_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary));report=artifacts.export(**args)
            state,_=artifacts.load(args['output'],report['manifest_sha256'])
            with self.assertRaises(ValueError):artifacts.compatible(state,args['source']/'research/recipes/fixture',schema=args['schema'][:-1])
            altered=args['root']/'altered';shutil.copytree(args['source']/'research/recipes/fixture',altered)
            p=altered/'heads.py';p.chmod(0o644);p.write_text('changed value computation\n')
            with self.assertRaises(ValueError):artifacts.compatible(state,altered,schema=args['schema'])

    def test_source_and_result_identities_are_not_optional(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary));args['result_sha256']='e'*64
            with self.assertRaises(ValueError):artifacts.export(**args)
            args['result_sha256']=checkpoints.sha256(args['result_path'])
            p=args['source']/'research/recipes/fixture/joint.py';p.chmod(0o644);p.write_text('changed source\n')
            with self.assertRaises(ValueError):artifacts.export(**args)

    def test_distinct_feature_providers_require_matching_qualified_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary));report=artifacts.export(**args)
            state,_=artifacts.load(args['output'],report['manifest_sha256'])
            artifacts.compatible_features(state,'a'*64)
            with self.assertRaises(ValueError):artifacts.compatible_features(state,'f'*64)
            evidence=dict(kind='exact_v7_provider_comparison',status='passed',training_provider_sha256='a'*64,
                inference_provider_sha256='f'*64,input_contract={k:v for k,v in state['input_contract'].items() if k!='feature_worker_binary_sha256'},
                shared_objects_identical=True,all_spatial_bytes_equal=True,all_global_bytes_equal=True,
                all_legal_masks_equal=True,games=19,positions=7992,
                **{key:'b'*64 for key in ('operator_sha256','dataset_manifest_sha256','offline_build_receipt_sha256','inference_build_receipt_sha256','shared_objects_sha256')})
            artifacts.compatible_features(state,'f'*64,evidence=evidence)
            for key,value in [('status','failed'),('training_provider_sha256','c'*64),
                              ('inference_provider_sha256','c'*64),('all_global_bytes_equal',False),
                              ('shared_objects_identical',False),('positions',0)]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    artifacts.compatible_features(state,'f'*64,evidence={**evidence,key:value})

    def test_candidate_cannot_reassign_source_version_context_or_input_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary));report=artifacts.export(**args)
            state,_=artifacts.load(args['output'],report['manifest_sha256'])
            descriptor=dict(schema_version=1,kind='joint_policy_value_candidate',
                parameters=dict(path='export',manifest_sha256=report['manifest_sha256']),
                **{key:state[key] for key in ('training_snapshot','network_version','training_purpose','input_contract')},max_positions=64)
            restored,params=artifacts.load_candidate(args['root'],descriptor)
            self.assertEqual(restored,state);self.assertEqual(len(params),2)
            for key,value in [('network_version',5),('training_snapshot','f'*64),('training_purpose','learning'),
                              ('max_positions',128),('input_contract',{**state['input_contract'],'size':9})]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    artifacts.load_candidate(args['root'],{**descriptor,key:value})


if __name__=='__main__':unittest.main()
