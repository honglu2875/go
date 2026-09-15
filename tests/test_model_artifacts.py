import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gozero import checkpoints
from gozero.model_artifacts import validate_candidate
from gozero.snapshots import canonical_json,freeze


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);recipe=self.root/'research/recipes/fixture';recipe.mkdir(parents=True)
        (recipe/'train.py').write_text('pass\n');(recipe/'recipe.json').write_text('{"id":"fixture"}\n')
        (recipe/'config.json').write_text('{"seed":27}\n');(self.root/'uv.lock').write_text('version=1\n')
        self.source=freeze(self.root,recipe,recipe/'config.json',self.root/'.gozero/snapshots')
        self.parameters={'p_0000':np.arange(4,dtype=np.float32)}
        self.checkpoint=self.root/'runs/attempt/checkpoints/turn-000000012'
        state={'schema_version':1,'snapshot_id':self.source.name,'config_sha256':checkpoints.sha256(self.source/'resolved_config.json'),
               'turn':12,'jax_rank':0,'world_size':1,'native_sha256':'f'*64,'counters':{'updates':5,'real_moves':96}}
        manifest=checkpoints.write(self.checkpoint,state=state,arrays={**self.parameters,'replay_x':np.ones((128,8),np.float32)},actors='{}',compress=True)
        group=self.checkpoint.parent/(self.checkpoint.name+'.group.json')
        group.write_bytes(canonical_json({'schema_version':1,'snapshot_id':self.source.name,'config_sha256':state['config_sha256'],
            'turn':12,'updates':5,'world_size':1,'rank_manifests':[manifest]}))
        self.weights=self.root/'weights.npz';np.savez(self.weights,**self.parameters)
        self.candidate={'schema_version':2,'training_snapshot':self.source.name,'network_version':5,
            'checkpoint':{'path':str(self.checkpoint.relative_to(self.root)),'manifest_sha256':manifest,'group_sha256':checkpoints.sha256(group)},
            'model_export_path':'weights.npz','model_export_sha256':checkpoints.sha256(self.weights)}

    def test_exact_parameters_are_loaded_without_materializing_replay(self):
        state,arrays,_=checkpoints.read(self.checkpoint,array_prefix='p_')
        self.assertEqual(set(arrays),{'p_0000'});np.testing.assert_array_equal(arrays['p_0000'],self.parameters['p_0000'])
        identity=validate_candidate(self.root,self.candidate)
        self.assertEqual(identity['checkpoint_turn'],12);self.assertEqual(identity['native_binary_sha256'],'f'*64)

    def test_updated_export_hash_cannot_disguise_weights_from_another_step(self):
        np.savez(self.weights,p_0000=self.parameters['p_0000']+1)
        self.candidate['model_export_sha256']=checkpoints.sha256(self.weights)
        with self.assertRaisesRegex(ValueError,'weights differ from checkpoint'):
            validate_candidate(self.root,self.candidate)

    def test_partial_read_still_verifies_the_entire_array_archive(self):
        path=self.checkpoint/'arrays.npz';blob=bytearray(path.read_bytes());blob[len(blob)//2]^=1
        path.chmod(0o644);path.write_bytes(blob)
        with self.assertRaisesRegex(ValueError,'Checkpoint integrity failure'):
            checkpoints.read(self.checkpoint,array_prefix='p_')

    def test_group_network_version_and_final_export_lineage_are_checked(self):
        with self.assertRaisesRegex(ValueError,'scientific identity'):
            validate_candidate(self.root,{**self.candidate,'network_version':6})
        final=self.root/'result.json'
        record={'status':'passed','snapshot_id':self.source.name,'counters':{'updates':5},'turn':12,
                'native':{'binary_sha256':'f'*64},'model_export_sha256':self.candidate['model_export_sha256']}
        final.write_bytes(canonical_json(record))
        old={k:v for k,v in self.candidate.items() if k!='checkpoint'}
        old.update(schema_version=1,training_result_path='result.json',training_result_sha256=checkpoints.sha256(final))
        self.assertEqual(validate_candidate(self.root,old)['checkpoint_turn'],12)
        np.savez(self.weights,p_0000=self.parameters['p_0000']+2);old['model_export_sha256']=checkpoints.sha256(self.weights)
        with self.assertRaisesRegex(ValueError,'final training identity'):
            validate_candidate(self.root,old)


if __name__=='__main__':unittest.main()
