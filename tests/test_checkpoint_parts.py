import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from gozero import checkpoints, checkpoint_parts
from gozero.checkpoint_archive import FILES


class CheckpointPartsTests(unittest.TestCase):
    def test_stream_verification_and_failed_transfer_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'.gozero/checkpoint-part-reservations';base.mkdir(parents=True)
            body=b'checkpoint-fragment'*71
            for suffix,stream in [('good',io.BytesIO(body)),('bad',io.BytesIO(body[:-1]+b'!'))]:
                reserved=base/suffix;reserved.write_bytes(b'\0'*(len(body)+1024))
                spec={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'reservation':str(reserved.relative_to(root))}
                if suffix=='bad':
                    # A distinct expected identity prevents reuse of the good part.
                    spec['sha256']=hashlib.sha256(body+b'!').hexdigest()
                    with self.assertRaises(ValueError):checkpoint_parts.receive(spec,stream,root)
                    self.assertFalse(checkpoint_parts.part_path(spec,root).exists())
                else:
                    result=checkpoint_parts.receive(spec,stream,root)
                    self.assertEqual(result['status'],'verified')
                    self.assertEqual(Path(result['path']).read_bytes(),body)
                    self.assertFalse(reserved.exists())
                    self.assertEqual(Path(result['path']).stat().st_size,len(body))
                    with self.assertRaises(FileExistsError):checkpoint_parts.receive(spec,io.BytesIO(body),root)

    def test_fragmented_archive_reassembles_identical_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'ram'
            identity=checkpoints.write(cache,state={'snapshot_id':'a'*64},arrays={'p_0000':np.random.default_rng(7).normal(size=4096).astype(np.float32)},actors='{}',compress=True)
            files={n:{'bytes':(cache/n).stat().st_size,'sha256':checkpoints.sha256(cache/n)} for n in FILES}
            receipt={'kind':'temporary_ram_checkpoint','status':'temporary','cache_path':str(cache),'manifest_sha256':identity,'files':files,'source_snapshot':'a'*64}
            base=root/'.gozero/checkpoint-part-reservations';base.mkdir(parents=True)
            allocations=[]
            for i in range(4):
                path=base/f'slot-{i}';size=files['arrays.npz']['bytes']//3+10;path.write_bytes(b'\0'*size)
                allocations.append({'host':i,'reservation':str(path.relative_to(root)),'bytes':size})
            def send(source,offset,spec,host,python):
                with Path(source).open('rb') as f:f.seek(offset);data=f.read(spec['bytes'])
                return checkpoint_parts.receive(spec,io.BytesIO(data),root)
            with patch.object(checkpoint_parts,'send_part',side_effect=send):
                result=checkpoint_parts.promote(receipt,allocations,python=sys.executable,root=root)
            self.assertEqual(result['status'],'committed');self.assertGreater(len(result['parts']),1)
            raw_part_path=checkpoint_parts.part_path
            def local_command(host,action,spec,python):
                self.assertEqual(action,'read')
                return [python,'-c','import sys;sys.stdout.buffer.write(open(sys.argv[1],"rb").read())',spec['path']]
            with patch.object(checkpoint_parts,'part_path',side_effect=lambda spec:raw_part_path(spec,root)),patch.object(checkpoint_parts,'command',side_effect=local_command):
                checkpoint_parts.restore(result['descriptor'],root/'restored',python=sys.executable)
            for name in FILES:self.assertEqual((cache/name).read_bytes(),(root/'restored'/name).read_bytes())
            _,arrays,_=checkpoints.read(root/'restored',expected_manifest_sha256=identity)
            np.testing.assert_array_equal(arrays['p_0000'],np.random.default_rng(7).normal(size=4096).astype(np.float32))
            record=json.loads(Path(result['descriptor']).read_text());record['parts'][1]['offset']+=1
            with patch.object(checkpoint_parts,'part_path',side_effect=lambda spec:raw_part_path(spec,root)):
                with self.assertRaises(ValueError):checkpoint_parts.validate_composition(record)


if __name__=='__main__':unittest.main()
