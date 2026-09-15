"""Compare the streaming V7 provider with every frozen execution-fixture input."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.corpus_format import unpack_spatial,unpack_legal
from gozero.snapshots import canonical_json,read_json
from gozero.v7_replay import Replay


def main():
    output=STUDY/'feature-equivalence-001.json'
    if output.exists():raise FileExistsError(output)
    started=time.monotonic()
    data=Path('/dev/shm/gozero-datasets/joint19-execution-001')
    expected='da627b1968dbb353acbf3b630a9feb84cf30080957a5660c8bc4125917fbe0ae'
    if sha256(data/'manifest.json')!=expected:raise ValueError('Fixture changed')
    manifest=read_json(data/'manifest.json')
    offline_path=ROOT/'.gozero/build/katago-v7-12e944b6/receipt.json'
    stream_path=ROOT/'.gozero/build/katago-v7-stream-001/receipt.json'
    offline,stream=read_json(offline_path),read_json(stream_path)
    if (sha256(offline_path)!=stream['reference_receipt_sha256']
            or offline['source_revision']!=stream['source_revision'] or offline['objects']!=stream['objects']
            or offline['binary_sha256']!=manifest['feature_worker_binary_sha256']):
        raise ValueError('Providers do not share the pinned KataGo feature objects')
    binary=ROOT/'.gozero/build/katago-v7-stream-001/feature_stream'
    for record in (offline,stream):
        program=(ROOT/record['command'][record['command'].index('-o')+1]).resolve()
        if not program.is_relative_to(ROOT) or sha256(program)!=record['binary_sha256']:
            raise ValueError('Actual feature provider binary differs')
    for name,digest in offline['objects'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Shared feature object changed')
    games=positions=0;rows=[]
    with Replay(binary,stream['binary_sha256'],size=manifest['size'],komi=manifest['komi']) as replay:
        for shard in manifest['shards']:
            arrays={}
            for name in ('actions','expert_offsets','games','spatial','global_features','legal'):
                item=shard['files'][name];path=data/item['path']
                if sha256(path)!=item['sha256']:raise ValueError('Frozen feature inputs changed')
                arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
            for i,game in enumerate(arrays['games']):
                if game['split'] not in (0,1):raise ValueError('Test histories are closed')
                lo,hi=map(int,arrays['expert_offsets'][i:i+2]);tape=arrays['actions'][lo:hi].tolist()
                # Corpus rows are pre-move states. Omitting only the final move
                # includes every stored row and never queries a terminal state.
                actual=replay([tape[:-1]])[0]
                wanted=dict(spatial=unpack_spatial(arrays['spatial'][lo:hi],manifest['size']),
                            global_features=np.asarray(arrays['global_features'][lo:hi]),
                            legal=unpack_legal(arrays['legal'][lo:hi],manifest['size']))
                for name,target in wanted.items():
                    value=actual[name]
                    if target.shape!=value.shape or target.dtype!=value.dtype or target.tobytes()!=value.tobytes():
                        raise ValueError('Feature provider differs on '+name+' in '+bytes(game['game_id']).decode())
                games+=1;positions+=hi-lo
                rows.append(dict(game_id=bytes(game['game_id']).decode(),split=int(game['split']),positions=hi-lo))
    if games!=19 or positions!=7992:raise ValueError('Fixture coverage is incomplete')
    contract={k:manifest[k] for k in ('feature_version','size','komi','spatial_channels','global_channels')}
    result=dict(kind='exact_v7_provider_comparison',status='passed',created=time.time(),seconds=time.monotonic()-started,
                operator_sha256=sha256(Path(__file__)),dataset_manifest_sha256=expected,input_contract=contract,
                training_provider_sha256=offline['binary_sha256'],inference_provider_sha256=stream['binary_sha256'],
                source_revision=offline['source_revision'],offline_build_receipt_sha256=sha256(offline_path),
                inference_build_receipt_sha256=sha256(stream_path),shared_objects_identical=True,
                shared_objects_sha256=hashlib.sha256(canonical_json(offline['objects'])).hexdigest(),
                games=games,positions=positions,all_spatial_bytes_equal=True,all_global_bytes_equal=True,
                all_legal_masks_equal=True,test_targets_decoded=False,records=rows,
                scope='Every pre-move spatial/global/legality row in the frozen19 execution fixture matches bit-for-bit. Both binaries share the same verified KataGo core objects; their protocol wrappers differ. Qualified provider compatibility, not identical binaries or exhaustive equivalence over every legal Go history.')
    with output.open('xb') as f:f.write(canonical_json(result))
    output.chmod(0o444)
    print(json.dumps(dict(status='passed',games=games,positions=positions,seconds=result['seconds'],sha256=sha256(output))))


if __name__=='__main__':main()
