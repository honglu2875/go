"""Replay diverse released games with the existing pinned KataGo V7 adapter."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--worker-receipt',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.time()
    if sha(a.worker_receipt)!='9e7d8d4757c4214e25ea34a8ff3cbe57e1817abd7daf535117e0850eb0a6bd07':
        raise ValueError('Feature worker provenance differs')
    worker=json.loads(a.worker_receipt.read_text());binary=a.worker_receipt.parent/'feature_worker'
    if sha(binary)!=worker['binary_sha256'] or worker['returncode']!=0:
        raise ValueError('Feature worker binary differs')
    manifest=json.loads((a.release/'manifest.json').read_text())
    if manifest['release_id']!='6a26ffa743f3c7be95d783102f6b3596e0ce69893c5a8e28dcca154dc54f05f4':
        raise ValueError('Release differs')
    groups=defaultdict(list)
    for item in manifest['indices']:
        if sha(a.release/item['path'])!=item['sha256']:
            raise ValueError('Index differs')
        with gzip.open(a.release/item['path'],'rt') as f:
            for line in f:
                row=json.loads(line)
                if row['split']=='train':
                    groups[row['opponent_index']].append(row)
    selected=[row for opponent in range(8) for row in sorted(groups[opponent],key=lambda r:r['game_id'])[:64]]
    a.scratch.mkdir(parents=True,exist_ok=True)
    fs=os.statvfs(a.scratch)
    if fs.f_bavail*fs.f_frsize<64*(1<<30)+(1<<30):
        raise ValueError('Insufficient RAM scratch headroom')
    originals=[];games=[]
    for row in selected:
        with (a.release/row['shard']).open('rb') as f:
            f.seek(row['offset']);raw=f.read(row['bytes'])
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('Game bytes differ')
        with np.load(io.BytesIO(raw),allow_pickle=False) as data:
            originals.append((data['stones'].reshape(-1,81),data['legal']))
            games.append(dict(size=9,komi=7.5,actions=data['actions'].tolist()))
    n=sum(row['rows'] for row in selected)
    with tempfile.TemporaryDirectory(prefix='go9-feature-gate-',dir=a.scratch) as temporary:
        tmp=Path(temporary)
        (tmp/'games.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in games))
        completed=subprocess.run([str(binary),str(tmp/'games.jsonl'),str(tmp/'spatial.u8'),
            str(tmp/'global.f32'),str(tmp/'audit.u8')],capture_output=True,text=True,check=True,timeout=120)
        receipt=json.loads(completed.stdout)
        if receipt['games']!=len(selected) or receipt['positions']!=n:
            raise ValueError('Incomplete adapter replay')
        spatial=np.fromfile(tmp/'spatial.u8',np.uint8).reshape(n,9,9,22)
        global_features=np.fromfile(tmp/'global.f32','<f4').reshape(n,19)
        audit=np.fromfile(tmp/'audit.u8',np.uint8).reshape(n,163)
        boards=np.concatenate([b for b,_ in originals]);legal=np.concatenate([l for _,l in originals])
        if not np.array_equal(audit[:,:81],boards) or not np.array_equal(audit[:,81:].astype(bool),legal):
            raise ValueError('KataGo features use different rules or board orientation')
        if not np.isfinite(global_features).all() or not np.all(spatial[...,0]==1) or np.any(spatial>1):
            raise ValueError('Invalid neural features')
        packed=np.packbits(spatial.reshape(n,-1),axis=1,bitorder='little')
        restored=np.unpackbits(packed,axis=1,count=9*9*22,bitorder='little').reshape(spatial.shape)
        packed_legal=np.packbits(legal,axis=1,bitorder='little')
        if not np.array_equal(restored,spatial) or not np.array_equal(np.unpackbits(packed_legal,axis=1,count=82,bitorder='little').astype(bool),legal):
            raise ValueError('Bit packing changed an input')
        out=dict(status='passed',games=len(selected),positions=n,selected_game_ids=[r['game_id'] for r in selected],
            release_id=manifest['release_id'],worker_receipt_sha256=sha(a.worker_receipt),binary_sha256=sha(binary),
            all_board_rows_equal=True,all_legal_rows_equal=True,spatial_bitpacking_exact=True,
            spatial_bytes_per_position=9*9*22,packed_spatial_bytes_per_position=packed.shape[1],
            source_spatial_sha256=sha(tmp/'spatial.u8'),source_global_sha256=sha(tmp/'global.f32'),
            feature_version=7,spatial_channels=22,global_channels=19,rules='positional superko; area; multi-stone suicide; komi 7.5',
            selection='64 lexicographically smallest training game IDs per opponent, independent of target values',
            started_unix=started,finished_unix=time.time(),operator_sha256=sha(Path(__file__)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps({k:out[k] for k in ('status','games','positions','all_board_rows_equal','all_legal_rows_equal','packed_spatial_bytes_per_position')}))


if __name__=='__main__':
    main()
