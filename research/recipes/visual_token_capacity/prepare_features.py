#!/usr/bin/env python3
"""Reconstruct inputs only, verifying every board/legal row against Rust replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visual-dataset',type=Path,required=True)
    p.add_argument('--visual-sha256',required=True)
    p.add_argument('--worker-receipt',type=Path,required=True)
    p.add_argument('--worker-receipt-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--first-shards',type=int,default=16)
    args=p.parse_args(); verify(SOURCE); started=time.time()
    if not 1 <= args.workers <= 16 or not 1 <= args.first_shards <= 16: raise ValueError('Bounded jobs required')
    if sha256(args.visual_dataset/'manifest.json') != args.visual_sha256: raise ValueError('Visual manifest differs')
    if sha256(args.worker_receipt) != args.worker_receipt_sha256: raise ValueError('Feature worker receipt differs')
    worker=read_json(args.worker_receipt); binary=args.worker_receipt.parent/'feature_worker'
    if worker['returncode'] != 0 or sha256(binary) != worker['binary_sha256']: raise ValueError('Worker differs')
    visual=read_json(args.visual_dataset/'manifest.json'); parent=visual['parent_dataset']; directory=Path(parent['path'])
    if sha256(directory/'manifest.json') != parent['manifest_sha256']: raise ValueError('Parent manifest differs')
    raw=read_json(directory/'manifest.json'); size=visual['rules']['size']; area=size*size
    args.output.mkdir(parents=True,exist_ok=False)
    def one(pair):
        original, overlay=pair
        if original['id'] != overlay['id']: raise ValueError('Shard order differs')
        original_path=directory/original['arrays']; overlay_path=args.visual_dataset/overlay['arrays']
        if sha256(original_path) != original['sha256'] or sha256(overlay_path) != overlay['sha256']:
            raise ValueError('Shard bytes differ')
        with np.load(original_path,allow_pickle=False) as f:
            actions=f['expert_actions']; offsets=f['expert_offsets']
        n=len(actions)
        with tempfile.TemporaryDirectory(prefix='go-katago-v7-',dir=args.scratch) as temporary:
            tmp=Path(temporary)
            with (tmp/'games.jsonl').open('w') as f:
                for begin,end in zip(offsets[:-1],offsets[1:]):
                    f.write(json.dumps({'size':size,'komi':visual['rules']['komi'],
                                        'actions':actions[int(begin):int(end)].tolist()})+'\n')
            command=[str(binary),str(tmp/'games.jsonl'),str(tmp/'spatial.u8'),str(tmp/'global.f32'),str(tmp/'audit.u8')]
            result=subprocess.run(command,capture_output=True,text=True,timeout=1800)
            (args.output/f'worker-{original["id"]:02d}.log').write_text(result.stdout+result.stderr)
            result.check_returncode(); summary=json.loads(result.stdout)
            if summary['positions'] != n or summary['games'] != len(offsets)-1: raise ValueError('Worker coverage differs')
            spatial=np.fromfile(tmp/'spatial.u8',np.uint8).reshape(n,size,size,22)
            glob=np.fromfile(tmp/'global.f32',np.dtype('<f4')).reshape(n,19)
            audit=np.fromfile(tmp/'audit.u8',np.uint8).reshape(n,2*area+1)
            with np.load(overlay_path,allow_pickle=False) as f:
                if not np.array_equal(audit[:,:area],f['expert_stones']): raise ValueError('KataGo/Rust board disagreement')
                if not np.array_equal(audit[:,area:].astype(bool),f['expert_legal']):
                    mismatch=np.argwhere(audit[:,area:].astype(bool) != f['expert_legal'])
                    raise ValueError(f'KataGo/Rust legality disagreement: {mismatch[:8].tolist()}')
            if not np.isfinite(glob).all() or not np.all(spatial[...,0] == 1): raise ValueError('Invalid features')
            target=args.output/f'shard-{original["id"]:02d}.npz'
            with target.open('xb') as f: np.savez_compressed(f,spatial=spatial,global_features=glob)
            target.chmod(0o444)
            entry={'id':original['id'],'arrays':target.name,'sha256':sha256(target),'bytes':target.stat().st_size,
                   'expert_games':len(offsets)-1,'expert_rows':n,'parent_arrays_sha256':original['sha256'],
                   'visual_arrays_sha256':overlay['sha256'],'all_boards_equal':True,'all_legal_masks_equal':True}
            print(json.dumps(entry),flush=True); return entry
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        entries=list(pool.map(one,list(zip(raw['shards'],visual['shards']))[:args.first_shards]))
    manifest={'schema_version':1,'kind':'katago_v7_expert_input_overlay','operator_snapshot':SOURCE.name,
              'parent_dataset':parent,'visual_dataset':{'path':str(args.visual_dataset.resolve()),'manifest_sha256':args.visual_sha256},
              'worker_receipt':{'path':str(args.worker_receipt.resolve()),'sha256':args.worker_receipt_sha256},
              'rules':visual['rules'],'feature_version':7,'spatial_channels':22,'global_channels':19,
              'history_modes':'explicit all-legacy BoardHistoryModes; default MiscNNInputParams',
              'shards':entries,'targets_changed':False,'expert_only':True,
              'started_unix':started,'finished_unix':time.time()}
    (args.output/'manifest.json').write_bytes(canonical_json(manifest)); (args.output/'manifest.json').chmod(0o444)
    print(json.dumps({'status':'passed','manifest_sha256':sha256(args.output/'manifest.json'),
                      'positions':sum(x['expert_rows'] for x in entries)}),flush=True)
if __name__=='__main__': main()
