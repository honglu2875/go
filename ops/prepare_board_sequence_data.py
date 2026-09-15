#!/usr/bin/env python3
"""Derive exact pre-action stones without changing causal games or targets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256,_sync_directory
from gozero.model_artifacts import artifact
from gozero.native import load_library
from gozero.snapshots import canonical_json,read_json,verify


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('workspace-root','spec','native-receipt','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-spec-sha256',required=True);args=p.parse_args();verify(SOURCE)
    root=args.workspace_root.resolve();spec=read_json(args.spec)
    if sha256(args.spec)!=args.expected_spec_sha256 or spec['kind']!='board_causal_teacher_dataset':raise ValueError('Overlay registration differs')
    if type(spec['workers']) is not int or not 1<=spec['workers']<=8:raise ValueError('Bounded preprocessing workers required')
    base=spec['parent_dataset'];parent=artifact(root,base['path']);parent_manifest=read_json(parent/'manifest.json')
    if sha256(parent/'manifest.json')!=base['manifest_sha256'] or parent_manifest['kind']!='causal_teacher_dataset':raise ValueError('Parent dataset differs')
    if len(parent_manifest['shards'])>16:raise ValueError('Parent dataset exceeds bounded source selection')
    rules={k:parent_manifest['spec'][k] for k in ('size','komi','scoring')}
    receipt=read_json(args.native_receipt)
    if receipt['snapshot_id']!=SOURCE.name:raise ValueError('Observation replay native source differs')
    native=load_library(args.native_receipt.parent/receipt['filename'],receipt['binary_sha256'])
    if getattr(native,'OBSERVATION_REPLAY_ABI_VERSION',None)!=1:raise ValueError('Observation replay ABI missing')
    output=args.output.resolve()
    if not output.is_relative_to(root) or output.is_relative_to(SOURCE):raise ValueError('Overlay output must be outside source in workspace')
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+output.name+'.partial-',dir=output.parent))
    started=time.time();report={'schema_version':1,'kind':'board_overlay_preparation','status':'running','operator_snapshot':SOURCE.name,'native':receipt,'started_unix':started,'spec_sha256':args.expected_spec_sha256}
    def process(item):
        path=parent/item['arrays']
        if Path(item['arrays']).name!=item['arrays'] or sha256(path)!=item['sha256']:raise ValueError('Parent shard bytes differ')
        arrays={};counts={}
        with np.load(path,allow_pickle=False) as saved:
            for role in ('expert','behavior'):
                actions=saved[role+'_actions'];offsets=saved[role+'_offsets']
                stones,legal,outcome_json=native.replay_observations(json.dumps(rules),np.ascontiguousarray(actions),np.ascontiguousarray(offsets))
                stones=stones.reshape(len(actions),rules['size']**2);legal=legal.reshape(len(actions),rules['size']**2+1)
                outcomes=json.loads(outcome_json)
                if len(outcomes)!=len(offsets)-1:raise ValueError('Native episode coverage differs')
                if role=='expert' and not np.array_equal(legal,saved['expert_legal']):raise ValueError('Replayed exact legality differs from stored MCTS observations')
                targets=saved['expert_values'] if role=='expert' else None
                capped=saved['behavior_capped'] if role=='behavior' else None
                for game,ending in enumerate(outcomes):
                    begin,end=map(int,offsets[game:game+2])
                    if role=='expert':
                        if not ending['terminal'] or ending['white_score'] is None:raise ValueError('Expert episode lacks exact terminal state')
                        expected=np.where(np.arange(end-begin)%2==0,-np.sign(ending['white_score']),np.sign(ending['white_score'])).astype(np.float32)
                        if not np.array_equal(targets[begin:end],expected):raise ValueError('Native outcome differs from stored expert values')
                    elif ending['terminal']==bool(capped[game]) or (ending['white_score'] is None)!=bool(capped[game]):
                        raise ValueError('Behavior terminal/capped status differs')
                arrays[role+'_stones']=stones
                counts[role+'_rows']=len(actions);counts[role+'_games']=len(outcomes)
                counts[role+'_terminal_games']=sum(o['terminal'] for o in outcomes)
        name=f'shard-{item["id"]:02d}.npz';path=temporary/name
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        path.chmod(0o444)
        entry={'id':item['id'],'arrays':name,'sha256':sha256(path),'bytes':path.stat().st_size,'parent_arrays_sha256':item['sha256'],**counts}
        print(json.dumps({'kind':'board_shard_verified',**entry}),flush=True);return entry
    try:
        with ThreadPoolExecutor(max_workers=spec['workers']) as pool:shards=list(pool.map(process,parent_manifest['shards']))
        if sha256(parent/'manifest.json')!=base['manifest_sha256'] or sha256(args.spec)!=args.expected_spec_sha256:raise ValueError('Input manifest changed')
        verify(SOURCE)
        manifest={'schema_version':1,'kind':'board_causal_teacher_dataset','operator_snapshot':SOURCE.name,'native':receipt,'spec_sha256':args.expected_spec_sha256,'spec':spec,'parent_dataset':base,'rules':rules,'shards':shards,
            'input_contract':'Absolute colors 0/1/2 immediately BEFORE each target action. Whole histories start empty; exact native rules replay every action. Future states are separate per-position observations, never inputs to an earlier prediction.','targets_changed':False}
        (temporary/'manifest.json').write_bytes(canonical_json(manifest))
        report.update(status='passed',finished_unix=time.time(),dataset_manifest_sha256=sha256(temporary/'manifest.json'))
        (temporary/'receipt.json').write_bytes(canonical_json(report))
        for path in temporary.iterdir():
            with path.open('rb') as stream:os.fsync(stream.fileno())
            path.chmod(0o444)
        _sync_directory(temporary)
        if output.exists():raise FileExistsError(output)
        temporary.rename(output);_sync_directory(output.parent)
    except BaseException as e:
        report.update(status='failed',error=repr(e),finished_unix=time.time())
        (temporary/'failure.json').write_bytes(canonical_json(report));raise
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
