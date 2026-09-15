"""Exact pre-action board overlays on unchanged causal teacher populations."""
from pathlib import Path

import numpy as np

from .checkpoints import sha256
from .sequence_batches import Dataset as HistoryDataset
from .snapshots import read_json


class Dataset(HistoryDataset):
    def __init__(self,directory,expected_sha256,*,rank=0,world=1,board_mode='exact'):
        directory=Path(directory).resolve(strict=True)
        if sha256(directory/'manifest.json')!=expected_sha256:raise ValueError('Board overlay identity differs')
        manifest=read_json(directory/'manifest.json')
        if manifest['kind']!='board_causal_teacher_dataset' or board_mode not in ('exact','empty'):
            raise ValueError('Unknown board overlay or input condition')
        base=manifest['parent_dataset']
        super().__init__(base['path'],base['manifest_sha256'],rank=rank,world=world)
        self.parent_manifest=self.manifest;self.manifest=manifest;self.directory=directory;self.board_mode=board_mode
        selected=[x for x in manifest['shards'] if x['id']%world==rank]
        parent=[x for x in self.parent_manifest['shards'] if x['id']%world==rank]
        if len(selected)!=len(self.shards) or [x['id'] for x in selected]!=[x['id'] for x in parent]:
            raise ValueError('Board overlay shard coverage differs')
        for item,original,arrays in zip(selected,parent,self.shards):
            if item['parent_arrays_sha256']!=original['sha256']:raise ValueError('Overlay targets a different parent shard')
            path=directory/item['arrays']
            if Path(item['arrays']).name!=item['arrays'] or sha256(path)!=item['sha256']:
                raise ValueError('Board overlay shard differs')
            with np.load(path,allow_pickle=False) as archive:
                if set(archive.files)!={'expert_stones','behavior_stones'}:raise ValueError('Unexpected board overlay arrays')
                for role in ('expert','behavior'):
                    stones=archive[role+'_stones']
                    if stones.dtype!=np.uint8 or stones.shape!=(len(arrays[role+'_actions']),self.size**2) or not np.isin(stones,[0,1,2]).all():
                        raise ValueError('Invalid absolute-color board observations')
                    if np.any(stones[arrays[role+'_offsets'][:-1]]):raise ValueError('Episode must begin on an empty board')
                    arrays[role+'_stones']=stones

    def batch(self,entries):
        result=super().batch(entries)
        result['stones']=np.zeros((len(entries),self.time,self.size**2),np.uint8)
        if self.board_mode=='exact':
            for row,entry in enumerate(entries):
                if entry is None:continue
                role,shard,episode=entry;data=self.shards[shard];start,end=data[role+'_offsets'][episode:episode+2]
                result['stones'][row,:end-start]=data[role+'_stones'][start:end]
        return result
