"""Pinned KataGo V7 inputs over unchanged archived expert policy episodes."""
from pathlib import Path
import numpy as np
from .checkpoints import sha256
from .snapshots import read_json
from .sequence_batches import Dataset as HistoryDataset
from .sequence_symmetry import transform_grid, action_map


class Dataset(HistoryDataset):
    def __init__(self,directory,expected_sha256):
        directory=Path(directory).resolve(strict=True)
        if sha256(directory/'manifest.json') != expected_sha256: raise ValueError('Feature manifest differs')
        overlay=read_json(directory/'manifest.json')
        if overlay['kind'] != 'katago_v7_expert_input_overlay' or overlay['targets_changed']:
            raise ValueError('Expected an input-only expert overlay')
        parent=overlay['parent_dataset']; super().__init__(parent['path'],parent['manifest_sha256'])
        self.parent_manifest=self.manifest; self.manifest=overlay; self.directory=directory
        if len(overlay['shards']) != len(self.parent_manifest['shards']): raise ValueError('Incomplete overlay')
        for original,entry,arrays in zip(self.parent_manifest['shards'],overlay['shards'],self.shards):
            path=directory/entry['arrays']
            if (original['id'] != entry['id'] or original['sha256'] != entry['parent_arrays_sha256']
                    or Path(entry['arrays']).name != entry['arrays'] or sha256(path) != entry['sha256']
                    or not entry['all_boards_equal'] or not entry['all_legal_masks_equal']):
                raise ValueError('Input overlay provenance differs')
            with np.load(path,allow_pickle=False) as f:
                if set(f.files) != {'spatial','global_features'}: raise ValueError('Unexpected overlay arrays')
                spatial,glob=f['spatial'],f['global_features']
            n=len(arrays['expert_actions'])
            if (spatial.shape != (n,self.size,self.size,22) or spatial.dtype != np.uint8
                    or np.any(spatial > 1) or not np.all(spatial[...,0] == 1)
                    or glob.shape != (n,19) or glob.dtype != np.float32 or not np.isfinite(glob).all()):
                raise ValueError('Invalid V7 inputs')
            arrays.update(spatial=spatial,global_features=glob)

    def bucket_entries(self,buckets,split=0):
        if list(buckets) != sorted(set(buckets)) or not buckets or buckets[-1] < self.time:
            raise ValueError('Buckets must cover complete histories')
        result={('expert',b):[] for b in buckets}
        for shard,episode in self.indices['expert',split]:
            offsets=self.shards[shard]['expert_offsets']; n=int(offsets[episode+1]-offsets[episode])
            bucket=next(b for b in buckets if b >= n)
            result['expert',bucket].append(('expert',shard,episode))
        return result

    def batch(self,entries,*,positions,loss_weights=None):
        if loss_weights is not None: raise ValueError('This dataset supports one policy objective')
        b,t,s=len(entries),positions,self.size
        result={'spatial':np.zeros((b,t,s,s,22),np.float32),'global_features':np.zeros((b,t,19),np.float32),
                'actions':np.zeros((b,t),np.int32),'counts':np.zeros(b,np.int32),
                'policies':np.zeros((b,t,s*s+1),np.float32),'legal':np.ones((b,t,s*s+1),bool)}
        for row,entry in enumerate(entries):
            if entry is None: continue
            role,shard,episode=entry
            if role!='expert': raise ValueError('Only expert policy targets are registered')
            a=self.shards[shard]; begin,end=map(int,a['expert_offsets'][episode:episode+2]); n=end-begin
            if n > t: raise ValueError('Would truncate game history')
            result['counts'][row]=n
            for name,key in [('spatial','spatial'),('global_features','global_features'),('actions','expert_actions'),
                             ('policies','expert_policies'),('legal','expert_legal')]:
                result[name][row,:n]=a[key][begin:end]
        return result


def augment(batch,symmetries):
    b,t,s,_,channels=batch['spatial'].shape
    codes=np.asarray(symmetries)
    if channels != 22 or codes.shape != (b,) or codes.dtype.kind not in 'iu' or np.any(codes<0) or np.any(codes>7):
        raise ValueError('Invalid feature symmetry')
    result=dict(batch)
    for key in ('spatial','actions','policies','legal'): result[key]=np.empty_like(batch[key])
    for row,code in enumerate(codes):
        code=int(code)
        result['spatial'][row]=transform_grid(batch['spatial'][row],code,(1,2))
        result['actions'][row]=action_map(s,code)[batch['actions'][row]]
        for key in ('policies','legal'):
            result[key][row,:,:-1]=transform_grid(batch[key][row,:,:-1].reshape(t,s,s),code,(1,2)).reshape(t,s*s)
            result[key][row,:,-1]=batch[key][row,:,-1]
    return result
