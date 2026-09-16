"""Pin a stratified 16-game inspection without consulting model predictions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.snapshots import canonical_json,freeze
from gozero.corpus_sequence_batches import Dataset
import numpy as np


def main():
    parent='5e3978d62bc6d8611cd703f5ac894bb70a73289b7a6d62fe615c44e2727a5173'
    c=json.loads((ROOT/'.gozero/snapshots'/parent/'resolved_config.json').read_text())
    path=Path('/dev/shm/gozero-staged-checkpoints/pod-20260916T010420Z-754c0b22/rank-0/artifacts/checkpoints/turn-000000108')
    manifest='13bab6b1e065abb98f78b2c27ad0a0adc91a73793e12be8242b2a58bf4e2eef8'
    saved,arrays,_=checkpoints.read(path,expected_manifest_sha256=manifest,array_prefix='p_')
    h=hashlib.sha256()
    for key in sorted(arrays):
        a=arrays[key];h.update(canonical_json([key,list(a.shape),str(a.dtype)]));h.update(a.tobytes(order='C'))
    d=Dataset(c['dataset']['path'],c['dataset']['manifest_sha256']);selected=[]
    for split in (0,1):
        entries=[('expert',s,e) for s,e in d.indices['expert',split]]
        for opponent in range(8):
            candidates=[e for e in entries if int(d.game_info(e)['opponent'])==opponent]
            e=min(candidates,key=lambda e:bytes(d.game_info(e)['game_id']))
            info=d.game_info(e)
            selected.append(dict(entry=e,split=split,opponent=opponent,game_id=bytes(info['game_id']).decode(),length=int(info['length'])))
    c.update(kind='joint_value_diagnostic',positions=768,selection=selected,
        checkpoint=dict(snapshot=parent,owner_path=str(path),manifest_sha256=manifest,
            parameters_sha256=h.hexdigest(),model_schema=saved['model_schema']))
    config=STUDY/'diagnostic-config-001.json'
    with config.open('xb') as f:f.write(canonical_json(c))
    config.chmod(0o444)
    hp={r['path'].removeprefix('value_head.'):arrays[f'p_{i:04d}'] for i,r in enumerate(saved['model_schema']) if r['path'].startswith('value_head.')}
    with (STUDY/'original-head-001.npz').open('xb') as f:np.savez(f,**hp)
    frozen=freeze(ROOT,Path('research/recipes/strong19_value_debug'),config,ROOT/'.gozero/snapshots')
    result=dict(status='prepared',snapshot=frozen.name,config_sha256=checkpoints.sha256(config),
        positions=sum(r['length'] for r in selected),games=len(selected),
        selection_rule='Smallest game SHA within each of eight opponent strata, independently in train and validation; full histories; no test targets.',
        checkpoint_manifest_sha256=manifest,parameters_sha256=h.hexdigest(),
        head_sha256=checkpoints.sha256(STUDY/'original-head-001.npz'),updates=0,
        operator_sha256=checkpoints.sha256(Path(__file__)))
    with (STUDY/'diagnostic-preparation-001.json').open('xb') as f:f.write(canonical_json(result))
    print(json.dumps(result))


if __name__=='__main__':main()
