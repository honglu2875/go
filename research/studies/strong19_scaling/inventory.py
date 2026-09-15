"""Snapshot completed 19x19 metadata without reading test policy/value targets."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

SSH = ['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']
PYTHON = '/home/go-user/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'
CODE = '''import hashlib,json,sys
from pathlib import Path
root=Path('/dev/shm/go-corpus19');d=json.loads((root/'runs/expert19-v2/config.json').read_text())
sys.path.insert(0,str(Path(d['producer_environment'])/'site-packages'))
import numpy as np
records=[]
for run_id in ('expert19-v1','expert19-v2'):
 c=json.loads((root/'runs'/run_id/'config.json').read_text())['contract']
 cid=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 for p in sorted((root/'corpora'/cid/f"host-{d['host_index']}").glob('worker-*/*.npz')):
  assert not p.is_symlink()
  with np.load(p,allow_pickle=False) as a:m=json.loads(a['metadata'].tobytes())
  assert m['game_id']==p.stem and m['contract_id']==cid and m['board_size']==19
  assert m['teacher_sha256']==c['teacher']['sha256']
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  record={k:m[k] for k in ('game_id','opening_family','split','rows','terminal','opponent_index','expert_color','teacher_sha256','opponent_sha256','board_size','komi','contract_id')}
  records.append(dict(record,path=str(p),sha256=digest,bytes=p.stat().st_size,host=d['host_index'],run_id=run_id))
print(json.dumps(records))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    def read(host):
        command='taskset -c 0,1 '+shlex.join([PYTHON,'-B','-c',CODE])
        return json.loads(subprocess.check_output(SSH+[f'go-user@worker-{host}.example.invalid',command],text=True,timeout=120))
    with ThreadPoolExecutor(4) as pool: records=[r for rows in pool.map(read,range(4)) for r in rows]
    assert len({r['game_id'] for r in records})==len(records)
    families={}
    for r in records:
        assert families.setdefault(r['opening_family'],r['split'])==r['split']
    groups=Counter();positions=Counter()
    for r in records:
        if r['terminal']:
            key=r['split']+'/opponent-'+str(r['opponent_index'])
            groups[key]+=1;positions[key]+=r['rows']
    selected=[]
    for host in range(4):
        for opponent in range(8):
            candidates=[r for r in records if r['host']==host and r['opponent_index']==opponent
                        and r['split']=='train' and r['terminal']]
            if candidates:selected.append(min(candidates,key=lambda r:r['game_id']))
    result=dict(kind='completed_19_inventory',created=time.time(),records=records,
                terminal_games=sum(r['terminal'] for r in records),capped_games=sum(not r['terminal'] for r in records),
                terminal_positions=sum(r['rows'] for r in records if r['terminal']),
                terminal_groups={key:dict(games=n,positions=positions[key]) for key,n in sorted(groups.items())},
                feature_qualification_selection=selected,
                selection='Lexicographically first terminal training game per available host/opponent stratum',
                test_policy_value_targets_decoded=False,operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Inventory and feature qualification inputs only; not a registered training view')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    print(json.dumps({key:result[key] for key in ('terminal_games','capped_games','terminal_positions')}|dict(feature_games=len(selected))))


if __name__=='__main__':main()
