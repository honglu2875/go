"""Freeze a fresh paired KataGo panel before either final baseline is inspected."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'eval/visual_causal'
letters='ABCDEFGHJ'
def number(v):return (int(v[1:])-1)*9+letters.index(v[0])
def vertex(a):return letters[a%9]+str(a//9+1)
def key(opening):
    values=[]
    for flip in (False,True):
        for rotation in range(4):
            tape=[]
            for a in opening:
                y,x=divmod(a,9)
                if flip:x=8-x
                for _ in range(rotation):y,x=x,8-y
                tape.append(y*9+x)
            values.append(tuple(tape))
    return min(values)
def write(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True);f.write('\n')
existing=set()
for path in OUT.glob('*.json'):
    d=json.loads(path.read_text())
    if d.get('size')==9:
        for opening in d.get('openings',[]):
            if len(opening)==2 and all(v.lower()!='pass' for v in opening):existing.add(key([number(v) for v in opening]))
seed=91312817;random=np.random.default_rng(seed);openings=[]
while len(openings)<32:
    a=random.choice(81,size=2,replace=False).tolist();k=key(a)
    if k in existing:continue
    existing.add(k);openings.append(list(map(vertex,a)))
template=json.loads((OUT/'online-control-level1-h0-p00.json').read_text())
recipe=ROOT/'research/recipes/visual_baseline'
for arm in ('cnn','transformer'):
    by_host={str(h):[] for h in range(4)}
    for host in range(4):
        job=0
        for level in (0,1):
            for pair in range(host*8,host*8+8):
                spec={**template,'candidate':f'eval/visual_causal/baseline_{arm}.json',
                    'candidate_cpus':[88+job],'katago_cpus':[64+job],
                    'katago_weights':f'eval/katago_early/level-{level}.json',
                    'openings':[openings[pair]],'seed_prefix':f'gozero-visual-baseline-{seed}-pair-{pair}',
                    'purpose':'Fixed128-update CNN versus causal transformer endpoint. Same weak-teacher training, paired fresh openings and search budgets; distinct history contracts and compute. KataGo is evaluation-only.'}
                name=f'eval/visual_causal/baseline-{arm}-level{level}-h{host}-p{pair:02d}.json'
                write(ROOT/name,spec);by_host[str(host)].append(name);job+=1
    c=json.loads((ROOT/'research/recipes/visual_online/matches_online_control.json').read_text())
    c.update(candidate=f'eval/visual_causal/baseline_{arm}.json',matches_by_host=by_host,max_block=1,seconds=750)
    write(recipe/f'matches_{arm}.json',c)
design={'schema_version':1,'kind':'baseline_fixed_endpoint_katago_design','registered_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
'learning_registration_sha256':hashlib.sha256((ROOT/'research/studies/visual_causal/cnn_baseline_registration.json').read_bytes()).hexdigest(),
'anchors':['eval/katago_early/level-0.json','eval/katago_early/level-1.json'],'pairs_per_anchor_per_arm':32,'games_per_arm':128,
'openings':openings,'openings_sha256':hashlib.sha256(json.dumps(openings,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
'maximum_match_attempts':2,'maximum_seconds_per_attempt':900,'maximum_service_qualification_attempts':2,'service_qualification_timeout':300,
'fixed_selection':'Use the exact final128-update CNN and transformer endpoints, irrespective of held-out losses.',
'scope':'32 fresh D4-disjoint opening pairs per anchor, both colors. Candidate16simulations excluding root, KataGo1visit1thread. Caps retained as unresolved; no Elo or matched-compute claim. The random draw is deterministic from PCG64 seed91312817 after excluding every prior9x9 two-move opening in the workspace.'}
write(ROOT/'research/studies/visual_causal/cnn_katago_design.json',design)
print(design['openings_sha256'])
