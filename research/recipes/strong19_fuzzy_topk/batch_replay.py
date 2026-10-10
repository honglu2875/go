"""Rebatch one canonical 128-game stream without changing games or D4 draws."""
import hashlib
import numpy as np
from gozero.snapshots import canonical_json


def validate(c):
    r = c.get('batch_replay')
    if r != dict(source_games_per_host=32, source_steps=512):
        raise ValueError('Expected the registered 128-game, 512-update source stream')
    size = c['learner']['games_per_host']
    if size not in (16, 32, 64) or c['expected_processes'] != 4 or c['learner']['augmentation'] != 'd4':
        raise ValueError('Unsupported replay batch/topology/augmentation')
    if (c['steps']*size != 512*32 or c['learner']['warmup_steps']*size != 40*32
            or c['eval_every']*size != 16*32):
        raise ValueError('Schedule/evaluation must follow the canonical game clock')


def build(c, pools, rank):
    validate(c)
    rng = np.random.Generator(np.random.PCG64(c['seed']+1+104729*rank))
    brng = np.random.Generator(np.random.PCG64(c['seed']+9143))
    arng = np.random.Generator(np.random.PCG64(c['seed']+400003+104729*rank))
    entries=[]; symmetries=[]; lengths=[]
    for i in range(c['batch_replay']['source_steps']):
        d=c['dataset']; warm=d['warmup_buckets']
        bucket=warm[i] if i<len(warm) else int(brng.choice(d['buckets'],p=d['bucket_probabilities']))
        pool=pools['expert',bucket]
        entries.extend(pool[int(j)] for j in rng.integers(len(pool),size=32))
        symmetries.extend(arng.integers(0,8,32).tolist()); lengths.extend([bucket]*32)
    size=c['learner']['games_per_host']
    rows=[dict(bucket=max(lengths[i:i+size]),entries=entries[i:i+size],symmetries=symmetries[i:i+size])
          for i in range(0,len(entries),size)]
    digest=hashlib.sha256(canonical_json(rows)).hexdigest()
    return rows,digest


def state(c, digest, turn):
    if type(turn) is not int or not 0<=turn<=c['steps']: raise ValueError('Invalid replay cursor')
    return dict(schema_version=1,plan_sha256=digest,next_update=turn,
                consumed_local_games=turn*c['learner']['games_per_host'])
