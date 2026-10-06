"""Independent canonical-stream reconstruction for batch comparisons."""
import hashlib
import json
import numpy as np
from execute_run import ROOT, read, require, sha
from gozero.snapshots import canonical_json


def canonical(c, data, rank):
    path=ROOT/'research/studies/strong19_recovery/draw-replay-001.json'
    require(sha(path)=='067f2f29c8d5c009fcc11d0e27a12b1bbc7d60a251262c035cbb5f3a5e026dbf','Canonical reference changed')
    source=read(path); pools=data.bucket_entries(c['dataset']['buckets'])
    r=np.random.Generator(np.random.PCG64(c['seed']+1+104729*rank))
    a=np.random.Generator(np.random.PCG64(c['seed']+400003+104729*rank))
    games=[]
    for row in source['draws']:
        bucket=row['bucket'];pool=pools['expert',bucket]
        chosen=[pool[int(i)] for i in r.integers(len(pool),size=32)]
        syms=a.integers(0,8,32).tolist();expected=row['ranks'][rank]
        require(hashlib.sha256(canonical_json(chosen)).hexdigest()==expected['local_entries_sha256']
                and syms==expected['local_symmetries'],'Independent canonical reconstruction differs')
        games.extend(dict(entry=e,symmetry=s,bucket=bucket,positions=int(data.game_info(e)['length']))
                     for e,s in zip(chosen,syms,strict=True))
    return games


def expected(c, data, rank):
    games=canonical(c,data,rank);size=c['learner']['games_per_host'];rows=[]
    for turn in range(c['steps']):
        chunk=games[turn*size:(turn+1)*size]
        require(len(chunk)==size,'Incomplete canonical batch')
        rows.append(dict(bucket=max(g['bucket'] for g in chunk),entries=[g['entry'] for g in chunk],
                         symmetries=[g['symmetry'] for g in chunk]))
    return rows,games


def audit_replay(c,data,reports,metrics,states,horizon):
    size=c['learner']['games_per_host'];positions=slots=0;counts=[0]*horizon
    for host,report in reports.items():
        rank=report['jax_rank'];plan,games=expected(c,data,rank)
        digest=hashlib.sha256(canonical_json(plan)).hexdigest()
        require(report['batch_replay']['plan_sha256']==digest,'Reported replay differs')
        require(states[host]['batch_replay_state']==dict(schema_version=1,plan_sha256=digest,next_update=horizon,
                consumed_local_games=horizon*size),'Saved canonical cursor differs')
        for index,row in enumerate(metrics[host]):
            draw=plan[index]
            require(row['bucket']==draw['bucket']
                    and row['local_entries_sha256']==hashlib.sha256(canonical_json(draw['entries'])).hexdigest()
                    and row['local_symmetries']==draw['symmetries']
                    and row['canonical_update_equivalent']==(index+1)*size/32,'Rebatched draw differs')
            counts[index]+=sum(g['positions'] for g in games[index*size:(index+1)*size])
            slots+=size*draw['bucket']
        # These legacy generators are intentionally unused; the replay cursor
        # above is the actual sampling state and must be saved explicitly.
        for key,offset in (('numpy_rng',1+104729*rank),('bucket_rng',9143),('augmentation_rng',400003+104729*rank)):
            require(states[host][key]==np.random.Generator(np.random.PCG64(c['seed']+offset)).bit_generator.state,
                    'Unused legacy RNG changed')
    positions=sum(counts)
    for i,count in enumerate(counts): require(metrics[0][i]['positions']==count,'Actual rebatch loss mass differs')
    for state in states.values():
        require(state['counters']==dict(updates=horizon,expert_positions=positions,padded_position_slots=slots),'Rebatch exposure counters differ')
    return positions,slots
