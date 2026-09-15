"""Count complete-game duplicates on train/validation; keep test arrays closed."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def action_maps(size):
    mappings = []
    for flip in (False, True):
        for turns in range(4):
            current = []
            for action in range(size*size):
                row, col = divmod(action, size)
                if flip:
                    col = size - 1 - col
                for _ in range(turns):
                    row, col = col, size - 1 - row
                current.append(row*size+col)
            mappings.append([*current, size*size])
    return np.asarray(mappings, dtype=np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); started = time.time()
    manifest = json.loads((a.release/'manifest.json').read_text())
    if manifest['release_id'] != '6a26ffa743f3c7be95d783102f6b3596e0ce69893c5a8e28dcca154dc54f05f4':
        raise ValueError('Unexpected release')
    a.output.mkdir(parents=True, exist_ok=False)
    mappings = action_maps(9)
    exact = defaultdict(Counter); symmetric = defaultdict(Counter)
    all_families = {}; lengths = {}; positions = Counter(); games = Counter()
    current = None; stream = None
    try:
        with gzip.open(a.output/'identities.jsonl.gz', 'xt') as output:
            for item in manifest['indices']:
                if sha(a.release/item['path']) != item['sha256']:
                    raise ValueError('Index hash differs')
                with gzip.open(a.release/item['path'], 'rt') as index:
                    for line in index:
                        row = json.loads(line)
                        split = row['split']
                        if split == 'test':
                            continue
                        if row['shard'] != current:
                            if stream is not None:
                                stream.close()
                            current = row['shard']; stream = (a.release/current).open('rb')
                        stream.seek(row['offset']); raw = stream.read(row['bytes'])
                        if hashlib.sha256(raw).hexdigest() != row['sha256']:
                            raise ValueError('Record bytes differ')
                        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                            actions = archive['actions']
                        if actions.shape != (row['rows'],) or np.any(actions<0) or np.any(actions>81):
                            raise ValueError('Invalid actions')
                        transformed = mappings[:, actions]
                        trajectory = hashlib.sha256(min(x.tobytes() for x in transformed)).hexdigest()
                        raw_trajectory = hashlib.sha256(actions.astype(np.uint8).tobytes()).hexdigest()
                        if all_families.setdefault(trajectory, split) != split:
                            raise ValueError('Complete trajectory crosses splits')
                        for group in (split, f'{split}/opponent-{row["opponent_index"]}'):
                            exact[group][raw_trajectory] += 1
                            symmetric[group][trajectory] += 1
                            positions[group] += row['rows']; games[group] += 1
                        lengths[trajectory] = row['rows']
                        kept = {key: row[key] for key in ('game_id', 'split', 'rows', 'opponent_index', 'expert_color', 'opening_family')}
                        output.write(json.dumps({**kept, 'trajectory_sha256': trajectory,
                                                 'oriented_trajectory_sha256': raw_trajectory}, sort_keys=True)+'\n')
                        if sum(games[s] for s in ('train', 'validation')) % 10000 == 0:
                            print(json.dumps(dict(games=sum(games[s] for s in ('train','validation')), elapsed=time.time()-started)), flush=True)
    finally:
        if stream is not None:
            stream.close()
    groups = {}
    for key, count in symmetric.items():
        groups[key] = dict(games=games[key], positions=positions[key], exact_unique_trajectories=len(exact[key]),
            d4_unique_trajectories=len(count), d4_unique_positions=sum(lengths[t] for t in count),
            duplicate_games=games[key]-len(count), largest_duplicate_group=max(count.values()),
            repeated_groups=sum(v>1 for v in count.values()))
    report = dict(status='passed', source_release_id=manifest['release_id'], started_unix=started,
        finished_unix=time.time(), groups=groups, cross_split_complete_trajectories=0,
        identity_rule='SHA256 of lexicographically smallest complete uint8 action sequence over D4; includes pass=81',
        scope='Only training/validation actions decoded; test NPZ records and all target arrays remain closed',
        identities_sha256=sha(a.output/'identities.jsonl.gz'), operator_sha256=sha(Path(__file__)))
    with (a.output/'result.json').open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    for path in a.output.iterdir():
        path.chmod(0o444)
    print(json.dumps({k:groups[k] for k in ('train','validation')}),flush=True)


if __name__ == '__main__':
    main()
