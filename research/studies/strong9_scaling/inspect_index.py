"""Audit release structure without opening any policy/value targets."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.release / 'manifest.json').read_text())
    if manifest['release_id'] != '6a26ffa743f3c7be95d783102f6b3596e0ce69893c5a8e28dcca154dc54f05f4':
        raise ValueError('Unexpected release')
    groups = defaultdict(list)
    families = defaultdict(Counter)
    ids, family_splits = set(), {}
    for item in manifest['indices']:
        path = a.release / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Index identity differs')
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['game_id'] in ids:
                    raise ValueError('Duplicate game identity')
                ids.add(row['game_id'])
                split, family = row['split'], row['opening_family']
                if family_splits.setdefault(family, split) != split:
                    raise ValueError('Opening family crosses splits')
                for key in (split, f'{split}/opponent-{row["opponent_index"]}'):
                    groups[key].append(row['rows'])
                    families[key][family] += 1
    result = {}
    for key, lengths in sorted(groups.items()):
        counts = np.asarray(lengths, dtype=np.int32)
        concentration = families[key]
        result[key] = dict(
            games=len(lengths), positions=int(counts.sum()),
            length_quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'p99', 'max'),
                                     map(float, np.quantile(counts, [0, .25, .5, .75, .9, .99, 1])))),
            buckets={str(b): int(np.count_nonzero((counts > low) & (counts <= b)))
                     for low, b in zip((0, 128, 256), (128, 256, 384))},
            opening_families=len(concentration),
            largest_family_games=max(concentration.values()),
            ten_largest_families_games=sum(v for _, v in concentration.most_common(10)),
            opening_family_effective_count=len(lengths)**2 / sum(v*v for v in concentration.values()))
    report = dict(status='passed', created_unix=time.time(), release_id=manifest['release_id'],
                  manifest_sha256=sha(a.release/'manifest.json'), groups=result,
                  duplicate_game_ids=0, cross_split_opening_families=0,
                  test_access='Structural index fields only; no policy/value targets opened',
                  operator_sha256=sha(Path(__file__)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    a.output.chmod(0o444)
    print(json.dumps({key: result[key] for key in ('train', 'validation', 'test')}))


if __name__ == '__main__':
    main()
