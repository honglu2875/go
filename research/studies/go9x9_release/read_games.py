"""Read exact game records from local downloaded shards; NumPy is the only dependency."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def games(root, split='train', opponent_index=None, verify=True):
    root = Path(root)
    if split not in ('train', 'validation', 'test', None):
        raise ValueError('Unknown split')
    for index in sorted((root / 'index').glob('host-*.jsonl.gz')):
        current = None
        stream = None
        try:
            with gzip.open(index, 'rt') as rows:
                for line in rows:
                    row = json.loads(line)
                    if split is not None and row['split'] != split:
                        continue
                    if opponent_index is not None and row['opponent_index'] != opponent_index:
                        continue
                    if row['shard'] != current:
                        if stream is not None:
                            stream.close()
                        current = row['shard']
                        stream = (root / current).open('rb')
                    stream.seek(row['offset'])
                    data = stream.read(row['bytes'])
                    if verify and hashlib.sha256(data).hexdigest() != row['sha256']:
                        raise ValueError('Game checksum mismatch: '+row['game_id'])
                    with np.load(io.BytesIO(data), allow_pickle=False) as a:
                        arrays = {name:a[name] for name in a.files if name != 'metadata'}
                        metadata = json.loads(a['metadata'].tobytes())
                    yield metadata, arrays
        finally:
            if stream is not None:
                stream.close()
