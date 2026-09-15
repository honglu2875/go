#!/usr/bin/env python3
"""Create fresh paired opening specifications before the architecture pilot."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'eval/state_expert_distillation'


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    used = set(); inputs = {}
    for path in sorted((ROOT / 'eval').rglob('*.json')):
        if path.is_relative_to(OUT):
            continue
        record = json.loads(path.read_text())
        openings = record.get('openings') if isinstance(record, dict) else None
        if isinstance(openings, list):
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for opening in openings:
                if len(opening) == 2 and all(isinstance(v, str) for v in opening):
                    used.add(tuple(v.upper() for v in opening))
    vertices = [c + str(y) for y in range(9, 0, -1) for c in 'ABCDEFGHJ']
    random = np.random.default_rng(91310241); fresh = []; draws = 0; previous_count = len(used)
    while len(fresh) < 34:
        draws += 1
        if draws > 100000:
            raise ValueError('Opening pool exhausted')
        opening = tuple(random.choice(vertices, size=2, replace=False).tolist())
        if opening in used or opening[::-1] in used:
            continue
        fresh.append(list(opening)); used.add(opening); used.add(opening[::-1])
    book = {'schema_version': 1, 'seed': 91310241, 'draws': draws, 'early': fresh[:32], 'strong': fresh[32:],
            'excluded_previous_ordered_pairs': previous_count, 'exclusion_inputs_sha256': inputs,
            'scope': 'Fresh exact two-stone ordered pairs and their reversals versus existing match configurations. D4 equivalents are not excluded; this is no claim of untouched board distributions.'}
    write(OUT / 'openings.json', book)
    for arm, old in (('history', 'mse'), ('state', 'bce2')):
        for panel_name in ('early', 'strong'):
            template_path = ROOT / f'eval/value_logit_distillation/{old}_{panel_name}_panel.json'
            panel = json.loads(template_path.read_text())
            for index, entry in enumerate(panel['matches']):
                config = json.loads((ROOT / entry['spec']).read_text())
                config['candidate'] = f'eval/state_expert_distillation/{arm}.json'
                config['purpose'] = 'Fixed-final expert-only history/spatial architecture ablation with a common frozen opponent observer.'
                if panel_name == 'early':
                    config['openings'] = book['early'][8 * index:8 * (index + 1)]
                    config['seed_prefix'] = f'state-expert-91310241-early-{index}'
                else:
                    config['openings'] = [book['strong'][index % 2]]
                    config['seed_prefix'] = f'state-expert-91310241-strong-v{config["katago_max_visits"]}-{index % 2}'
                entry['id'] = entry['id'].replace(old, arm)
                entry['group'] = entry['group'].replace(old, arm)
                entry['spec'] = f'eval/state_expert_distillation/{entry["id"]}.json'
                write(ROOT / entry['spec'], config)
            write(OUT / f'{arm}_{panel_name}_panel.json', panel)
    print(json.dumps({'opening_pairs': len(fresh), 'excluded_previous_pairs': previous_count, 'draws': draws,
                      'opening_sha256': hashlib.sha256((OUT / 'openings.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
