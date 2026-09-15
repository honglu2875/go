#!/usr/bin/env python3
"""Register fresh opening pairs and fixed single-thread KataGo panels."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'eval/online_annealing'


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    OUT.mkdir(exist_ok=False); used = set(); inputs = {}
    for path in sorted((ROOT / 'eval').rglob('*.json')):
        if path.is_relative_to(OUT): continue
        record = json.loads(path.read_text()); openings = record.get('openings') if isinstance(record, dict) else None
        if isinstance(openings, list):
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for opening in openings:
                if len(opening) == 2 and all(isinstance(v, str) for v in opening):
                    used.add(tuple(v.upper() for v in opening))
    vertices = [c + str(y) for y in range(9, 0, -1) for c in 'ABCDEFGHJ']
    random = np.random.default_rng(91310301); fresh = []; draws = 0; previous = len(used)
    while len(fresh) < 34:
        draws += 1
        if draws > 100000: raise ValueError('Opening pool exhausted')
        opening = tuple(random.choice(vertices, size=2, replace=False).tolist())
        if opening in used or opening[::-1] in used: continue
        fresh.append(list(opening)); used.add(opening); used.add(opening[::-1])
    book = {'schema_version': 1, 'seed': 91310301, 'draws': draws, 'primary': fresh[:32], 'strong': fresh[32:],
            'excluded_previous_ordered_pairs': previous, 'exclusion_inputs_sha256': inputs,
            'scope': 'New exact two-stone pairs and reversals versus existing match files; D4 equivalents are not excluded.'}
    write(OUT / 'openings.json', book)
    write(OUT / 'parent.json', json.loads((ROOT / 'eval/gumbel_attention/candidate.json').read_text()))
    for arm in ('parent', 'inherit', 'anneal'):
        for panel_name in ('primary', 'strong'):
            entries = []; count = 4 if panel_name == 'primary' else 2
            for chunk in range(count):
                start = 64 + 8 * chunk; label = f'{arm}-{panel_name}-{chunk}'
                config = {'schema_version': 1, 'candidate': f'eval/online_annealing/{arm}.json', 'candidate_cpuct': 0.0,
                    'candidate_simulations_excluding_root': 16, 'candidate_cpus': list(range(start, start + 4)),
                    'katago_cpus': list(range(start + 4, start + 8)), 'game_timeout_seconds': 180,
                    'katago_config': 'eval/continued_selfplay/katago_v16_single_thread.cfg',
                    'katago_weights': 'eval/katago_early/level-3.json' if panel_name == 'primary' else 'eval/katago_9x9.json',
                    'katago_max_visits': 16, 'size': 9, 'komi': 7.5, 'max_game_moves': 648,
                    'openings': book['primary'][8 * chunk:8 * (chunk + 1)] if panel_name == 'primary' else [book['strong'][chunk]],
                    'seed_prefix': f'online-annealing-91310301-{panel_name}-{chunk}',
                    'purpose': 'Registered final-checkpoint online learning-rate pilot; parent is a contemporaneous baseline. Fixed CPU/search conditions; no KataGo targets in training.'}
                write(OUT / (label + '.json'), config)
                entries.append({'id': label, 'group': f'{arm}-{panel_name}', 'spec': f'eval/online_annealing/{label}.json'})
            write(OUT / f'{arm}_{panel_name}_panel.json', {'schema_version': 1, 'workers': count, 'matches': entries})
    print(json.dumps({'opening_pairs': len(fresh), 'draws': draws, 'excluded_pairs': previous,
                      'opening_sha256': hashlib.sha256((OUT / 'openings.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
