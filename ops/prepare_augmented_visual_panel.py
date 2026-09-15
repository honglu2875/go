"""Write fresh, symmetry-disjoint paired openings and fixed augmented endpoints."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace-root', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    letters = 'ABCDEFGHJ'
    def point(vertex):
        return (int(vertex[1:]) - 1) * 9 + letters.index(vertex[0])
    def vertex(index):
        return letters[index % 9] + str(index // 9 + 1)
    def canonical(pair):
        transformed = []
        for flip in (False, True):
            for rotate in range(4):
                row = []
                for index in pair:
                    y, x = divmod(index, 9)
                    if flip: x = 8 - x
                    for _ in range(rotate): y, x = x, 8 - y
                    row.append(y * 9 + x)
                transformed.append(tuple(row))
        return min(transformed)
    previous = read_json(SOURCE / 'eval/visual_causal/openings.json')
    excluded = {canonical(tuple(map(point, pair))) for pair in previous['excluded_pairs'] + previous['openings']}
    rng = np.random.default_rng(91312341); chosen = []; seen = set(excluded)
    while len(chosen) < 32:
        pair = tuple(int(x) for x in rng.choice(81, size=2, replace=False)); key = canonical(pair)
        if key in seen: continue
        seen.add(key); chosen.append([vertex(x) for x in pair])
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f: f.write(canonical_json(value))
    write(root / 'eval/visual_causal/augmented_openings.json', {'schema_version': 1, 'seed': 91312341,
        'operator_snapshot': SOURCE.name, 'previous_openings_sha256': sha256(SOURCE / 'eval/visual_causal/openings.json'),
        'excluded_d4_classes': sorted(excluded), 'openings': chosen, 'numpy_version': np.__version__,
        'scope': 'Ordered two-move D4 classes exclude all prior recorded opening pairs and each other. No held-out training-position claim.'})
    template = read_json(SOURCE / 'eval/visual_causal/early_exit-early-h0-p00.json')
    base = read_json(SOURCE / 'research/recipes/visual_fast/augmented_service_qualification.json')
    anchors = [('weak', 'eval/katago_early/level-0.json'), ('early', 'eval/katago_early/level-3.json'),
               ('current', 'eval/visual_causal/katago_current.json')]
    for arm, recipe, candidate in [('reference', 'visual_draft', 'early_exit'), ('augmented', 'visual_fast', 'augmented')]:
        config = {**base, 'candidate': f'eval/visual_causal/{candidate}.json', 'mode': 'matches',
                  'qualification': None, 'seconds': 1500, 'matches_by_host': {str(h): [] for h in range(4)}}
        for host in range(4):
            for anchor, weights in anchors:
                for local in range(8):
                    pair = host * 8 + local; index = len(config['matches_by_host'][str(host)])
                    name = f'eval/visual_causal/d4-{arm}-{anchor}-h{host}-p{pair:02d}.json'
                    spec = {**template, 'candidate': config['candidate'], 'katago_weights': weights,
                        'candidate_cpus': [88 + index], 'katago_cpus': [64 + index], 'openings': [chosen[pair]],
                        'seed_prefix': f'gozero-visual-d4-91312341-pair-{pair}',
                        'purpose': 'Fresh D4-disjoint paired endpoint comparison after the fixed offline augmentation screen. Same search and no training use of KataGo.'}
                    write(root / name, spec); config['matches_by_host'][str(host)].append(name)
        write(root / f'research/recipes/{recipe}/matches_d4_{arm}.json', config)
    print(canonical_json({'status': 'prepared', 'pairs_per_anchor_per_arm': 32,
        'games_per_arm': 192, 'openings_sha256': sha256(root / 'eval/visual_causal/augmented_openings.json')}).decode().strip())


if __name__ == '__main__': main()
