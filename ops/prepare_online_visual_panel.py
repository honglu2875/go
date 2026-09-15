"""Freeze fresh paired KataGo openings for the fixed online continuation endpoints."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256
from gozero.sequence_symmetry import action_map


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--workspace-root', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    letters = 'ABCDEFGHJ'; maps = [action_map(9, i) for i in range(8)]
    def point(v): return (int(v[1:]) - 1) * 9 + letters.index(v[0])
    def vertex(i): return letters[i % 9] + str(i // 9 + 1)
    def canonical(pair): return min(tuple(m[list(pair)].tolist()) for m in maps)
    previous = read_json(SOURCE / 'eval/visual_causal/openings.json')
    augmented = read_json(SOURCE / 'eval/visual_causal/augmented_openings.json')
    excluded = {canonical(tuple(map(point, pair))) for pair in previous['excluded_pairs'] + previous['openings'] + augmented['openings']}
    rng = np.random.default_rng(91312519); chosen = []; seen = set(excluded)
    while len(chosen) < 32:
        pair = tuple(int(x) for x in rng.choice(81, size=2, replace=False)); key = canonical(pair)
        if key in seen: continue
        seen.add(key); chosen.append([vertex(i) for i in pair])
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f: f.write(canonical_json(value))
    write(root / 'eval/visual_causal/online_openings.json', {'schema_version': 1, 'seed': 91312519,
        'operator_snapshot': SOURCE.name, 'previous_openings_sha256': sha256(SOURCE / 'eval/visual_causal/openings.json'),
        'augmented_openings_sha256': sha256(SOURCE / 'eval/visual_causal/augmented_openings.json'),
        'excluded_d4_classes': sorted(excluded), 'openings': chosen, 'numpy_version': np.__version__,
        'scope': 'Fresh ordered two-move D4 classes exclude all recorded earlier opening pairs and one another. Not a held-out training-position claim.'})
    template = read_json(SOURCE / 'eval/visual_causal/d4-augmented-weak-h0-p00.json')
    base = read_json(SOURCE / 'research/recipes/visual_online/suffix_service_qualification.json')
    receipt = read_json(root / base['native_receipt'])
    anchors = [('level1', 'eval/katago_early/level-1.json'), ('level3', 'eval/katago_early/level-3.json')]
    for arm in ('control', 'selfplay'):
        config = {**base, 'candidate': f'eval/visual_causal/online_{arm}.json', 'mode': 'matches',
                  'qualification': None, 'seconds': 900, 'matches_by_host': {str(h): [] for h in range(4)}}
        for host in range(4):
            for anchor, weights in anchors:
                for local in range(8):
                    pair = host * 8 + local; index = len(config['matches_by_host'][str(host)])
                    name = f'eval/visual_causal/online-{arm}-{anchor}-h{host}-p{pair:02d}.json'
                    spec = {**template, 'candidate': config['candidate'], 'katago_weights': weights,
                        'visual_inference': base['inference'], 'inference_native_snapshot': receipt['snapshot_id'],
                        'inference_native_binary_sha256': receipt['binary_sha256'],
                        'candidate_cpus': [88 + index], 'katago_cpus': [64 + index], 'openings': [chosen[pair]],
                        'seed_prefix': f'gozero-visual-online-91312519-pair-{pair}',
                        'purpose': 'Fixed256-update endpoint comparison from identical D4 parent and full Adam state. Old expert data versus new self-play expert data. Fresh D4-disjoint paired openings; KataGo remains evaluation-only.'}
                    write(root / name, spec); config['matches_by_host'][str(host)].append(name)
        write(root / f'research/recipes/visual_online/matches_online_{arm}.json', config)
    print(canonical_json({'status': 'prepared', 'pairs_per_anchor_per_arm': 32,
        'games_per_arm': 128, 'openings_sha256': sha256(root / 'eval/visual_causal/online_openings.json')}).decode().strip())


if __name__ == '__main__': main()
