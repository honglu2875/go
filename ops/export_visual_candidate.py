#!/usr/bin/env python3
"""Publish a verified visual candidate descriptor without duplicating its weights."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_artifacts import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace-root', 'training-result', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    result_path = a.training_result.resolve(); result = read_json(result_path)
    saved = result['latest_checkpoint']
    candidate = {'schema_version': 1, 'kind': 'visual_causal_checkpoint',
        'training_snapshot': result['snapshot_id'], 'network_version': result['turn'],
        'training_complete': result['training_complete'],
        'training_result': {'path': str(result_path.relative_to(root)), 'sha256': sha256(result_path)},
        'checkpoint': {'path': str(Path(saved['owner_checkpoint_path']).relative_to(root)),
                       'group_sha256': saved['group_sha256'], 'manifest_sha256': saved['manifest_sha256']}}
    checked = validate(root, candidate, allow_partial=a.allow_partial)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        f.write(canonical_json(candidate))
    print(canonical_json({'status': 'passed', 'operator_snapshot': SOURCE.name,
          'candidate_sha256': sha256(a.output), 'parameters': sum(s['elements'] for s in checked['model_schema']),
          'training_complete': candidate['training_complete']}).decode().strip())


if __name__ == '__main__':
    main()
