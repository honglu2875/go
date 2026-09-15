"""Require exact parameters, Adam moments, rank streams and resumed updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero import checkpoints
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json, read_json, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', required=True); p.add_argument('--resumed', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = SOURCE.parents[2]
    registration = root / 'research/studies/visual_causal/fork_tpu_registration.json'; reg = read_json(registration)
    source = root / '.gozero/snapshots' / reg['source_snapshot']; verify(source)
    c = read_json(source / 'resolved_config.json')
    if checkpoints.sha256(source / 'resolved_config.json') != reg['config_sha256']: raise ValueError('Registered fork differs')
    candidate_path = artifact(source, c['fork']['candidate'])
    if checkpoints.sha256(candidate_path) != c['fork']['candidate_sha256']: raise ValueError('Parent identity differs')
    parent = read_json(candidate_path); parent_path = artifact(root, parent['checkpoint']['path'])
    parent_state, parent_arrays, _ = checkpoints.read(parent_path, expected_manifest_sha256=parent['checkpoint']['manifest_sha256'])
    digest = hashlib.sha256()
    for key in sorted(k for k in parent_arrays if k.startswith('p_')):
        v = parent_arrays[key]; digest.update(canonical_json([key, list(v.shape), str(v.dtype)])); digest.update(v.tobytes())
    initial = digest.hexdigest(); evidence = {}; records = []; total_arrays = 0
    for attempt in (a.full, a.resumed):
        path = root / 'runs' / attempt / 'result.json'; closed = read_json(path)
        if closed['status'] != 'passed' or closed['snapshot_id'] != source.name: raise ValueError('Unqualified fork attempt')
        evidence[str(path.relative_to(root))] = checkpoints.sha256(path)
    resumed_launch = read_json(root / 'runs' / a.resumed / 'launch.json')
    if resumed_launch['resume_attempt'] != a.full or resumed_launch['resume_turn'] != 6: raise ValueError('Wrong controlled restart')
    for host in range(4):
        states, arrays, logs = [], [], []
        for attempt in (a.full, a.resumed):
            base = root / 'runs' / attempt / f'rank-{host}/artifacts'
            report = read_json(base / 'result.json')
            if (report['status'] != 'passed' or not report['training_complete'] or report['turn'] != 8
                    or report['initial_parameter_elements_sha256'] != initial): raise ValueError('Fork initialization or completion differs')
            evidence[str((base / 'result.json').relative_to(root))] = checkpoints.sha256(base / 'result.json')
            path = base / 'checkpoints/turn-000000008'; group = read_json(path.with_suffix('.group.json'))
            state, arr, actors = checkpoints.read(path, expected_manifest_sha256=group['host_manifests'][str(host)])
            if actors != '{}' or state['initial_turn'] != 4 or state['fork_lineage']['parent_turn'] != 4: raise ValueError('Fork lineage differs')
            states.append(state); arrays.append(arr)
            logs.append([json.loads(line) for line in (base / 'metrics.jsonl').read_text().splitlines()
                         if json.loads(line).get('turn', 0) > 6])
        if states[0] != states[1] or logs[0] != logs[1] or set(arrays[0]) != set(arrays[1]): raise ValueError('Recovered scientific state or update records differ')
        for key in arrays[0]: np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
        total_arrays += len(arrays[0])
        records.append({'host': host, 'jax_rank': states[0]['jax_rank'], 'matched_arrays': len(arrays[0]),
                        'matched_metrics_records': len(logs[0]), 'counters': states[0]['counters']})
    result = {'schema_version': 1, 'kind': 'visual_complete_adam_fork_tpu_recovery', 'status': 'passed',
              'operator_snapshot': SOURCE.name, 'source_snapshot': source.name,
              'registration_sha256': checkpoints.sha256(registration), 'evidence': evidence,
              'parent_parameter_elements_sha256': initial, 'matched_arrays': total_arrays, 'hosts': records,
              'scope': 'Exact four-host checkpoint-boundary recovery of an Adam fork. Parameters, both moments, absolute optimizer step, all rank sampler/D4 states and subsequent updates match. No host-loss or mid-update kill claim.'}
    with a.output.open('xb') as f: f.write(canonical_json(result))
    print(json.dumps(result))


if __name__ == '__main__': main()
