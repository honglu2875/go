#!/usr/bin/env python3
"""Longer interleaved checks of two already-qualified native benchmark binaries.

This follow-up addresses short-run timing noise; it does not rerun or alter the
scientific rules qualification. Inputs and executable hashes come from attempts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        path = path.resolve()
        record = json.loads((path/'result.json').read_text())
        binary = path/'native/release/go-bench'
        if record['status'] != 'passed' or sha(binary) != record['binary_sha256']:
            raise ValueError('Unqualified or changed benchmark binary: '+name)
        sources[name] = {'binary': str(binary), 'binary_sha256': record['binary_sha256'],
                         'snapshot_id': record['snapshot_id'], 'attempt': str(path)}
    # Primary mask tests now run long enough that even the faster implementation
    # takes seconds. Recheck the only apparent >5% advancement-only regression.
    cases = [(9, 1, 2_000_000, 1), (19, 1, 500_000, 1), (19, 16, 20_000_000, 0)]
    manifest = {'schema_version': 1, 'kind': 'interleaved_rules_confirmation', 'sources': sources,
                'controller_sha256': sha(Path(__file__)), 'cases': cases, 'repeats': 3,
                'reason': 'Initial per-case timings were 0.01–0.3 seconds; clarify mask gains and apparent 16-core advancement regression.',
                'started_unix': time.time(), 'status': 'running'}
    (output/'specification.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows = []
    try:
        for size, workers, proposals, masks in cases:
            for repeat in range(3):
                order = ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']
                for name in order:
                    cpus = list(range(8, 8+workers))
                    argv = ['taskset', '-c', ','.join(map(str,cpus)), sources[name]['binary'],
                            str(size), str(proposals), str(workers), str(masks), '1']
                    before = resource.getrusage(resource.RUSAGE_CHILDREN); start = time.monotonic()
                    execution = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=180)
                    elapsed = time.monotonic()-start; after = resource.getrusage(resource.RUSAGE_CHILDREN)
                    row = json.loads(execution.stdout)
                    row.update(implementation=name, repeat=repeat, cpus=cpus, argv=argv,
                               process_wall_seconds=elapsed,
                               mean_cpu_cores_used=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/elapsed)
                    rows.append(row)
                    with (output/'measurements.jsonl').open('a') as log:
                        log.write(json.dumps(row,sort_keys=True)+'\n')
                    print(json.dumps({k:row[k] for k in ('implementation','size','workers','mask_every','repeat','seconds','legal_moves_per_second','mean_cpu_cores_used')}),flush=True)
        for size, workers, _, masks in cases:
            pair = [r for r in rows if r['size']==size and r['workers']==workers and r['mask_every']==masks]
            for key in ('proposals','legal_moves','passes','terminal_games','truncations','legal_masks'):
                if len({r[key] for r in pair}) != 1:
                    raise AssertionError('Implementations executed different deterministic workloads: '+key)
        manifest['status'] = 'passed'
    except BaseException as error:
        manifest['status'] = 'failed'; manifest['error'] = repr(error); raise
    finally:
        manifest['finished_unix'] = time.time(); manifest['measurements'] = rows
        (output/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
