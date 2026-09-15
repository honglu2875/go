#!/usr/bin/env python3
"""Compile frozen Rust, run differential qualification, and measure CPU scaling."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify


def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        identity = tuple(int((base/name).read_text()) for name in ('physical_package_id', 'core_id'))
        cores.setdefault(identity, cpu)
    return [{'socket': key[0], 'core': key[1], 'cpu': cpu} for key, cpu in sorted(cores.items())]


def run(args):
    if args.snapshot.resolve() != SOURCE:
        raise ValueError('Execute this script from its specified snapshot')
    verify(SOURCE)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    root = args.artifacts_root.resolve()
    config = json.loads((SOURCE/'ops/rules_bench.json').read_text())
    environment = {**os.environ, 'RUSTUP_HOME': str(root/'.gozero/rustup'), 'CARGO_HOME': str(root/'.gozero/cargo')}
    base = [str(root/'.gozero/cargo/bin/cargo')]
    common = ['--manifest-path', str(SOURCE/'Cargo.toml'), '--target-dir', str(output/'native'), '--release', '--locked']
    chosen = physical_cpus()[args.skip_cores:]
    if len(chosen) < max(config['workers']):
        raise ValueError('Insufficient distinct physical cores for benchmark')
    records = []
    result = {'schema_version': 1, 'snapshot_id': SOURCE.name, 'status': 'running',
              'kind': 'rules_qualification', 'config': config, 'cpu_topology': chosen,
              'started_unix': time.time(), 'measurements': records, 'claims_selfplay_throughput': False}
    try:
        for label, argv in [
            ('tests', base+['test', '--workspace']+common+['--', '--include-ignored', '--nocapture']),
            ('build', base+['build', '--workspace']+common),
        ]:
            result[label+'_argv'] = argv
            with (output/(label+'.log')).open('w') as log:
                subprocess.run(argv, env=environment, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=config['per_command_timeout_seconds'])
        binary = output/'native/release/go-bench'
        with binary.open('rb') as f:
            result['binary_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
        result['rust_version'] = subprocess.check_output([str(root/'.gozero/cargo/bin/rustc'), '--version'],
                                                         env=environment, cwd=SOURCE, text=True).strip()
        for size in config['sizes']:
            for case in config['proposal_cases']:
                for repeat in range(config['repeats']):
                    # Rotate order to reduce confounding from thermal/order effects.
                    workers = config['workers'][repeat:]+config['workers'][:repeat]
                    for count in workers:
                        cpus = [entry['cpu'] for entry in chosen[:count]]
                        argv = ['taskset', '-c', ','.join(map(str, cpus)), str(binary), str(size),
                                str(case['proposals_per_worker']), str(count), str(case['mask_every']), str(config['seed'])]
                        before = resource.getrusage(resource.RUSAGE_CHILDREN)
                        start = time.monotonic()
                        execution = subprocess.run(argv, capture_output=True, text=True, check=True,
                                                   timeout=config['per_command_timeout_seconds'])
                        wall = time.monotonic()-start
                        after = resource.getrusage(resource.RUSAGE_CHILDREN)
                        record = json.loads(execution.stdout)
                        record.update(repeat=repeat, cpus=cpus, argv=argv, process_wall_seconds=wall,
                                      process_cpu_seconds=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
                        record['mean_cpu_cores_used'] = record['process_cpu_seconds']/wall
                        records.append(record)
                        with (output/'measurements.jsonl').open('a') as log:
                            log.write(json.dumps(record, sort_keys=True)+'\n')
                        print(json.dumps({k:record[k] for k in ('size','workers','mask_every','repeat','legal_moves_per_second','mean_cpu_cores_used')}),flush=True)
        verify(SOURCE); result['status'] = 'passed'
    except BaseException as error:
        result['status'] = 'failed'; result['error'] = repr(error); raise
    finally:
        result['finished_unix'] = time.time()
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-cores', type=int, default=8, help='Reserve early cores for other task components')
    run(parser.parse_args())
