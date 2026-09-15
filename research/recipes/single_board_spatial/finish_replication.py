"""Close a registered replication with bounded, sequential, frozen CPU analyses."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import verify, read_json, canonical_json
from gozero.checkpoints import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--control-audit', type=Path, required=True)
    p.add_argument('--control-audit-sha256', required=True)
    p.add_argument('--first-contrast', type=Path, required=True)
    p.add_argument('--first-contrast-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--replication-output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=7300)
    a = p.parse_args()
    verify(SOURCE)
    root = a.workspace_root.resolve()
    if not 0 <= a.wait_seconds <= 7500:
        raise ValueError('Bounded wait required')
    pins = {str(a.registration): a.registration_sha256,
            str(a.control_audit): a.control_audit_sha256,
            str(a.first_contrast): a.first_contrast_sha256}
    for path, wanted in pins.items():
        if sha256(Path(path)) != wanted:
            raise ValueError('Pinned input changed: '+path)
    registration = read_json(a.registration)
    if read_json(a.control_audit)['training_snapshot'] != registration['control_snapshot']:
        raise ValueError('Control audit differs from registration')
    if read_json(a.attempt/'launch.json')['snapshot_id'] != registration['candidate_snapshot']:
        raise ValueError('Candidate attempt differs from registration')
    a.output.mkdir(parents=True, exist_ok=False)
    record = {'kind':'registered_spatial_replication_analysis',
              'operator_snapshot':SOURCE.name, 'input_files':pins,
              'attempt':str(a.attempt), 'automatic_retry':False,
              'wait_seconds':a.wait_seconds, 'analysis_timeout_seconds':600,
              'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (a.output/'launch.json').write_bytes(canonical_json(record))
    environment = {**os.environ, 'JAX_PLATFORMS':'cpu',
                   'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1',
                   'MPLCONFIGDIR':'/tmp/gozero-shared-spatial-mpl',
                   'PYTHONDONTWRITEBYTECODE':'1'}
    python = root/'.venv/bin/python'
    plotting = root/'.gozero/analysis-environments/plotting/bin/python'

    def run(name, executable, script, arguments):
        argv = [str(executable), '-B', str(RECIPE/script), *map(str, arguments)]
        command = {'argv':argv, 'operator_snapshot':SOURCE.name, 'timeout_seconds':600}
        (a.output/(name+'-command.json')).write_bytes(canonical_json(command))
        print(json.dumps({'phase':name, 'status':'starting'}), flush=True)
        with (a.output/(name+'.log')).open('xb') as log:
            subprocess.run(argv, cwd=root, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, timeout=600, check=True)
        print((a.output/(name+'.log')).read_text(), end='', flush=True)

    try:
        deadline = time.monotonic()+a.wait_seconds
        print(json.dumps({'phase':'waiting_for_candidate_closure',
                          'attempt':str(a.attempt)}), flush=True)
        while not (a.attempt/'result.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('Candidate has not closed within the analysis wait budget')
            time.sleep(10)
        closed = read_json(a.attempt/'result.json')
        if closed['status'] != 'passed' or closed['snapshot_id'] != registration['candidate_snapshot']:
            raise ValueError('Candidate did not close successfully with registered source')
        audit = a.output/'spatial-seed2-audit.json'
        run('audit', python, 'audit_learning.py',
            ['--workspace-root',root,'--attempt',a.attempt,'--output',audit])
        comparison = a.output/'paired-analysis'
        run('paired-comparison', plotting, 'report_comparison.py',
            ['--workspace-root',root,
             '--audit','One-token control',a.control_audit,a.control_audit_sha256,
             '--audit','Spatial readout',audit,sha256(audit),
             '--output',comparison])
        contrast = a.output/'spatial-contrast.json'
        run('spatial-contrast', python, 'compare_spatial.py',
            ['--workspace-root',root,'--comparison',comparison/'comparison.json',
             '--comparison-sha256',sha256(comparison/'comparison.json'),
             '--control','One-token control','--candidate','Spatial readout',
             '--output',contrast])
        run('two-seed-replication', plotting, 'compare_replication.py',
            ['--workspace-root',root,
             '--pair',a.first_contrast,a.first_contrast_sha256,
             '--pair',contrast,sha256(contrast),
             '--registration',a.registration,
             '--registration-sha256',a.registration_sha256,
             '--output',a.replication_output])
        result = {'status':'passed', 'operator_snapshot':SOURCE.name,
                  'attempt':str(a.attempt), 'candidate_audit':str(audit),
                  'candidate_audit_sha256':sha256(audit),
                  'candidate_contrast':str(contrast),
                  'candidate_contrast_sha256':sha256(contrast),
                  'replication':str(a.replication_output/'replication.json'),
                  'replication_sha256':sha256(a.replication_output/'replication.json')}
    except Exception as exc:
        result = {'status':'failed', 'operator_snapshot':SOURCE.name,
                  'attempt':str(a.attempt), 'error':repr(exc), 'automatic_retry':False}
        with (a.output/'result.json').open('xb') as stream:
            stream.write(canonical_json(result))
        print(json.dumps(result), flush=True)
        raise
    with (a.output/'result.json').open('xb') as stream:
        stream.write(canonical_json(result))
    for path in a.output.iterdir():
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
