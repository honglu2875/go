"""Run one pinned intervention: qualify, learn, audit, and conditionally replicate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); manifest = verify(SOURCE)
    recipe = SOURCE / manifest['recipe']
    if sha256(a.registration) != a.registration_sha256:
        raise ValueError('Prospective registration changed')
    spec = read_json(a.registration)
    if spec['kind'] != 'registered_sequential_followup' or spec['intervention'] not in ('encoder_attention', 'first_pass_aux'):
        raise ValueError('Unsupported intervention')
    if spec['replication_flag'] not in ('replication_screen_passed', 'replication_eligible', 'exploratory_quality_replication'):
        raise ValueError('Unrecognized replication rule')
    exploratory = spec['replication_flag'] == 'exploratory_quality_replication'
    if exploratory and (spec['intervention'] != 'encoder_attention' or spec.get('minimum_replicated_quality_gain') != .05
                        or 'existing_first_seed' not in spec or 'existing_qualification' not in spec):
        raise ValueError('Exploratory follow-up requires the explicit existing attention screen and a 5% quality criterion')
    expected_script = {'encoder_attention': 'compare_encoder.py', 'first_pass_aux': 'compare_auxiliary.py'}
    if spec['contrast_script'] != expected_script[spec['intervention']]:
        raise ValueError('Wrong scientific contrast')
    def pinned(pin):
        path = (root / pin['path']).resolve()
        path.relative_to(root)
        if sha256(path) != pin['sha256']:
            raise ValueError('Pinned evidence changed: ' + str(path))
        return path
    for pin in spec['prerequisite_evidence']:
        path = pinned(pin)
        if pin.get('required_status') and read_json(path).get('status') != pin['required_status']:
            raise ValueError('Prerequisite did not pass')
    references = [pinned(x) for x in spec['references']]
    if len(references) != 2:
        raise ValueError('Two paired reference audits required')
    budget = pinned(spec['cpu_budget'])
    minimum_checkpoint_space = 3 * 2**30
    if 'checkpoint_space_contract' in spec:
        contract = spec['checkpoint_space_contract']
        parameters = read_json(budget)['cases']['128']['analytical']['trainable_parameters']
        if (contract['parameter_count'] != parameters or contract['float32_array_copies'] != 3
                or contract['artifact_margin_bytes'] != 256 * 2**20):
            raise ValueError('Checkpoint-space contract must cover full float32 parameters, Adam moments and artifact margin')
        for reference in references:
            audit = read_json(reference)
            report = read_json(root / 'runs' / audit['attempt'] / 'rank-0/artifacts/result.json')
            if (report['parameter_count'] != parameters or audit['checkpoint_array_count'] != 3 * len(report['model_schema'])):
                raise ValueError('Parent checkpoint schema does not match the registered storage bound')
        minimum_checkpoint_space = parameters * 4 * 3 + contract['artifact_margin_bytes']
    snapshots = {phase: root / '.gozero/snapshots' / spec[phase + '_snapshot']
                 for phase in ('qualification', 'first_seed', 'second_seed')}
    for source in snapshots.values():
        verify(source)
    storage = root / '.gozero/snapshots' / spec['storage_operator_snapshot']; verify(storage)
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time(); commands = []; attempts = {}; result = {'status': 'running'}
    python = root / '.venv/bin/python'; plot = root / '.gozero/analysis-environments/plotting/bin/python'
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    cpu = {**env, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
           'MPLCONFIGDIR': '/tmp/gozero-spatial-followups-mpl'}
    launch = {'operator_snapshot': SOURCE.name, 'registration': str(a.registration),
        'registration_sha256': a.registration_sha256, 'specification': spec,
        'created_unix': started, 'automatic_retry': False, 'max_new_tpu_attempts': 1 if exploratory else 3}
    (a.output / 'launch.json').write_bytes(canonical_json(launch)); (a.output / 'launch.json').chmod(0o444)
    def execute(label, argv, *, timeout=600, is_cpu=True):
        command = {'phase': label, 'argv': list(map(str, argv)), 'timeout_seconds': timeout}
        commands.append(command)
        with (a.output / (label + '-command.json')).open('xb') as f:
            f.write(canonical_json(command))
        print(json.dumps({'phase': label, 'status': 'starting'}), flush=True)
        with (a.output / (label + '.log')).open('xb') as f:
            subprocess.run(command['argv'], cwd=root, env=cpu if is_cpu else env,
                stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        print(json.dumps({'phase': label, 'status': 'passed'}), flush=True)
    def analysis(label, script, args, plotting=False):
        execute(label, ['taskset', '-c', '112-119', plot if plotting else python,
                       '-B', recipe / script, *args])
    def pod(phase):
        for i, relocation in enumerate(spec.get('before_phase_relocations', {}).get(phase, [])):
            attempt = relocation['attempt']
            if attempt.startswith('@'):
                attempt = attempts[attempt[1:]]
            label = f'{phase}-relocate-{i}'
            execute(label, [python, '-B', storage / 'ops/retain_checkpoint_replicas.py',
                '--workspace-root', root, '--output', a.output / (label + '-receipt.json'),
                '--apply', '--attempt', attempt, '--copy-missing-replicas',
                '--retained-hosts', str(relocation['host']), '--relocate-single-owner'],
                timeout=1500, is_cpu=False)
        if any(not (x.parent / 'result.json').exists() for x in (root / 'runs').glob('pod-*/launch.json')):
            raise ValueError('Another TPU attempt is open')
        disk = os.statvfs(root)
        if disk.f_bavail * disk.f_frsize < minimum_checkpoint_space:
            raise ValueError('Insufficient persistent checkpoint space: need ' + str(minimum_checkpoint_space) + ' bytes')
        source = snapshots[phase]
        try:
            execute(phase, [python, '-B', source / 'ops/pod_run.py', '--snapshot', source,
                '--workspace-root', root, '--timeout', '7200', '--prepare-timeout', '180',
                '--controller-cpus', '32'], timeout=7500, is_cpu=False)
        finally:
            log = a.output / (phase + '.log')
            if log.exists():
                entries = [json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
                launches = [x for x in entries if x.get('kind') == 'pod_attempt']
                if len(launches) == 1:
                    attempts[phase] = Path(launches[0]['attempt']).name
        attempt = root / 'runs' / attempts[phase]; closed = read_json(attempt / 'result.json')
        if closed['status'] != 'passed' or closed['snapshot_id'] != source.name:
            raise ValueError('Attempt did not pass')
        return attempt
    def pair(phase, attempt, index):
        audit = a.output / (phase + '-audit.json')
        analysis(phase + '-audit', 'audit_learning.py',
            ['--workspace-root', root, '--attempt', attempt, '--output', audit])
        folder = a.output / (phase + '-comparison')
        analysis(phase + '-compare', 'report_comparison.py', ['--workspace-root', root,
            '--audit', spec['control_label'], references[index], spec['references'][index]['sha256'],
            '--audit', spec['candidate_label'], audit, sha256(audit), '--output', folder], plotting=True)
        contrast = a.output / (phase + '-contrast.json')
        analysis(phase + '-contrast', spec['contrast_script'], ['--workspace-root', root,
            '--comparison', folder / 'comparison.json', '--comparison-sha256', sha256(folder / 'comparison.json'),
            '--control', spec['control_label'], '--candidate', spec['candidate_label'], '--output', contrast])
        return contrast
    try:
        if 'existing_qualification' in spec:
            qfile = pinned(spec['existing_qualification'])
            prior_q = read_json(qfile)
            if prior_q['status'] != 'passed' or prior_q['snapshot'] != snapshots['qualification'].name or prior_q['cpu_budget_sha256'] != sha256(budget):
                raise ValueError('Prior qualification differs')
            pinned({'path':prior_q['audit_path'],'sha256':prior_q['audit_sha256']})
            attempts['qualification'] = prior_q['attempt']
        else:
            qualification = pod('qualification')
            qfile = a.output / 'qualification.json'
            analysis('qualification-audit', 'audit_qualification.py', ['--workspace-root', root,
                '--attempt', qualification, '--cpu-budget', budget, '--output', qfile,
                '--audit-output', a.output / 'qualification-learning-audit.json'])
        q = read_json(qfile)
        for phase in ('first_seed', 'second_seed'):
            source = snapshots[phase]; m = verify(source); c = read_json(source / 'resolved_config.json')
            if c['model'] != q['model'] or c['dataset'] != q['dataset']:
                raise ValueError('Qualified model or data differ')
            for name, wanted in q['numerical_sources'].items():
                if sha256(source / m['recipe'] / name) != wanted:
                    raise ValueError('Qualified numerical source differs: ' + name)
        if 'existing_first_seed' in spec:
            first_contrast = pinned(spec['existing_first_seed'])
            previous = read_json(first_contrast)
            comp = read_json(pinned({'path':previous['comparison'],'sha256':previous['comparison_sha256']}))
            candidate = next(x for x in comp['arms'] if x['label'] == previous['candidate'])
            if previous['status'] != 'passed' or candidate['snapshot'] != snapshots['first_seed'].name:
                raise ValueError('Prior first seed differs')
            attempts['first_seed'] = candidate['attempt']
        else:
            first = pod('first_seed'); first_contrast = pair('first_seed', first, 0)
        screen = read_json(first_contrast); replicated_main = False; replicated_draft = False; replicated_quality = False
        eligible = (screen['relative_endpoint_kl_improvement'] >= .05 if exploratory else screen[spec['replication_flag']])
        if eligible:
            registration = a.output / 'replication-registration.json'
            second_config = read_json(snapshots['second_seed'] / 'resolved_config.json')
            r = {'kind': 'prospective_' + spec['intervention'] + '_replication', 'created_unix': time.time(),
                'screen_contrast_sha256': sha256(first_contrast), 'seed': second_config['seed'],
                'control_snapshot': read_json(references[1])['training_snapshot'],
                'candidate_snapshot': snapshots['second_seed'].name,
                'selection_rule': spec['replication_rule'], 'intervention_registration_sha256': a.registration_sha256}
            registration.write_bytes(canonical_json(r)); registration.chmod(0o444)
            second = pod('second_seed'); second_contrast = pair('second_seed', second, 1)
            analysis('replication', 'compare_replication.py', ['--workspace-root', root,
                '--pair', first_contrast, sha256(first_contrast), '--pair', second_contrast, sha256(second_contrast),
                '--registration', registration, '--registration-sha256', sha256(registration),
                '--output', a.output / 'replication'], plotting=True)
            r = read_json(a.output / 'replication/replication.json')
            replicated_main = r['each_seed_passed_one_percent_latency_screen']
            replicated_draft = r.get('each_seed_passed_draft_utility_screen', False)
            replicated_quality = exploratory and all(x['relative_kl_improvement'] >= .05 for x in r['pairs'])
        result = {'status': 'passed', 'first_contrast': str(first_contrast),
            'first_contrast_sha256': sha256(first_contrast), 'first_seed_screen_passed': screen['replication_screen_passed'],
            'replication_eligible': eligible, 'replicated_gain': replicated_main,
            'replicated_quality_gain': replicated_quality,'exploratory_followup':exploratory,
            'replicated_draft_utility': replicated_draft,
            'next_action': 'Review both scientific outcomes before selecting any subsequent parent. This controller launches no other intervention.'}
    except BaseException as error:
        result = {'status': 'failed', 'error': repr(error)}; raise
    finally:
        reused = {phase: attempt for phase, attempt in attempts.items()
                  if (phase == 'qualification' and 'existing_qualification' in spec)
                  or (phase == 'first_seed' and 'existing_first_seed' in spec)}
        result.update(operator_snapshot=SOURCE.name, registration_sha256=a.registration_sha256,
            started_unix=started, ended_unix=time.time(), attempts=attempts, commands=commands, automatic_retry=False)
        result.update(reused_attempts=reused,new_attempts={k:v for k,v in attempts.items() if k not in reused})
        with (a.output / 'result.json').open('xb') as f:
            f.write(canonical_json(result))
        for path in a.output.iterdir():
            if path.is_file():
                path.chmod(0o444)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
