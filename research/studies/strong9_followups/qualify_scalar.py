"""Qualify scalar isolation and the new initialization/decision semantics."""
import copy
import json
from pathlib import Path
import tempfile
import time

import scalar_source as source
import scalar_compare as comparison
import continue_scalar as continuation
from lr_compare import read, sha

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def main():
    output = STUDY/'scalar-cpu-001.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    parent = ROOT/'.gozero/snapshots/23c9dfe60c0a4c1c376e74f634b568fd1c85f3b13a0c7fef9172859ad38a40fb'
    original = read(parent/'resolved_config.json')
    rejected = []
    def reject(label, action):
        try:
            action()
        except ValueError:
            rejected.append(label)
        else:
            raise ValueError('Invalid fixture accepted: '+label)
    changes = [('seed', ('seed',), original['seed']+1), ('horizon', ('steps',), 2048),
        ('data', ('dataset','manifest_sha256'), '0'*64), ('warmup', ('learner','warmup_steps'), 128),
        ('decay', ('learner','weight_decay'), .02), ('peak', ('learner','learning_rate'), .002),
        ('width', ('model','width'), 1024), ('probe', ('evaluation','training_probe_games'), 128),
        ('test', ('evaluation','run_test'), True)]
    cases = []
    with tempfile.TemporaryDirectory(prefix='strong9-scalar-proof-', dir='/tmp') as folder:
        folder = Path(folder)
        manifest = source.verify(parent)
        same = source.freeze(parent, Path(manifest['recipe']), parent/'resolved_config.json', folder)
        if same.name != parent.name:
            raise ValueError('Unchanged snapshot identity differs')
        for mechanism, value in [('encoder_scale', .01), ('lr_floor', .0001)]:
            candidate = source.configuration(original, mechanism, value)
            for label, path, altered in changes:
                bad = copy.deepcopy(candidate); target = bad
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = altered
                reject(mechanism+':'+label, lambda:source.scalar_only(original, bad, mechanism))
            other = copy.deepcopy(candidate)
            section, field = source.FIELDS['lr_floor' if mechanism == 'encoder_scale' else 'encoder_scale']
            other[section][field] = .0001 if mechanism == 'encoder_scale' else .01
            reject(mechanism+':second_scalar', lambda:source.scalar_only(original, other, mechanism))
            for invalid in (0, -1, float('nan'), True):
                reject(mechanism+':invalid:'+repr(invalid), lambda:source.configuration(original, mechanism, invalid))
            child = source.clone(parent, mechanism, value, folder)
            isolation = source.identical_source(parent, child, mechanism)
            cases.append(dict(mechanism=mechanism, numerical_fixture_value=value,
                              temporary_snapshot=child.name, isolation=isolation))
        child_file = child/manifest['recipe']/'encoder.py'
        raw = child_file.read_bytes();child_file.chmod(0o644)
        child_file.write_bytes(raw+b'\n# disposable source mutation\n')
        reject('source_mutation', lambda:source.identical_source(parent, child, 'lr_floor'))
        child_file.write_bytes(raw);child_file.chmod(0o444);source.verify(child)
    a = [dict(initial_parameter_elements_sha256='a') for _ in range(4)]
    b = [dict(initial_parameter_elements_sha256='b') for _ in range(4)]
    if comparison.initialization_identical(a,a,'lr_floor') is not True:
        raise ValueError('Unchanged LR-floor initialization differs')
    if comparison.initialization_identical(a,b,'encoder_scale') is not False:
        raise ValueError('Changed encoder initialization was called identical')
    reject('floor_changed_initialization', lambda:comparison.initialization_identical(a,b,'lr_floor'))
    reject('scale_unchanged_initialization', lambda:comparison.initialization_identical(a,a,'encoder_scale'))
    reject('rank_disagreement', lambda:comparison.initialization_identical(a,b[:3]+a[:1],'encoder_scale'))
    reject('missing_rank', lambda:comparison.initialization_identical(a,b[:3],'encoder_scale'))
    registration = dict(stages={'seed1':dict(parent_snapshot='parent', snapshot='candidate')},
                        screen_min_relative_gain=.005)
    sample = dict(kind='strong9_scalar_paired_contrast', status='passed', plan_sha256='registration',
        stage='seed1', candidate=dict(snapshot='candidate'), parent=dict(snapshot='parent'),
        screen_min_relative_gain=.005, endpoint_and_tail_gains={k:dict(relative_endpoint_gain=.005,
            relative_last_three_gain=0.) for k in ('expert_kl','family_kl')},
        sustained_overfit=False, screen_passed=True)
    decide = lambda x:continuation.confirmed_screen(x, registration, 'seed1', 'registration')
    if not decide(sample):
        raise ValueError('Exact registered boundary should pass')
    outcomes = ['exact_threshold_passed']
    for metric in ('expert_kl','family_kl'):
        for key in ('relative_endpoint_gain','relative_last_three_gain'):
            bad = copy.deepcopy(sample);bad['endpoint_and_tail_gains'][metric][key] = -.01
            bad['screen_passed'] = False
            if decide(bad):
                raise ValueError('A failed co-primary or tail should stop replication')
            outcomes.append(metric+':'+key+':stopped')
    bad = copy.deepcopy(sample);bad.update(sustained_overfit=True, screen_passed=False)
    if decide(bad):
        raise ValueError('Overfit should stop replication')
    outcomes.append('overfit_stopped')
    for label, key, value in [('wrong_stage','stage','seed2'),('false_flag','screen_passed',False),
                              ('wrong_registration','plan_sha256','changed')]:
        bad = copy.deepcopy(sample);bad[key] = value
        reject(label, lambda:decide(bad))
    bad = copy.deepcopy(sample);bad['endpoint_and_tail_gains']['expert_kl']['relative_endpoint_gain'] = float('nan')
    reject('nonfinite_gain', lambda:decide(bad))
    result = dict(kind='scalar_intervention_cpu_qualification', status='passed', created=time.time(),
        parent_snapshot=parent.name, unchanged_snapshot_reproduced=True, cases=cases,
        rejected_fixtures=rejected, decision_fixtures=outcomes, initialization_semantics_passed=True,
        source_sha256={p.name:sha(p) for p in [Path(__file__),*[STUDY/(x+'.py') for x in
            ('scalar_source','scalar_compare','launch_scalar','finalize_scalar','continue_scalar','lr_compare','lr_source')]]},
        inherited_paired_comparison_qualification_sha256=sha(STUDY/'lr-comparison-cpu-001.json'),
        initialization_preparation_sha256=sha(STUDY/'encoder-scale-preparation-001.json'),
        seconds=time.monotonic()-started,
        scope='Source/config cloning and new initialization/decision checks only. Temporary scalar configurations and decision records are synthetic fixtures and were removed. No scientific scalar selected, learning registration, launch or actual stage finalization. The real-plan launch/continuation inspect remains required after the current LR outcome exists.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    output.chmod(0o444)
    print(json.dumps(dict(status='passed', mechanisms=len(cases), rejected=len(rejected),
                         decisions=len(outcomes), seconds=result['seconds'], sha256=sha(output))))


if __name__ == '__main__':
    main()
