"""Check reporting/selection with real replay records and labelled synthetic metrics."""
import copy
import json
from pathlib import Path
import tempfile

import compare
from execute_run import ROOT, STUDY, publish, read, require, sha


def main():
    reference = 'pod-20260928T030525Z-44bb05aa'
    audit_path = ROOT / 'research/studies/strong19_recovery/flat-stage-003/audit.json'
    original = read(audit_path)
    replay_path = ROOT / 'research/studies/strong19_recovery/draw-replay-001.json'
    replay = read(replay_path)
    compare.compare_draws(reference, replay, 128)
    logs = [compare.rows(reference, h) for h in range(4)]
    ranks = [read(ROOT/'runs'/reference/f'rank-{h}/artifacts/result.json')['jax_rank'] for h in range(4)]
    checks = ['real_control_replay_512_rank_updates']
    with tempfile.TemporaryDirectory(prefix='dense-lr-report-') as td:
        root = Path(td); study = root/'study'; study.mkdir()
        compare.ROOT = root; compare.STUDY = study
        (root/'replay.json').write_text(json.dumps(replay))
        for arm in ('control', 'lr06', 'lr15'):
            for h in range(4):
                folder = root/'runs'/arm/f'rank-{h}/artifacts'; folder.mkdir(parents=True)
                (folder/'result.json').write_text(json.dumps(dict(jax_rank=ranks[h])))
        registration = dict(reference={}, arms={}, order=['lr06','lr15'], draw_reference='replay.json',
            dataset_manifest_sha256=original['dataset_manifest_sha256'],
            validation_population_sha256=original['validation_history'][0]['episode_ids_sha256'],
            probe_population_sha256=original['training_probe_history'][0]['episode_ids_sha256'])
        for arm, lr in (('lr06', .0006), ('lr15', .0015)):
            registration['arms'][arm] = dict(peak_learning_rate=lr, output_directory=arm)
            (study/arm).mkdir()

        def fixture(name, modify=None, expected='control', rejection=None):
            audits = {a:copy.deepcopy(original) for a in ('control','lr06','lr15')}
            arm_logs = {a:copy.deepcopy(logs) for a in audits}
            for a, factor in (('lr06',.6),('lr15',1.5)):
                for host_rows in arm_logs[a]:
                    for row in host_rows: row['learning_rate'] *= factor
            if modify: modify(audits, arm_logs)
            compare.rows = lambda a, host=0: arm_logs[a][host]
            control_path = root/'control.json'; control_path.write_text(json.dumps(audits['control']))
            registration['reference'] = dict(attempt='control', peak_learning_rate=.001,
                audit='control.json', audit_sha256=sha(control_path))
            outcomes = {}
            for a in ('lr06','lr15'):
                path = study/a/'audit.json'; path.write_text(json.dumps(audits[a]))
                outcomes[a] = dict(attempt=a, audit_sha256=sha(path))
            folder=root/name; folder.mkdir()
            try: result=compare.report(registration, outcomes, folder)
            except ValueError as error:
                require(rejection and rejection in str(error), f'Unexpected rejection {name}: {error}')
            else:
                require(not rejection, 'Invalid fixture accepted: '+name)
                require(result['selected_for_possible_longer_confirmation']==expected, 'Wrong selection: '+name)
                require(result['positions']==7001181 and not result['production_settings_changed'], 'Wrong result scope')
            checks.append(name)

        def improve(audits, logs):
            for row in audits['lr15']['validation_history']:
                if row['turn']:
                    for k in ('expert_kl','family_kl'): row['metrics'][k] *= .98

        fixture('unchanged_metrics_keep_control')
        fixture('policy_improvement_selects_candidate', improve, expected='lr15')
        def value_regression(a,l):
            improve(a,l)
            for row in a['lr15']['validation_history']:
                if row['turn']: row['metrics']['value_mse'] *= 1.1
        fixture('value_regression_rejects_candidate', value_regression)
        fixture('reject_changed_corpus', lambda a,l:a['lr06'].update(dataset_manifest_sha256='wrong'), rejection='Different corpus')
        fixture('reject_short_run', lambda a,l:a['lr06'].update(steps=127), rejection='Wrong audited model')
        fixture('reject_initial_weights', lambda a,l:a['lr06'].update(initial_parameters_sha256='wrong'), rejection='Initial weights differ')
        fixture('reject_population', lambda a,l:a['lr06']['validation_history'][1].update(episode_ids_sha256='wrong'), rejection='Evaluation differs')
        fixture('reject_draw', lambda a,l:l['lr06'][3][17].update(local_entries_sha256='wrong'), rejection='Logical-rank sampler differs')
        fixture('reject_lr_shape', lambda a,l:l['lr06'][0][42].update(learning_rate=.1), rejection='LR shape or scale differs')
    result=dict(status='passed', report_operator_sha256=sha(STUDY/'compare.py'),
        operator_sha256=sha(Path(__file__)), checks=checks, reference_audit_sha256=sha(audit_path),
        draw_replay_sha256=sha(replay_path), scope='CPU reporting qualification. Synthetic metric changes are fixtures, not learning results; fixtures discarded.')
    publish(STUDY/'reporting-qualification-001.json',result)
    print(json.dumps(result))


if __name__=='__main__': main()
