"""Freeze both endpoints before starting either new learning curve."""
import math
import time
from pathlib import Path
import sys

from execute_run import ROOT, STUDY, publish, read, require, sha
from monitor import resource_observation
from gozero.snapshots import freeze, verify


def main():
    recipe = ROOT / 'research/recipes/strong19_moe_comparison'
    sys.path.insert(0, str(recipe))
    import train_config
    prereqs = [STUDY / name for name in ('cpu-ce-qualification-001.json',
        'source-budget-lineage-001.json', 'cnn-system-stage-001/audit.json',
        'cnn-system-stage-001/result.json', 'cnn-system-review-001.json',
        'archive-20261004-result-001.json', 'reporting-qualification-001.json')]
    prereqs += [ROOT / 'research/studies' / name for name in (
        'strong19_moe_temporal/cpu-qualification-002.json',
        'strong19_moe_temporal/system-stage-001/audit.json',
        'strong19_moe_temporal/system-review-001.json',
        'strong19_recovery/cohort-disk-backups-001.json',
        'strong19_recovery/draw-replay-001.json')]
    refs = {}
    for arm, attempt, audit in (
        ('dense', 'pod-20260928T030525Z-44bb05aa', 'strong19_recovery/flat-stage-003/audit.json'),
        ('attention', 'pod-20260927T215053Z-92eb8d47', 'strong19_recovery/attention-stage-003/audit.json'),
        ('all_experts', 'pod-20261004T143934Z-cd561c20', 'strong19_moe/balance010-001-stage/audit.json')):
        path = ROOT / 'research/studies' / audit; prereqs.append(path)
        report = read(path); closure = read(ROOT / 'runs' / attempt / 'result.json')
        require(report['status'] == closure['status'] == 'passed' and
                report['snapshot'] == closure['snapshot_id'], 'Reference identity differs')
        refs[arm] = dict(attempt=attempt, audit=str(path.relative_to(ROOT)), audit_sha256=sha(path))
    for path in prereqs:
        require(read(path)['status'] in ('passed', 'prepared'), 'Prerequisite failed: ' + str(path))
    require(read(STUDY / 'reporting-qualification-001.json')['report_operator_sha256'] == sha(STUDY / 'compare.py'),
            'Reporting qualification source changed')
    # The CPU and full-size systems checks bind exactly these source files.
    system = read(STUDY / 'cnn-system-registration-001.json')
    system_snapshot = ROOT / '.gozero/snapshots' / system['snapshot']; verify(system_snapshot)
    manifest = read(system_snapshot / 'manifest.json')
    for name, info in manifest['files'].items():
        if name.endswith('.py') and (name.startswith('packages/') or name.startswith('research/recipes/')):
            require(sha(ROOT / name) == info['sha256'], 'Qualified source changed: ' + name)
    budget = read(STUDY / 'source-budget-lineage-001.json')['budgets']
    dense = read(ROOT / 'research/studies/strong19_recovery/flat-config-001.json')
    drawpath = ROOT / 'research/studies/strong19_recovery/draw-replay-001.json'
    replay = read(drawpath)
    reference = read(ROOT / refs['dense']['audit'])
    arms, reserves = {}, {}
    for arm, peer in (('temporal', 2), ('cnn', 3)):
        path = STUDY / (arm + '-config-001.json'); config = read(path); train_config.validate(config)
        for key in ('seed', 'dataset', 'learner', 'evaluation', 'eval_every', 'steps', 'value_model'):
            require(config[key] == dense[key], 'Common setting changed: ' + key)
        require(config['checkpoint_every'] == 64 and config['checkpoint_disk']['peer'] == peer
                and config['training']['optimizer'] == 'adamw'
                and config['training']['value_objective'] == 'signed_target_cross_entropy'
                and config['training']['value_weight'] == dense['training']['value_weight'], 'Training setting differs')
        if arm == 'temporal':
            parent = read(ROOT / 'research/studies/strong19_moe_temporal/proposal-config-001.json')
            require(config['model'] == parent['model'] and config['training'] == dense['training'],
                    'Temporal-only qualification differs')
        else:
            old = read(ROOT / 'research/studies/strong19_long_pair/cnn-config-001.json')
            require(config['model'] == old['model'] and config['training'] == old['training'], 'CNN control differs')
        snapshot = freeze(ROOT, recipe, path, ROOT / '.gozero/snapshots')
        parameters = budget[arm]['parameters']
        reserves[arm] = math.ceil((parameters * 12 + 4) * 1.01) + (32 << 20)
        arms[arm] = dict(snapshot=snapshot.name, config_sha256=sha(snapshot / 'resolved_config.json'),
            parameters=parameters, replica_peer=peer, timeout_seconds=16000,
            output_directory=arm + '-stage-001')
        systems_path = (ROOT / 'research/studies/strong19_moe_temporal/system-stage-001/audit.json'
                        if arm == 'temporal' else STUDY / 'cnn-system-stage-001/audit.json')
        cases = read(systems_path)['reports'][0]['cases']
        case = next(c for c in cases if c['bucket'] == 512 and
                    c['moe'] == (arm == 'temporal'))
        arms[arm]['first_update_metrics'] = case['updates'][0]['metrics']
    # The owner retires each arm's local midpoint only after its audited endpoint
    # and peer proof. Peer midpoint copies are retained, hence two per peer.
    owner_peak = max(2 * reserves['temporal'], reserves['temporal'] + 2 * reserves['cnn'])
    floors = {0: (2 << 30) + owner_peak, 1: 8 << 30,
              2: (8 << 30) + 2 * reserves['temporal'], 3: (8 << 30) + 2 * reserves['cnn']}
    resources = resource_observation(ROOT / '.gozero/snapshots' / arms['temporal']['snapshot'])
    for observed in resources:
        require(observed['disk_free'] > floors[observed['rank']]
                and observed['shm_free'] > 64 << 30 and observed['memory_available'] > 96 << 30,
                'Insufficient aggregate sequence headroom on rank ' + str(observed['rank']))
    operators = [STUDY / name for name in ('PROTOCOL.md', 'prepare_pair.py', 'execute_pair.py',
        'execute_run.py', 'audit_run.py', 'replicate.py', 'compare.py', 'observation.py',
        'monitor.py', 'backup_pair.py')]
    plan = dict(status='prepared', kind='matched_cnn_dense_moe_pair', created=time.time(),
        order=['temporal', 'cnn'], steps=128, schedule_steps=512, checkpoint_every=64,
        expected_positions=replay['draws'][127]['cumulative_positions'],
        dataset_manifest_sha256=dense['dataset']['manifest_sha256'],
        validation_positions=64371,
        validation_population_sha256=reference['validation_history'][0]['episode_ids_sha256'],
        probe_population_sha256=reference['training_probe_history'][0]['episode_ids_sha256'],
        arms=arms, references=refs, budgets=budget, checkpoint_bytes_per_arm=reserves,
        aggregate_disk_minimum_by_host={str(h): n for h, n in floors.items()},
        admission_resources=resources, draw_reference=str(drawpath.relative_to(ROOT)), draw_sha256=sha(drawpath),
        audit_python=str(ROOT / '.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'),
        operators={str(p.relative_to(ROOT)): sha(p) for p in operators},
        prerequisites={str(p.relative_to(ROOT)): sha(p) for p in prereqs},
        scope='Two fresh 128-update arms. Unchanged 512-update schedule, exact shared samples and targets. '
              'Reuse dense/attention prefixes and all-expert 64 endpoint; no new all-expert learner.')
    path = STUDY / 'pair-registration-001.json'; publish(path, plan)
    print(__import__('json').dumps(dict(status='prepared', sha256=sha(path),
        arms=arms, positions=plan['expected_positions'], aggregate_minimum_gib={h: n / 2**30 for h, n in floors.items()})))


if __name__ == '__main__': main()
