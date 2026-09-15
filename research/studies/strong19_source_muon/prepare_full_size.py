"""Freeze a full-size execution fixture from the qualified source optimizer.

Preparation does not launch a TPU job or select scientific training settings.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, freeze, read_json, verify

CPU = '3bd93c2a8fb2ee67fa8c071ebd435b9325bb921eae3a218c14d8092ad2f9cb70'
FULL = '3d33b7b8e3d255de5acad1e63c81513f8b89421bc654d3a0e1989e4cdc62b194'
EVIDENCE = {
    'research/studies/strong19_source_muon/harness-result-001.json':
        '0cab9d71e6aa105c7189bf0569cd573150698c8005ad2228d23ca5d3ebcad7e5',
    'research/studies/strong19_source_muon/inference-cpu-001/result.json':
        '3678c8735a4e95a3263727015d9aca1c592237cd0ed92bdba44da99ab7b774fc',
    'research/studies/strong19_train/resume-transfer-result-002.json':
        'c4267a77d11ab664f77e57c44ba122757a6cd74ca62a49a1d4ffe91706c83d91',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unchanged(manifest, names):
    result = {}
    for name in names:
        digest = manifest['files'][name]['sha256']
        if sha(ROOT / name) != digest:
            raise ValueError('Qualified source changed: ' + name)
        result[name] = digest
    return result


def main():
    output = STUDY / 'full-size-preparation-001.json'
    config_path = STUDY / 'tpu-full-size-config-001.json'
    if output.exists() or config_path.exists():
        raise FileExistsError('Inspect the existing preparation before continuing')
    for name, expected in EVIDENCE.items():
        if sha(ROOT / name) != expected or read_json(ROOT / name)['status'] != 'passed':
            raise ValueError('Prerequisite has not passed with the recorded identity')
    cpu = ROOT / '.gozero/snapshots' / CPU
    full = ROOT / '.gozero/snapshots' / FULL
    cpu_manifest, full_manifest = verify(cpu), verify(full)
    recipe = cpu_manifest['recipe']
    recipe_sources = unchanged(cpu_manifest, [n for n in cpu_manifest['files']
        if n.startswith(recipe + '/')])
    # The source optimizer shares the exact qualified loader and checkpoint
    # transport with the corrected full-size preparation.
    shared_names = [n for n in full_manifest['files'] if
        n.startswith('packages/gozero/') or n in
        ('ops/pod_run.py', 'ops/run_host.py', 'ops/stage_joint_checkpoint.py')]
    shared_sources = unchanged(full_manifest, shared_names)
    c = read_json(cpu / 'resolved_config.json')
    reference = read_json(full / 'resolved_config.json')
    if c['dataset'] != reference['dataset'] or c['seed'] != reference['seed']:
        raise ValueError('Execution fixture or sampling seed differs')
    for name in ('model', 'value_model', 'platform', 'expected_processes',
                 'expected_devices', 'checkpoint_temporary',
                 'checkpoint_temporary_uncompressed'):
        c[name] = reference[name]
    c['steps'] = c['checkpoint_every'] = 4
    c['eval_every'] = 2
    c['learner']['games_per_host'] = reference['learner']['games_per_host']
    c['learner']['source_runtime']['epochs'] = [[4]]
    c['training']['chunk_frames'] = reference['training']['chunk_frames']
    sys.path.insert(0, str(cpu / recipe))
    import train_config
    train_config.validate(c)
    # Verify every broad memory/model choice against the already traced joint
    # configuration, while keeping source-optimizer settings explicit.
    assert c['model'] == reference['model']
    assert c['value_model'] == reference['value_model']
    assert c['training']['purpose'] == 'qualification'
    assert c['training']['optimizer'] == 'muon_source_runtime'
    assert c['training']['path'] == reference['training']['path'] == 'bounded'
    assert c['training']['value_weight'] == reference['training']['value_weight']
    with config_path.open('xb') as stream:
        stream.write(canonical_json(c))
    config_path.chmod(0o444)
    snapshot = freeze(ROOT, ROOT / recipe, config_path, ROOT / '.gozero/snapshots')
    frozen = verify(snapshot)
    for name, digest in {**recipe_sources, **shared_sources}.items():
        if frozen['files'][name]['sha256'] != digest:
            raise ValueError('Prepared source differs from qualified dependency')
    record = dict(kind='joint_source_muon_full_size_preparation', status='prepared',
        created=time.time(), operator_sha256=sha(Path(__file__)),
        cpu_snapshot=CPU, full_shape_reference_snapshot=FULL,
        snapshot=snapshot.name, config=dict(path=str(config_path.relative_to(ROOT)),
            sha256=sha(config_path)), prerequisites=EVIDENCE,
        recipe_sources=recipe_sources, shared_sources=shared_sources,
        planned_stages=['full_4', 'prefix_2', 'resume_to_4'],
        expected_transition=dict(lookahead_synchronized=[3],
            source_norm_snapshot=[3], source_epoch_flush=[4]),
        tpu_jobs_launched=False,
        scope='Full-size execution fixture only. Same CNN/value shapes and '
            '128 complete-game batch as the prior prepared AdamW qualification. '
            'Includes source Muon/AuxAdam, all fast/slow state and runtime scalars. '
            'The 256-position schedule reference, initial sample offset, short '
            'epoch and Lookahead period are qualification settings, not selected '
            'scientific hyperparameters. No TPU memory/timing/recovery claim.')
    with output.open('xb') as stream:
        stream.write(canonical_json(record))
    output.chmod(0o444)
    print(json.dumps(dict(status='prepared', snapshot=snapshot.name,
        receipt_sha256=sha(output), tpu_jobs_launched=False)))


if __name__ == '__main__':
    main()
