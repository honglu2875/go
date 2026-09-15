"""Freeze real-board TPU qualification sources after the recovery correction."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, freeze, verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = STUDY / 'real19-snapshots-002.json'
    if output.exists():
        raise FileExistsError('Preparation receipt already exists')
    evidence = {}

    def checked(name, expected):
        path = STUDY / name
        if sha(path) != expected:
            raise ValueError('Qualification evidence changed: ' + name)
        record = json.loads(path.read_text())
        if record.get('status') != 'passed':
            raise ValueError('Qualification did not pass: ' + name)
        evidence[name] = expected
        return record

    cpu = checked('harness-real19-result-001.json',
                  '3c666756d4beb671a6f4bf7796838934ac1c3f60c363158eccbf2835cad8e6f0')
    transfer = checked('resume-transfer-result-002.json',
                       'c4267a77d11ab664f77e57c44ba122757a6cd74ca62a49a1d4ffe91706c83d91')
    regression_path = STUDY / 'pod-regression-001.json'
    regression = checked(regression_path.name, sha(regression_path))
    for source, digest in regression['source_sha256'].items():
        if sha(ROOT / source) != digest:
            raise ValueError('Pod regression source changed: ' + source)

    recipe_sources = {}
    for arm, record in cpu['models'].items():
        snapshot = ROOT / '.gozero/snapshots' / record['snapshot']
        manifest = verify(snapshot)
        for source, item in manifest['files'].items():
            if source.startswith('research/recipes/strong19_train/') and source.endswith('.py'):
                if sha(ROOT / source) != item['sha256']:
                    raise ValueError('Numerical recipe differs from CPU harness: ' + source)
                recipe_sources[source] = item['sha256']

    transfer_source = ROOT / '.gozero/snapshots' / transfer['snapshot']
    transfer_manifest = verify(transfer_source)
    recovery_sources = {}
    for source in ('ops/stage_joint_checkpoint.py', 'ops/pod_run.py', 'ops/run_host.py',
                   'packages/gozero/src/gozero/joint_resume.py',
                   'packages/gozero/src/gozero/checkpoints.py'):
        digest = transfer_manifest['files'][source]['sha256']
        if sha(ROOT / source) != digest:
            raise ValueError('Recovery differs from four-host qualification: ' + source)
        recovery_sources[source] = digest

    previous_path = STUDY / 'real19-snapshots-001.json'
    previous = json.loads(previous_path.read_text())
    snapshots = {}; configs = {}
    for arm in ('cnn', 'transformer'):
        name = 'tpu_' + arm
        record = previous['configurations'][name]
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise ValueError('TPU qualification configuration changed')
        config = json.loads(path.read_text())
        if (config['training']['purpose'] != 'qualification' or config['steps'] != 4
                or config['dataset']['manifest_sha256'] !=
                'da627b1968dbb353acbf3b630a9feb84cf30080957a5660c8bc4125917fbe0ae'):
            raise ValueError('Expected unchanged execution-fixture qualification')
        snapshot = freeze(ROOT, ROOT / 'research/recipes/strong19_train', path,
                          ROOT / '.gozero/snapshots')
        snapshots[name] = snapshot.name
        configs[name] = record

    result = dict(kind='real19_execution_preparation_snapshots', status='prepared',
                  created=time.time(), operator_sha256=sha(Path(__file__)),
                  supersedes_preparation_sha256=sha(previous_path),
                  evidence=evidence, configurations=configs, snapshots=snapshots,
                  recipe_sources_equal_to_real19_cpu_harness=recipe_sources,
                  recovery_sources_equal_to_four_host_transfer=recovery_sources,
                  tpu_jobs_launched=False,
                  scope='Qualification sources only. Numerical recipe matches the passed real19 CPU continuation harness; recovery matches the passed four-host transport case. No full-size TPU memory, timing or distributed continuation claim. The registered larger9 comparison retains accelerator ownership; scientific19 data and hyperparameters remain unregistered.')
    with output.open('xb') as stream:
        stream.write(canonical_json(result))
    output.chmod(0o444)
    print(json.dumps(dict(status='prepared', snapshots=snapshots, receipt_sha256=sha(output))))


if __name__ == '__main__':
    main()
