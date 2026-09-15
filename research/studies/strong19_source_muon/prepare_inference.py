"""Freeze the already-qualified serving recipe against this trained checkpoint."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.snapshots import canonical_json, freeze, read_json, verify


def main():
    parent_id = '741a7c0f89b771e9281334d05dde68b8dc6fab81faa470d8fdbbab389095aeb3'
    training_id = '3bd93c2a8fb2ee67fa8c071ebd435b9325bb921eae3a218c14d8092ad2f9cb70'
    result_sha = '42de2b6578a054b979ca909e1018c47c0f4fc91752d2673c6c77478af9df24eb'
    recovery_sha = '0cab9d71e6aa105c7189bf0569cd573150698c8005ad2228d23ca5d3ebcad7e5'
    result = STUDY / 'harness-cpu-001/full/artifacts/result.json'
    recovery = STUDY / 'harness-result-001.json'
    output = STUDY / 'inference-cpu-plan-001.json'
    config_path = STUDY / 'inference-cpu-config-001.json'
    if output.exists() or config_path.exists():
        raise FileExistsError('Inference preparation already exists')
    if sha256(result) != result_sha or sha256(recovery) != recovery_sha:
        raise ValueError('Training evidence changed')
    if any(read_json(p)['status'] != 'passed' for p in (result, recovery)):
        raise ValueError('Training qualification did not pass')
    parent = ROOT / '.gozero/snapshots' / parent_id
    training = ROOT / '.gozero/snapshots' / training_id
    manifest, training_manifest = verify(parent), verify(training)
    modules = ('causal.py', 'compact.py', 'draft_mask.py', 'draft_model.py',
               'encoder.py', 'heads.py', 'inference.py', 'joint.py', 'katago.py',
               'observation_attention.py', 'policy_model.py', 'spatial_readout.py')
    for module in modules:
        if sha256(parent / manifest['recipe'] / module) != sha256(training / training_manifest['recipe'] / module):
            raise ValueError('Serving and training numerical sources differ')
    config = read_json(parent / 'resolved_config.json')
    config['training_snapshot'] = training_id
    config['training_result'] = dict(path=str(result.relative_to(ROOT)), sha256=result_sha)
    with config_path.open('xb') as stream:
        stream.write(canonical_json(config))
    config_path.chmod(0o444)
    staging = Path(tempfile.mkdtemp(prefix='source-muon-inference-', dir=ROOT / '.gozero'))
    try:
        for name in manifest['files']:
            if name == 'resolved_config.json':
                continue
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(parent / name, destination)
        snapshot = freeze(staging, Path(manifest['recipe']), config_path, ROOT / '.gozero/snapshots')
        frozen = verify(snapshot)
        changed = [name for name in manifest['files'] if manifest['files'][name] != frozen['files'].get(name)]
        if changed != ['resolved_config.json'] or set(manifest['files']) != set(frozen['files']):
            raise ValueError('Serving snapshot changed beyond checkpoint configuration')
    finally:
        shutil.rmtree(staging)
    plan = dict(kind='joint_source_muon_trained_inference_cpu_plan', created=time.time(),
                operator_sha256=sha256(Path(__file__)), parent_inference_snapshot=parent_id,
                training_snapshot=training_id, training_result_sha256=result_sha,
                recovery_receipt_sha256=recovery_sha, snapshot_id=snapshot.name,
                numerical_modules_unchanged=list(modules),
                scope='Trained combined source-optimizer checkpoint export, cached policy/value inference and native Rust search. CPU execution qualification only.')
    with output.open('xb') as stream:
        stream.write(canonical_json(plan))
    output.chmod(0o444)
    print(json.dumps(dict(status='prepared', snapshot=snapshot.name, plan_sha256=sha256(output))))


if __name__ == '__main__':
    main()
