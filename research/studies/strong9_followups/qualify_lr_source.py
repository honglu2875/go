"""Exercise source cloning and reject confounded interventions without learning."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time

import lr_source

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
PARENT = '23c9dfe60c0a4c1c376e74f634b568fd1c85f3b13a0c7fef9172859ad38a40fb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    parent = ROOT/'.gozero/snapshots'/PARENT
    original = lr_source.read_json(parent/'resolved_config.json')
    candidate = lr_source.configuration(original,.0015)
    rejected = []
    changes = [
        ('horizon',('steps',),1024),
        ('seed',('seed',),original['seed']+1),
        ('data',('dataset','manifest_sha256'),'0'*64),
        ('sampling',('dataset','bucket_probabilities'),[.5,.5]),
        ('encoder',('model','encoder_blocks'),original['model']['encoder_blocks']+1),
        ('normalization',('model','norm_epsilon'),1e-5),
        ('warmup',('learner','warmup_steps'),128),
        ('optimizer',('learner','beta2'),.999),
        ('decay',('learner','weight_decay'),.02),
        ('end_ratio',('learner','end_learning_rate'),.0003),
        ('evaluation',('eval_every',),512),
        ('test_targets',('evaluation','run_test'),True),
        ('nonfinite_rate',('learner','learning_rate'),float('nan')),
    ]
    for label,path,value in changes:
        changed=copy.deepcopy(candidate);target=changed
        for key in path[:-1]:target=target[key]
        target[path[-1]]=value
        try:lr_source.rate_only(original,changed)
        except ValueError:rejected.append(label)
        else:raise ValueError('Confounded intervention accepted: '+label)
    with tempfile.TemporaryDirectory(prefix='strong9-lr-source-check-',dir='/tmp') as folder:
        folder=Path(folder)
        unchanged=lr_source.freeze(parent,Path(lr_source.verify(parent)['recipe']),parent/'resolved_config.json',folder/'store')
        if unchanged.name!=PARENT:
            raise ValueError('Re-freezing unchanged source/config does not reproduce its identity')
        child=lr_source.clone(parent,.0015,folder/'store')
        files=lr_source.identical_source(parent,child)
        source_example=child/'research/recipes/strong9_policy/learner.py'
        if not source_example.exists():
            # Select a numerical source whose provenance is already qualified.
            source_example=child/'research/recipes/strong9_policy/causal.py'
        raw=source_example.read_bytes();source_example.chmod(0o644)
        source_example.write_bytes(raw+b'\n# qualification mutation\n')
        try:lr_source.identical_source(parent,child)
        except ValueError:rejected.append('source_mutation')
        else:raise ValueError('Mutated frozen numerical source accepted')
        # Restore this disposable fixture so the temporary directory has valid
        # evidence for its final verify before automatic removal.
        source_example.write_bytes(raw);source_example.chmod(0o444)
        lr_source.verify(child)
        child_id=child.name
    report=dict(kind='strong9_lr_source_qualification',status='passed',created=time.time(),
                parent_snapshot=PARENT,temporary_fixture_snapshot=child_id,
                unchanged_snapshot_reproduced=True,identical_source_files=files,
                rejected_confounders=rejected,operator_sha256=sha(Path(__file__)),
                source_operator_sha256=sha(STUDY/'lr_source.py'),seconds=time.monotonic()-started,
                scope='Source/config cloning and intervention-isolation checks only. The temporary 1.5e-3 fixture was removed; no research rate was selected, no registration created and no training launched.')
    output=STUDY/'lr-source-cpu-001.json'
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    output.chmod(0o444)
    print(json.dumps(dict(status='passed',source_files=files,rejected=len(rejected),sha256=sha(output))),flush=True)


if __name__=='__main__':main()
