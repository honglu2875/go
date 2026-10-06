"""Freeze one bounded temporal-expert systems screen after CPU qualification."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json,freeze


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prerequisites=[STUDY/n for n in ('cpu-qualification-002.json','budget-001.json')]
    prerequisites.append(ROOT/'research/studies/strong19_moe/kernel-qualification-004.json')
    for path in prerequisites:
        if json.loads(path.read_text())['status']!='passed':
            raise ValueError('Qualification did not pass: '+str(path))
    cpu=json.loads(prerequisites[0].read_text())
    for path,digest in cpu['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Qualified code changed: '+path)
    config=STUDY/'system-proposal-config-001.json'
    snapshot=freeze(ROOT,Path('research/recipes/strong19_moe_temporal'),config,ROOT/'.gozero/snapshots')
    operators=[STUDY/n for n in ('PLAN.md','prepare_system.py','execute_system.py','audit_qualification.py','review_system.py')]
    plan=dict(status='prepared',kind='moe_temporal_system_qualification_registration',
              created=time.time(),accelerator_job_started=False,snapshot=snapshot.name,
              config_sha256=sha(snapshot/'resolved_config.json'),
              must_follow_successful_attempt='pod-20261004T143934Z-cd561c20',
              timeout_seconds=2400,output_directory='system-stage-001',
              prerequisites={str(p.relative_to(ROOT)):sha(p) for p in prerequisites},
              operators={str(p.relative_to(ROOT)):sha(p) for p in operators},
              scope='Four fresh-state cases: dense and temporal-only experts at 512/768 positions. Two finite updates per case on pinned complete-game draws, memory guard, and all-rank metrics audit. No learned checkpoints or learning claim.')
    path=STUDY/'system-registration-001.json'
    with path.open('xb') as stream:stream.write(canonical_json(plan))
    path.chmod(0o444)
    print(json.dumps(dict(status='prepared',snapshot=snapshot.name,registration=str(path.relative_to(ROOT)),sha256=sha(path),accelerator_job_started=False)))


if __name__=='__main__':main()
