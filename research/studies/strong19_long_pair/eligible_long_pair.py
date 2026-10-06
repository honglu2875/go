"""Permit relocation only for six pinned, completed larger9 checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(attempt):
    plan = json.loads((STUDY / 'storage-plan-001.json').read_text())
    matches = [r for r in plan['cases'] if r['attempt'] == attempt]
    if len(matches) != 1:
        raise ValueError('Attempt is outside the six pinned closed checkpoints')
    row = matches[0]
    for key in ('audit', 'replica'):
        if sha(ROOT / row[key]) != row[key + '_sha256']:
            raise ValueError('Retention evidence changed')
    closed_path = ROOT / 'runs' / attempt / 'result.json'
    if sha(closed_path) != row['closure_sha256']:
        raise ValueError('Attempt closure changed')
    closed = json.loads(closed_path.read_text())
    audit = json.loads((ROOT / row['audit']).read_text())
    report = json.loads((closed_path.parent / 'rank-0/artifacts/result.json').read_text())
    if (closed['status'] != 'passed' or closed['snapshot_id'] != row['snapshot']
            or closed.get('resume_attempt') or audit['status'] != 'passed'
            or audit['attempt'] != attempt or audit['steps'] != row['turn']
            or report['turn'] != row['turn'] or report['status'] != 'passed'
            or report['latest_checkpoint']['replicated_arrays_elements_sha256'] !=
               audit['checkpoint_arrays_elements_sha256']):
        raise ValueError('Complete checkpoint audit does not match')
    return dict(turn=row['turn'], snapshot=row['snapshot'], audit=row['audit'],
                audit_sha256=row['audit_sha256'], plan_sha256=sha(STUDY / 'storage-plan-001.json'))
