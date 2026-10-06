"""Keep the registered source/evidence closure on two checked disk peers."""
import json
from pathlib import Path

from execute_run import ROOT, STUDY, read, sha
from gozero.disk_mirror import inventory, publish
from gozero.durable_files import atomic_json
from gozero.pod import load_hosts


def main():
    registration = STUDY / 'pair-registration-001.json'; plan = read(registration)
    paths = {registration, *[ROOT / p for p in plan['operators']], *[ROOT / p for p in plan['prerequisites']]}
    for arm in plan['arms'].values():
        snapshot = ROOT / '.gozero/snapshots' / arm['snapshot']
        paths.update(p for p in snapshot.rglob('*') if p.is_file())
    paths.update(p for p in STUDY.iterdir() if p.is_file() and p.suffix in ('.json', '.py', '.md')
                 and not p.name.startswith(('registration-backup', 'pair-launch')))
    files = inventory(ROOT, sorted(str(p.relative_to(ROOT)) for p in paths))
    hosts = {h.rank: h for h in load_hosts(ROOT / 'ops/hosts.json')}
    copies = []
    for rank in (1, 3):
        copies.append(dict(peer_rank=rank, **publish(ROOT,
            ROOT / '.gozero/retained-source-archives/20261005-moe-comparison-001',
            files=files, peer=hosts[rank].ssh, python='python3', floor_bytes=8 << 30)))
    result = dict(status='passed', registration_sha256=sha(registration), operator_sha256=sha(Path(__file__)),
        files=files, copies=copies, scope='Private source, frozen snapshots and registration evidence; no checkpoint duplication.')
    atomic_json(STUDY / 'registration-backups-001.json', result, replace=False)
    print(json.dumps(dict(status='passed', files=len(files), bytes=sum(r['bytes'] for r in files.values()), copies=2)))


if __name__ == '__main__': main()
