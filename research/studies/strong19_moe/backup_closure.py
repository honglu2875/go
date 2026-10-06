"""Mirror the closed October MoE evidence and editable sources to two disk peers."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.disk_mirror import inventory, publish
from gozero.durable_files import atomic_json, sha256
from gozero.pod import load_hosts


def main():
    closures = {}
    for name in (
        'strong19_moe/balance010-001-stage/result.json',
        'strong19_moe_permute/kernel-stage-003/result.json',
        'strong19_moe_temporal/system-stage-001/result.json',
    ):
        path = ROOT / 'research/studies' / name
        value = json.loads(path.read_text())
        if value['status'] not in ('passed', 'failed'):
            raise ValueError('Experiment is not closed: ' + name)
        closures[name] = dict(status=value['status'], sha256=sha256(path))
    if not (STUDY / 'ROUND_20261004.md').is_file():
        raise ValueError('Round report is missing')
    release = json.loads((STUDY / 'pod-release-001.json').read_text())
    if release['status'] != 'passed':
        raise ValueError('Final resource inspection did not pass')
    suffixes = {'.py', '.md', '.json', '.jsonl', '.log'}
    paths = set()
    for name in ('strong19_moe', 'strong19_moe_permute', 'strong19_moe_temporal', 'strong19_moe_remat'):
        for section in ('research/studies', 'research/recipes'):
            paths.update(p for p in (ROOT / section / name).rglob('*')
                         if p.is_file() and p.suffix in suffixes
                         and not p.name.startswith('closure-backup'))
    for name in ('pod-20261004T143934Z-cd561c20', 'pod-20261004T175313Z-94596b43',
                 'pod-20261004T180503Z-f8f8edec'):
        folder = ROOT / 'runs' / name
        if not (folder / 'result.json').is_file():
            raise ValueError('Attempt remains open: ' + name)
        paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in suffixes)
    paths.update(ROOT / name for name in (
        'ops/STATUS.md', 'research/README.md', 'packages/gozero/src/gozero/moe.py'))
    files = inventory(ROOT, sorted(str(p.relative_to(ROOT)) for p in paths))
    plan = dict(files=files, closures=closures, operator_sha256=sha256(Path(__file__)))
    atomic_json(STUDY / 'closure-backup-plan-001.json', plan, replace=False)
    hosts = {h.rank: h for h in load_hosts(ROOT / 'ops/hosts.json')}
    copies = []
    target = ROOT / '.gozero/retained-source-archives/20261004-moe-closure-001'
    for rank in (1, 3):
        copies.append(dict(peer_rank=rank, **publish(
            ROOT, target, files=files, peer=hosts[rank].ssh,
            python='python3', floor_bytes=8 << 30)))
    atomic_json(STUDY / 'closure-backups-001.json',
                dict(status='passed', copies=copies, **plan), replace=False)
    print(json.dumps(dict(status='passed', files=len(files),
                          bytes=sum(x['bytes'] for x in files.values()),
                          verified_disk_peers=len(copies))))


if __name__ == '__main__':
    main()
