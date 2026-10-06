"""Archive completed comparison evidence on two private disk peers."""
import json
from pathlib import Path

from execute_run import ROOT, STUDY, read, require, sha
from gozero.disk_mirror import inventory, publish
from gozero.durable_files import atomic_json
from gozero.pod import load_hosts


def main():
    result = read(STUDY / 'sequence-001/result.json')
    require(result['status'] == 'passed' and read(STUDY / 'pod-release-001.json')['status'] == 'passed',
            'Sequence or release inspection is incomplete')
    require(sha(STUDY / 'sequence-001/comparison.json') == result['comparison_sha256'], 'Comparison changed')
    suffixes = {'.py', '.md', '.json', '.jsonl', '.log', '.csv'}
    paths = {p for p in STUDY.rglob('*') if p.is_file() and p.suffix in suffixes
             and not p.name.startswith('closure-backup')}
    for arm in result['arms'].values():
        attempt = ROOT / 'runs' / arm['attempt']
        require(read(attempt / 'result.json')['status'] == 'passed', 'Attempt not closed')
        paths.update(p for p in attempt.rglob('*') if p.is_file() and p.suffix in suffixes)
    paths.update(ROOT / name for name in ('ops/STATUS.md', 'research/README.md'))
    files = inventory(ROOT, sorted(str(p.relative_to(ROOT)) for p in paths))
    plan = dict(files=files, result_sha256=sha(STUDY / 'sequence-001/result.json'), operator_sha256=sha(Path(__file__)))
    atomic_json(STUDY / 'closure-backup-plan-001.json', plan, replace=False)
    hosts = {h.rank: h for h in load_hosts(ROOT / 'ops/hosts.json')}; copies = []
    for rank in (1, 3):
        copies.append(dict(peer_rank=rank, **publish(ROOT,
            ROOT / '.gozero/retained-source-archives/20261005-moe-comparison-closure-001',
            files=files, peer=hosts[rank].ssh, python='python3', floor_bytes=8 << 30)))
    atomic_json(STUDY / 'closure-backups-001.json', dict(status='passed', copies=copies, **plan), replace=False)
    print(json.dumps(dict(status='passed', files=len(files), bytes=sum(r['bytes'] for r in files.values()), verified_disk_peers=2)))


if __name__ == '__main__': main()
