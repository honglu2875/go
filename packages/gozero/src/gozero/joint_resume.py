"""Collect complete four-host joint checkpoints, including RAM owner arrays."""
from pathlib import Path
import re

from .checkpoints import sha256
from .snapshots import read_json

RAM_BASES = (Path('/dev/shm/gozero-staged-checkpoints'), Path('/dev/shm/gozero-archived-checkpoints'))
MEMBERS = {'manifest.json', 'arrays.npz', 'state.json', 'actors.json'}
SHARED_KINDS = {'fixed_joint_learning'}


def regular(path, roots):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or not any(path.is_relative_to(root) for root in roots):
        raise ValueError('Checkpoint path escapes its declared storage roots')
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('Checkpoint requires regular files and parents')
    return path


def logical(root, attempt, host, turn):
    root = Path(root).resolve(); attempt = Path(attempt)
    if (attempt.parent != root / 'runs' or not re.fullmatch(r'pod-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}', attempt.name)
            or type(host) is not int or host not in range(4) or type(turn) is not int or not 0 < turn < 1_000_000_000):
        raise ValueError('Invalid replicated checkpoint location')
    return attempt / f'rank-{host}/artifacts/checkpoints/turn-{turn:09d}'


def owner_path(root, base, group):
    owner = Path(group['owner_checkpoint_path'])
    suffix = base.relative_to(Path(root).resolve() / 'runs')
    if owner not in (base, *(ram / suffix for ram in RAM_BASES)):
        raise ValueError('Owner arrays do not belong to this checkpoint')
    return owner


def resume_path(root, attempt, host, turn, snapshot):
    """Resolve the owner argument after the controller has staged the closure."""
    checkpoint = logical(root, attempt, host, turn)
    group_path = regular(checkpoint.with_suffix('.group.json'), (Path(root).resolve(),))
    group = read_json(group_path)
    if (group.get('kind') != 'visual_replicated_checkpoint_group' or group.get('snapshot_id') != snapshot
            or group.get('turn') != turn):
        raise ValueError('Resume group source/turn differs')
    owner = owner_path(root, logical(root, attempt, 0, turn), group)
    return owner if host == 0 else checkpoint


def collect(root, group_path, expected_sha256, *, snapshot):
    root = Path(root).resolve(); roots = (root, *RAM_BASES)
    group_path = regular(group_path, (root,))
    if sha256(group_path) != expected_sha256:
        raise ValueError('Checkpoint group identity differs')
    group = read_json(group_path)
    if (group.get('kind') != 'visual_replicated_checkpoint_group' or group.get('snapshot_id') != snapshot
            or set(group.get('host_manifests', {})) != {'0', '1', '2', '3'}):
        raise ValueError('Expected the pinned four-host joint checkpoint')
    attempt = group_path.parents[3]; turn = group['turn']
    base = logical(root, attempt, 0, turn)
    if group_path != base.with_suffix('.group.json'):
        raise ValueError('Group must be the logical owner checkpoint')
    mapping = group['host_jax_mapping']
    if (len(mapping) != 4 or {r['host'] for r in mapping} != set(range(4))
            or {r['jax_rank'] for r in mapping} != set(range(4))):
        raise ValueError('Invalid host/JAX mapping')
    owner = owner_path(root, base, group); files = {}; states = {}
    def add(path, expected=None):
        path = regular(path, roots); digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Checkpoint member identity differs: ' + str(path))
        files[path] = dict(sha256=digest, bytes=path.stat().st_size)
    add(group_path, expected_sha256)
    for host in range(4):
        directory = owner if host == 0 else logical(root, attempt, host, turn)
        manifest_path = directory / 'manifest.json'
        add(manifest_path, group['host_manifests'][str(host)])
        manifest = read_json(manifest_path)
        if set(manifest['files']) != MEMBERS - {'manifest.json'}:
            raise ValueError('Incomplete checkpoint payload')
        for name, record in manifest['files'].items():
            path = directory / name; add(path, record['sha256'])
            if files[path]['bytes'] != record['bytes']:
                raise ValueError('Checkpoint member length differs')
        state = read_json(directory / 'state.json'); states[host] = state
        if (state.get('snapshot_id') != snapshot or state.get('turn') != turn or state.get('host_rank') != host
                or state.get('config_sha256') != group['config_sha256']
                or state.get('jax_rank') != next(r['jax_rank'] for r in mapping if r['host'] == host)
                or state.get('owns_replicated_arrays') != (host == 0)):
            raise ValueError('Rank state does not belong to the checkpoint group')
        add(directory.with_suffix('.group.json'), expected_sha256)
    for state in states.values():
        if any(state[key] != states[0][key] for key in
               ('dataset_manifest_sha256', 'model_schema', 'optimizer_metadata', 'counters')):
            raise ValueError('Ranks disagree on shared scientific state')
    if owner != base:
        temporary = base.with_suffix('.temporary.json')
        archived = base.with_suffix('.archive.json')
        receipt_path = temporary if temporary.exists() else archived
        add(receipt_path); receipt = read_json(receipt_path)
        if (receipt.get('cache_path') != str(owner) or receipt.get('logical_path') != str(base)
                or receipt.get('manifest_sha256') != group['host_manifests']['0']):
            raise ValueError('RAM owner receipt differs')
    return files
