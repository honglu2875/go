"""Reserve persistent space before learning; stage standard checkpoints in RAM.

Only this run's newly-created reservation file is consumed. Existing scientific
artifacts are untouched. Publication retains the standard four-file format.
"""
import os
from pathlib import Path
import shutil
import tempfile
from gozero import checkpoints


def reservation_size(schema):
    # FP32 parameters plus one FP32 momentum. Two percent and16MiB minimum
    # exceed deflate's worst-case expansion and all NPZ/NPY headers here.
    raw = 2 * sum(s['elements'] for s in schema) * 4
    return raw + max(16 * 2**20, (raw + 49)//50)


def reserve(path, size):
    with Path(path).open('xb') as f:
        os.posix_fallocate(f.fileno(), 0, size)
        f.flush(); os.fsync(f.fileno())
    return size


def write(path, *, reservation, state, arrays, scratch_root='/dev/shm'):
    path, reservation = Path(path), Path(reservation)
    if path.exists() or reservation.is_symlink() or not reservation.is_file():
        raise ValueError('Invalid new checkpoint or reserved file')
    path.parent.mkdir(parents=True, exist_ok=True)
    if reservation.stat().st_dev != path.parent.stat().st_dev:
        raise ValueError('Reservation must be on the checkpoint filesystem')
    temporary = None
    with tempfile.TemporaryDirectory(prefix='go-sgd-checkpoint-', dir=scratch_root) as scratch:
        staged = Path(scratch)/'checkpoint'
        identity = checkpoints.write(staged, state=state, arrays=arrays, actors='{}', compress=True)
        source = staged/'arrays.npz'
        if source.stat().st_size > reservation.stat().st_size:
            raise ValueError('Checkpoint exceeds its reserved serialization bound')
        # Move this run's allocation into its incomplete checkpoint directory.
        # Opening r+b, rather than wb, keeps the allocated blocks until the copy
        # has completed. The only truncation removes unused reservation tail.
        temporary = Path(tempfile.mkdtemp(prefix='.'+path.name+'.partial-', dir=path.parent))
        destination = temporary/'arrays.npz'
        os.rename(reservation, destination)
        with source.open('rb') as incoming, destination.open('r+b') as outgoing:
            shutil.copyfileobj(incoming, outgoing, 8*2**20)
            outgoing.truncate(); outgoing.flush(); os.fsync(outgoing.fileno())
        destination.chmod(0o444)
        for name in ('manifest.json', 'state.json', 'actors.json'):
            with (temporary/name).open('xb') as out:
                out.write((staged/name).read_bytes()); out.flush(); os.fsync(out.fileno()); os.fchmod(out.fileno(),0o444)
        if checkpoints.sha256(destination) != checkpoints.sha256(source):
            raise ValueError('Persistent checkpoint copy differs from the staged bytes')
        checkpoints._sync_directory(temporary)
        temporary.rename(path); checkpoints._sync_directory(path.parent)
        return identity
