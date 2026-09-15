"""Host-local admission budgets for shared RAM storage; no automatic data deletion.

Every FlyGo job on a host uses the same root and limits. Reservations bound
additional peak file and heap allocations, conservatively retained until the
operation finishes. Live usage plus reservations may overestimate memory while
an operation is writing; this favors preserving headroom for other workloads.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import stat
import uuid


GIB = 1 << 30


class StoragePressure(RuntimeError):
    """Admission refused; retry after resources become available."""


@dataclass(frozen=True)
class Limits:
    files_cap: int = 100 * GIB
    free_files_floor: int = 64 * GIB
    available_memory_floor: int = 96 * GIB


def _process_identity(pid: int) -> str | None:
    try:
        # starttime survives pid reuse checks; comm itself may contain spaces.
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
    except FileNotFoundError:
        return None


def available_memory() -> int:
    memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemAvailable:"))
    # Common unified and legacy cgroup mounts, including a namespace-root limit.
    for base, limit_name, used_name in (
        (Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
        (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes"),
    ):
        limit_file, used_file = base / limit_name, base / used_name
        if limit_file.is_file() and used_file.is_file():
            limit = limit_file.read_text().strip()
            if limit != "max":
                memory = min(memory, max(0, int(limit) - int(used_file.read_text())))
    return memory


def allocated_bytes(root: Path) -> int:
    """Count allocated blocks once per inode; do not follow external symlinks."""
    seen = set()
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue
            key = info.st_dev, info.st_ino
            if stat.S_ISREG(info.st_mode) and key not in seen:
                seen.add(key)
                total += info.st_blocks * 512
    return total


class StorageBudget:
    def __init__(self, root: Path = Path("/dev/shm/gozero"), limits: Limits = Limits()):
        if any(value < 0 for value in asdict(limits).values()):
            raise ValueError("Storage limits must be nonnegative")
        self.root = Path(root).resolve()
        self.limits = limits
        self.control = self.root / "control"
        self.control.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _ledger(self):
        with (self.control / "storage.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.control / "storage.json"
            ledger = json.loads(path.read_text()) if path.exists() else {
                "schema_version": 1, "limits": asdict(self.limits), "reservations": {}}
            if ledger["limits"] != asdict(self.limits):
                raise ValueError("All jobs sharing a storage root must use identical limits")
            ledger["reservations"] = {key: value for key, value in ledger["reservations"].items()
                                      if _process_identity(value["pid"]) == value["process_identity"]}
            try:
                yield ledger
            finally:
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(ledger, indent=2) + "\n")
                temporary.replace(path)

    def _snapshot(self) -> dict:
        filesystem = os.statvfs(self.root)
        return dict(files_used=allocated_bytes(self.root),
                    files_free=filesystem.f_bavail * filesystem.f_frsize,
                    memory_available=available_memory())

    def _check(self, ledger: dict, files: int, heap: int) -> dict:
        snapshot = self._snapshot()
        pending_files = sum(r["files"] for r in ledger["reservations"].values()) + files
        pending_heap = sum(r["heap"] for r in ledger["reservations"].values()) + heap
        if snapshot["files_used"] + pending_files > self.limits.files_cap:
            raise StoragePressure("FlyGo storage cap would be exceeded")
        if snapshot["files_free"] - pending_files < self.limits.free_files_floor:
            raise StoragePressure("Shared filesystem headroom would be too small")
        if snapshot["memory_available"] - pending_files - pending_heap < self.limits.available_memory_floor:
            raise StoragePressure("Available RAM headroom would be too small")
        return {**snapshot, "reserved_files": pending_files, "reserved_heap": pending_heap}

    def check(self, *, files: int = 0, heap: int = 0) -> dict:
        """Recheck live pressure at safe boundaries; never evict another job's data."""
        if files < 0 or heap < 0:
            raise ValueError('Allocations must be nonnegative')
        with self._ledger() as ledger:
            return self._check(ledger, files, heap)

    @contextmanager
    def reserve(self, *, files: int, heap: int, purpose: str):
        if files < 0 or heap < 0:
            raise ValueError("Reservations must be nonnegative")
        token = uuid.uuid4().hex
        with self._ledger() as ledger:
            self._check(ledger, files, heap)
            ledger["reservations"][token] = dict(
                pid=os.getpid(), process_identity=_process_identity(os.getpid()),
                files=files, heap=heap, purpose=purpose)
        try:
            yield self
        finally:
            with self._ledger() as ledger:
                ledger["reservations"].pop(token, None)
