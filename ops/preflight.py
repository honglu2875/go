"""Read-only host inspection. Do not import JAX or claim accelerator devices."""

import glob
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time


def inspect_host():
    affinity = sorted(os.sched_getaffinity(0))
    topology = {}
    for cpu in affinity:
        root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        topology[str(cpu)] = {
            name: (root / name).read_text().strip()
            for name in ('physical_package_id', 'core_id', 'thread_siblings_list')
            if (root / name).is_file()
        }
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable'):
            memory[name + '_kib'] = int(value.strip().split()[0])
    versions = {}
    for name in ('jax', 'jaxlib', 'libtpu', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    devices = sorted(set(glob.glob('/dev/accel*') + glob.glob('/dev/accel/*') + glob.glob('/dev/vfio/*')))
    holders, unreadable = [], 0
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / 'fd').iterdir())
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            unreadable += 1
            continue
        matched = []
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target.startswith(('/dev/accel', '/dev/vfio')):
                matched.append(target)
        if matched:
            try:
                name = (process / 'comm').read_text().strip()
            except OSError:
                name = 'exited'
            holders.append({'pid': int(process.name), 'program': name, 'devices': sorted(set(matched))})
    python_candidates = glob.glob(str(Path.home() / '.local/share/uv/python/cpython-3.12-*/bin/python3.12'))
    tools = {name: shutil.which(name) for name in ('uv', 'pdsh', 'rustc', 'cargo', 'cmake', 'g++', 'katago', 'rsync')}
    tpu_environment = {key: os.environ[key] for key in (
        'TPU_WORKER_ID', 'TPU_WORKER_HOSTNAMES', 'TPU_ACCELERATOR_TYPE',
        'TPU_PROCESS_BOUNDS', 'TPU_CHIPS_PER_PROCESS_BOUNDS', 'TPU_VISIBLE_DEVICES',
        'TPU_WORKER_NAME', 'CLOUD_TPU_TASK_ID',
    ) if key in os.environ}
    disk = shutil.disk_usage(Path.home())
    return {
        'schema_version': 1, 'kind': 'host_preflight', 'hostname': socket.gethostname(),
        'unix_time': time.time(), 'python': platform.python_version(), 'kernel': platform.release(),
        'cpu_affinity': affinity, 'cpu_topology': topology, 'memory': memory,
        'disk_free_bytes': disk.free, 'system_packages': versions, 'tools': tools,
        'managed_python_312': python_candidates, 'device_paths': devices,
        'visible_device_holders': holders, 'unreadable_process_fd_tables': unreadable,
        'tpu_environment': tpu_environment,
    }


if __name__ == '__main__':
    print(json.dumps(inspect_host(), sort_keys=True))
