"""Physical-core allocations. Affinity is local to owned jobs, never system-wide."""
from __future__ import annotations

import os
import subprocess


def cpu_profile(generation_cores: int = 64, worker_cores: int = 8) -> dict:
    rows = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True)
    return allocate_cpus(rows, os.sched_getaffinity(0), generation_cores, worker_cores)


def allocate_cpus(rows: str, allowed, generation_cores: int, worker_cores: int) -> dict:
    cores = {}
    for line in rows.splitlines():
        if not line or line.startswith('#'):
            continue
        cpu, core, socket, node = map(int, line.split(','))
        if cpu in allowed:
            cores.setdefault((node, socket, core), []).append(cpu)
    nodes = sorted({key[0] for key in cores})
    if (not nodes or generation_cores <= 0 or worker_cores <= 0
            or generation_cores % (worker_cores * len(nodes))):
        raise ValueError('Generation cores must divide evenly into workers on NUMA nodes')
    per_node = generation_cores // len(nodes)
    selected = []
    workers = []
    for node in nodes:
        keys = sorted(key for key in cores if key[0] == node)
        if len(keys) <= per_node:
            raise ValueError('Leave physical cores available for research on each NUMA node')
        selected.extend(keys[:per_node])
        cpus = [min(cores[key]) for key in keys[:per_node]]
        workers.extend(cpus[i:i + worker_cores] for i in range(0, len(cpus), worker_cores))
    occupied = {cpu for key in selected for cpu in cores[key]}
    research_keys = sorted(set(cores) - set(selected))
    return dict(schema_version=1, physical_cores=len(cores), logical_cpus=sum(map(len, cores.values())),
                generation_physical_cores=generation_cores, workers=workers,
                generation_siblings=sorted(occupied),
                research_cpus=[min(cores[key]) for key in research_keys],
                research_logical_cpus=sorted(set(allowed) - occupied))


def pin(cpus):
    cpus = set(cpus)
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise ValueError('CPU allocation is not available to this process')
    os.sched_setaffinity(0, cpus)
