"""CPU-only deliberate rank crash, with an attempt-owned stubborn descendant."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero.snapshots import canonical_json,read_json,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(ROOT);c=read_json(args.config)
    if c!=read_json(ROOT/'resolved_config.json') or set(c)!=set('schema_version expected_hosts failure_rank failure_after_seconds maximum_wait_seconds cpus'.split()):
        raise ValueError('Invalid frozen fault configuration')
    if (c['schema_version']!=1 or c['expected_hosts']!=4 or c['failure_rank'] not in range(4)
            or not 1<=c['failure_after_seconds']<=5 or not 10<=c['maximum_wait_seconds']<=60):
        raise ValueError('Unbounded fault probe')
    if int(os.environ['GOZERO_WORLD_SIZE'])!=4:raise ValueError('Expected four hosts')
    if not c['cpus'] or not set(c['cpus'])<=os.sched_getaffinity(0):raise ValueError('CPU affinity unavailable')
    os.sched_setaffinity(0,c['cpus']);rank=int(os.environ['GOZERO_HOST_RANK'])
    args.output.mkdir(parents=True,exist_ok=False)
    # Inherits this trainer's process group; run_host owns its cleanup.
    script='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'
    child=subprocess.Popen([sys.executable,'-c',script])
    stat=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()
    record={'schema_version':1,'rank':rank,'snapshot_id':ROOT.name,'started_unix':time.time(),
            'descendant_pid':child.pid,'descendant_start_ticks':stat[19],'trainer_pid':os.getpid(),
            'declared_failure_rank':c['failure_rank'],'uses_tpu':False}
    (args.output/'probe.json').write_bytes(canonical_json(record));print(json.dumps(record),flush=True)
    time.sleep(c['failure_after_seconds'] if rank==c['failure_rank'] else c['maximum_wait_seconds'])
    raise SystemExit(23 if rank==c['failure_rank'] else 24)


if __name__=='__main__':main()
