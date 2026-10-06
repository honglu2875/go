"""Freeze two fixed MoE hypotheses and their systems qualification."""
import time
from pathlib import Path
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero.snapshots import freeze,verify
from campaign import RESERVE,PYTHON


def main():
    cpu=STUDY/'cpu-qualification-001.json';evidence=read(cpu);require(evidence['status']=='passed','CPU evidence incomplete')
    for name,digest in evidence['source_files'].items():require(sha(ROOT/name)==digest,'Qualified source changed')
    recipe=Path('research/recipes/strong19_moe_batch64');snapshots={}
    for arm in ('system','temporal','balance_low'):
        snap=freeze(ROOT,recipe,STUDY/(arm+'-config.json'),ROOT/'.gozero/snapshots');verify(snap);snapshots[arm]=snap
    control=ROOT/'research/studies/strong19_dense_tuning_keepall/batch64-stage/audit.json'
    closed=read(control.parent/'result.json');require(closed['status']=='passed' and closed['audit_sha256']==sha(control),'Dense baseline audit failed')
    base=read(control)
    references=[cpu,control,control.parent/'result.json',ROOT/'research/defaults/strong19.json',
        ROOT/'research/studies/strong19_recovery/draw-replay-001.json',*[ROOT/n for n in evidence['inherited']]]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [*STUDY.glob('*.py'),STUDY/'README.md',STUDY/'PROTOCOL.md',
        *STUDY.glob('*-config.json'),*references,*[p for p in (ROOT/recipe).iterdir() if p.is_file()]]}
    operators={str((STUDY/n).relative_to(ROOT)):sha(STUDY/n) for n in ('execute_run.py','audit_run.py','replicate.py','batch_audit.py','ram_copy.py')}
    arms={}
    for arm in ('temporal','balance_low'):
        snap=snapshots[arm]
        arms[arm]=dict(kind='registered_joint19_run',snapshot=snap.name,config_sha256=sha(snap/'resolved_config.json'),
            purpose='learning',steps=256,schedule_steps=1024,checkpoint_every=1024,parameters=317001492,timeout_seconds=14600,
            replica_peer=1,checkpoint_reserve_bytes=RESERVE,shm_floor_bytes=64<<30,memory_floor_bytes=96<<30,disk_floor_bytes=2<<30,
            producer_growth_reserve_bytes=0,disk_checkpoint_reserve_by_host={str(h):0 for h in range(4)},
            additional_reserve_by_host={str(h):RESERVE if h in (0,1,3) else 0 for h in range(4)},output_directory=arm+'-stage',
            audit_python=str(PYTHON),expected_positions=7001181,validation_positions=64371,operators=operators,
            prerequisites={str(cpu.relative_to(ROOT)):sha(cpu)},scope='Fixed fresh MoE arm with tuned dense settings and identical game exposure; no test access or file deletion.')
    registration=dict(status='prepared',kind='tuned_default_moe_campaign',created=time.time(),inputs=inputs,arms=arms,
        system_snapshot=snapshots['system'].name,max_campaign_hours=14,existing_checkpoint_deletion_allowed=False,
        dataset_manifest_sha256=base['dataset_manifest_sha256'],validation_population_sha256=base['validation_history'][0]['episode_ids_sha256'],
        probe_population_sha256=base['training_probe_history'][0]['episode_ids_sha256'],budgets=evidence['budgets'],
        dense_reference=dict(attempt=closed['attempt'],audit=str(control.relative_to(ROOT)),audit_sha256=sha(control)))
    path=STUDY/'registration-001.json';publish(path,registration)
    print(__import__('json').dumps(dict(status='prepared',registration_sha256=sha(path),snapshots={k:p.name for k,p in snapshots.items()})))

if __name__=='__main__':main()
