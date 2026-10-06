"""Bounded four-host training, disk-only restore, and exact continuation audit.

Only the explicitly registered tiny qualification attempts may be erased.
This verifies a stable current host/rank mapping, not topology migration.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).parent
sys.path.insert(0,str(ROOT/'packages/gozero/src'))
from gozero import checkpoints
from gozero.durable_files import atomic_json,sha256
from gozero.pod import load_hosts,SSH_OPTIONS
from gozero.snapshots import verify


def main():
    plan=json.loads((STUDY/'pod-recovery-plan-002.json').read_text())
    assert plan['operator_sha256']==sha256(Path(__file__))
    cpu=STUDY/'harness-qualification-001.json'
    assert sha256(cpu)==plan['cpu_qualification_sha256'] and json.loads(cpu.read_text())['status']=='passed'
    snapshot=ROOT/'.gozero/snapshots'/plan['snapshot'];verify(snapshot)
    config=json.loads((snapshot/'resolved_config.json').read_text())
    assert config['training']['purpose']=='qualification' and config['steps']==4 and config['model']['width']==16
    assert config['checkpoint_disk']['peer']==2
    python=plan['python'];hosts=load_hosts(snapshot/'ops/hosts.json');started=time.time()
    result=dict(status='running',attempts={},snapshot=snapshot.name,created=started,operator_sha256=sha256(Path(__file__)))
    def launch(label,stop=None,resume=None):
        argv=[str(ROOT/'.venv/bin/python'),'-B',str(snapshot/'ops/pod_run.py'),'--snapshot',str(snapshot),
            '--workspace-root',str(ROOT),'--timeout','900','--prepare-timeout','600','--controller-cpus','6',
            '--controller-cpu-list','2,3,4,5,6,7']
        if stop is not None:argv+=['--stop-after-turn',str(stop)]
        if resume is not None:argv+=['--resume-attempt',resume,'--resume-turn','2']
        log=STUDY/f'pod-recovery-{label}-002.log'
        print(json.dumps(dict(stage=label,status='starting')),flush=True)
        with log.open('xb') as f:
            subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        header=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{') and json.loads(x).get('kind')=='pod_attempt']
        assert len(header)==1
        attempt=Path(header[0]['attempt']);assert attempt.parent==ROOT/'runs'
        closed=json.loads((attempt/'result.json').read_text());assert closed['status']=='passed'
        reports=[json.loads((attempt/f'rank-{h}/artifacts/result.json').read_text()) for h in range(4)]
        assert all(r['status']=='passed' and r['turn']==(stop or 4) and r['latest_checkpoint']['disk']['all_rank_states'] for r in reports)
        result['attempts'][label]=dict(attempt=attempt.name,closure_sha256=sha256(attempt/'result.json'))
        atomic_json(STUDY/'pod-recovery-progress-002.json',result)
        print(json.dumps(dict(stage=label,status='passed',attempt=attempt.name)),flush=True)
        return attempt,reports
    try:
        previous=json.loads((STUDY/'pod-recovery-qualification-001.json').read_text())
        assert sha256(STUDY/'pod-recovery-qualification-001.json')==plan['predecessor_sha256']
        assert previous['status']=='failed' and previous['snapshot']==snapshot.name
        def completed(label,turn):
            item=previous['attempts'][label];attempt=ROOT/'runs'/item['attempt']
            assert sha256(attempt/'result.json')==item['closure_sha256']
            assert json.loads((attempt/'result.json').read_text())['status']=='passed'
            reports=[json.loads((attempt/f'rank-{h}/artifacts/result.json').read_text()) for h in range(4)]
            assert all(r['status']=='passed' and r['snapshot_id']==snapshot.name and r['turn']==turn for r in reports)
            result['attempts'][label]=item
            return attempt,reports
        full,full_reports=completed('full',4)
        prefix,prefix_reports=completed('prefix',2)
        result['reused_completed_attempts']=True
        result['predecessor_sha256']=plan['predecessor_sha256']
        cp=prefix_reports[0]['latest_checkpoint'];mirror=cp['disk'];assert mirror['status']=='passed'
        bundle=Path(mirror['target'])
        # Remove the disposable input on all hosts and every primary checkpoint
        # copy for this qualification prefix. Disk peer bundles remain untouched.
        code=r'''import json,shutil,sys
from pathlib import Path
p=json.load(sys.stdin);attempt=Path(p['attempt']);root=Path(p['root'])
assert attempt.parent==root/'runs' and attempt.name==p['id']
assert json.loads((attempt/f"rank-{p['host']}/start.json").read_text())['snapshot_id']==p['snapshot']
for h in range(4):
 cp=attempt/f'rank-{h}/artifacts/checkpoints/turn-000000002'
 if cp.exists():shutil.rmtree(cp)
 cp.with_suffix('.group.json').unlink(missing_ok=True)
fixture=Path(p['fixture']);assert fixture==Path('/dev/shm/gozero-datasets/recovery-fixture-20260925-001')
# Staging seals the files to root ownership; deleting user-owned directories
# still works because the directory itself retains the producer's ownership.
if fixture.exists():shutil.rmtree(fixture)
assert not fixture.exists()
print(json.dumps({'status':'erased','host':p['host']}))
'''
        def erase(host):
            payload=dict(attempt=str(prefix),root=str(ROOT),id=prefix.name,snapshot=snapshot.name,fixture=config['dataset']['path'],host=host.rank)
            r=subprocess.run(['ssh',*SSH_OPTIONS,host.ssh,shlex.join([python,'-B','-c',code])],input=json.dumps(payload),
                text=True,capture_output=True,timeout=60)
            (STUDY/f'erase-host-{host.rank}-002.log').write_text(r.stdout+r.stderr)
            r.check_returncode()
            return json.loads(r.stdout)
        with ThreadPoolExecutor(4) as pool:result['erased']=list(pool.map(erase,hosts))
        # Retrieve the only remaining prefix arrays from a different host's disk.
        with tempfile.TemporaryDirectory(prefix='pod-restore-',dir=ROOT/'.gozero') as name:
            staging=Path(name)
            subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),hosts[2].ssh+':'+str(bundle)+'/',str(staging)+'/'],check=True,timeout=180)
            assert sha256(staging/'mirror.json')==mirror['receipt_sha256']
            receipt=json.loads((staging/'mirror.json').read_text())
            for name,r in receipt['files'].items():assert sha256(staging/name)==r['sha256'] and (staging/name).stat().st_size==r['bytes']
            for h in range(4):
                target=prefix/f'rank-{h}/artifacts/checkpoints/turn-000000002';target.mkdir()
                for name in ('manifest.json','arrays.npz','state.json','actors.json'):
                    shutil.copyfile(staging/f'host-{h}'/name,target/name)
                shutil.copyfile(staging/f'host-{h}/group.json',target.with_suffix('.group.json'))
        shutil.copytree(ROOT/'.gozero/recovery-fixture-20260925-001',config['dataset']['path'])
        resumed,resumed_reports=launch('resumed',resume=prefix.name)
        def metrics(attempt,h):
            ignored={'cumulative_learning_seconds','cumulative_sampling_seconds'}
            return [{k:v for k,v in json.loads(x).items() if k not in ignored}
                for x in (attempt/f'rank-{h}/artifacts/metrics.jsonl').read_text().splitlines()]
        for h,(a,b) in enumerate(zip(full_reports,resumed_reports)):
            assert a['host_jax_mapping']==b['host_jax_mapping']
            for key in ('counters','validation_history','training_probe_history','overfit_observations'):
                assert a[key]==b[key],key
            ca,cb=a['latest_checkpoint'],b['latest_checkpoint']
            assert ca['replicated_arrays_elements_sha256']==cb['replicated_arrays_elements_sha256']
            sa,_,_=checkpoints.read(Path(ca['path']),expected_manifest_sha256=ca['manifest_sha256'])
            sb,_,_=checkpoints.read(Path(cb['path']),expected_manifest_sha256=cb['manifest_sha256'])
            assert sa==sb,'Complete per-rank state differs'
            assert metrics(full,h)==metrics(prefix,h)+metrics(resumed,h),'Subsequent updates differ'
        result.update(status='passed',all_rank_states_exact=True,all_parameter_and_adam_arrays_exact=True,
            all_update_metrics_exact_except_timings=True,disk_peer_restore=True,all_primary_prefix_copies_removed=True,
            all_disposable_inputs_removed=True,host_jax_mapping=full_reports[0]['host_jax_mapping'])
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result.update(seconds=time.time()-started,scope='Tiny real four-host TPU full-versus-2+2 learning and peer disk restoration. Fixed current host/JAX mapping; topology migration remains guarded, not qualified. No architecture-quality or Go-strength claim.')
        atomic_json(STUDY/'pod-recovery-qualification-002.json',result,replace=False)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
