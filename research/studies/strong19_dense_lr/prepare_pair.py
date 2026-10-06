"""Register the two fresh arms of the current dense LR grid."""
import math
import time
from pathlib import Path

from execute_run import ROOT, STUDY, publish, read, require, sha
from monitor import resource_observation
from gozero.snapshots import freeze, verify


def main():
    qualification = read(STUDY / 'source-qualification-001.json')
    require(qualification['status'] == 'passed', 'Numerical/source gate failed')
    for name, digest in qualification['sources'].items(): require(sha(ROOT / name) == digest, 'Source changed: ' + name)
    require(qualification['operator_sha256'] == sha(STUDY / 'qualify_source.py'), 'Qualifier changed')
    require(read(STUDY/'snapshot-dedup-result-001.json')['status']=='passed', 'Storage qualification incomplete')
    require(read(STUDY/'reporting-qualification-001.json')['status']=='passed'
            and read(STUDY/'reporting-qualification-001.json')['report_operator_sha256']==sha(STUDY/'compare.py'),
            'Reporting gate failed')
    base_path = ROOT / 'research/studies/strong19_recovery/flat-config-001.json'
    base = read(base_path); reference_path = ROOT / 'research/studies/strong19_recovery/flat-stage-003/audit.json'
    reference_audit = read(reference_path); attempt = 'pod-20260928T030525Z-44bb05aa'
    require(reference_audit['status']=='passed' and reference_audit['parameters']==232011540, 'Invalid dense reference')
    reference = dict(attempt=attempt,audit=str(reference_path.relative_to(ROOT)),audit_sha256=sha(reference_path),peak_learning_rate=.001)
    old_metrics = [__import__('json').loads(line) for line in (ROOT/'runs'/attempt/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()]
    metrics = {k:v for k,v in old_metrics[0].items() if k not in ('turn','bucket','local_entries_sha256','local_symmetries',
                                                               'cumulative_learning_seconds','cumulative_sampling_seconds')}
    arms = {}; reserve = math.ceil((232011540*12+4)*1.01)+(32<<20)
    for name in ('lr06','lr15'):
        cfg = STUDY / (name+'-config-001.json'); require(sha(cfg)==qualification['configs'][name], 'Configuration changed')
        c = read(cfg); factor = c['learner']['learning_rate']/.001
        snapshot = freeze(ROOT,Path('research/recipes/strong19_dense_lr'),cfg,ROOT/'.gozero/snapshots');verify(snapshot)
        first={k:(v*factor if k=='learning_rate' or k.startswith('update_norm_') else v) for k,v in metrics.items()}
        arms[name]=dict(snapshot=snapshot.name,config_sha256=sha(snapshot/'resolved_config.json'),parameters=232011540,
            peak_learning_rate=c['learner']['learning_rate'],replica_peer=2,timeout_seconds=16000,
            output_directory=name+'-stage-001',first_update_metrics=first)
    minima = {0:(2<<30)+2*reserve,1:8<<30,2:(8<<30)+2*reserve,3:8<<30}
    resources = resource_observation(ROOT/'.gozero/snapshots'/arms['lr06']['snapshot'])
    for x in resources:
        require(x['disk_free']>minima[x['rank']] and x['shm_free']>(64<<30) and x['memory_available']>(96<<30),
                'Insufficient total sequence capacity on rank '+str(x['rank']))
    prereqs = [STUDY/name for name in ('source-qualification-001.json','snapshot-dedup-result-001.json',
                                      'reporting-qualification-001.json','preflight-001.json')]
    prereqs += [reference_path,ROOT/'research/studies/strong19_recovery/draw-replay-001.json',
                ROOT/'research/studies/strong19_recovery/cohort-disk-backups-001.json',
                ROOT/'research/studies/strong19_moe_temporal/system-review-001.json',
                ROOT/'research/studies/strong19_moe_comparison/sequence-001/result.json']
    for p in prereqs: require(read(p)['status'] in ('passed','prepared'), 'Prerequisite failed: '+str(p))
    operators = [STUDY/name for name in ('PROTOCOL.md','prepare_pair.py','qualify_source.py','execute_pair.py',
        'execute_run.py','audit_run.py','replicate.py','compare.py','observation.py','monitor.py','backup_pair.py')]
    drawpath=ROOT/'research/studies/strong19_recovery/draw-replay-001.json';draw=read(drawpath)
    plan=dict(status='prepared',kind='dense_learning_rate_grid',created=time.time(),order=['lr06','lr15'],
        steps=128,schedule_steps=512,checkpoint_every=128,expected_positions=draw['draws'][127]['cumulative_positions'],
        dataset_manifest_sha256=base['dataset']['manifest_sha256'],validation_positions=64371,
        validation_population_sha256=reference_audit['validation_history'][0]['episode_ids_sha256'],
        probe_population_sha256=reference_audit['training_probe_history'][0]['episode_ids_sha256'],
        arms=arms,reference=reference,checkpoint_bytes_per_arm={name:reserve for name in arms},
        aggregate_disk_minimum_by_host={str(h):n for h,n in minima.items()},admission_resources=resources,
        draw_reference=str(drawpath.relative_to(ROOT)),draw_sha256=sha(drawpath),
        audit_python=str(ROOT/'.gozero/environments/a587255ebe0f78b1ec98bbb12ab1cb3315ec668424a08820c80da7353de40192/bin/python'),
        operators={str(p.relative_to(ROOT)):sha(p) for p in operators},
        prerequisites={str(p.relative_to(ROOT)):sha(p) for p in prereqs},
        scope='Initial LR bracket only. Two fresh 128-update arms; model/data/optimizer unchanged except complete LR scale. '
              'Endpoint disk checkpoint and peer. Subsequent adaptive LR/batch work requires its own prospective registration.')
    path=STUDY/'pair-registration-001.json';publish(path,plan)
    print(__import__('json').dumps(dict(status='prepared',sha256=sha(path),snapshots={a:i['snapshot'] for a,i in arms.items()},
        aggregate_minimum_gib={h:n/2**30 for h,n in minima.items()})))


if __name__=='__main__':main()
