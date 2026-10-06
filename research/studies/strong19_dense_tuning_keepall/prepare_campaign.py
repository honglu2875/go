"""Pin the bounded adaptive policy, validated operators and source before use."""
import time
from pathlib import Path
from execute_run import ROOT,STUDY,read,require,sha,publish


def main():
    cpu=read(STUDY/'batch-cpu-qualification-002.json')
    decisions=read(STUDY/'campaign-cpu-qualification-002.json')
    storage=read(STUDY/'storage-qualification-001.json')
    require(cpu['status']==decisions['status']==storage['status']=='passed','Qualification incomplete')
    for name,digest in cpu['source_files'].items():require(sha(ROOT/name)==digest,'Batch source changed')
    for name,digest in decisions['operators'].items():require(sha(ROOT/name)==digest,'Qualified operator changed: '+name)
    for name,digest in storage['operators'].items():require(sha(ROOT/name)==digest,'Copy-only storage operator changed')
    for name,digest in storage['sources'].items():require(sha(ROOT/name)==digest,'RAM LR source changed')
    initial=ROOT/'research/studies/strong19_dense_lr';first=initial/'pair-registration-001.json'
    require(sha(first)=='34943d80ca02eadf7af1f27b9b3025ff94c553741f7483f51200220b66d72309','Initial grid changed')
    original=read(initial/'source-qualification-001.json')
    for name,digest in original['sources'].items():require(sha(ROOT/name)==digest,'LR computation source changed')
    control_path=ROOT/'research/studies/strong19_recovery/flat-stage-003/audit.json';control=read(control_path)
    inputs={**cpu['source_files'],**original['sources'],**storage['sources']}
    paths=[*STUDY.glob('*.py'),STUDY/'PROTOCOL.md',STUDY/'README.md',
           STUDY/'batch-cpu-qualification-002.json',STUDY/'campaign-cpu-qualification-002.json',
           STUDY/'storage-qualification-001.json',STUDY/'auditor-reference-qualification-001.json',
           *STUDY.glob('*-template-001.json'),control_path,first,initial/'registration-backups-001.json',
           initial/'source-qualification-001.json',initial/'lr06-config-001.json',
           ROOT/'research/studies/strong19_recovery/draw-replay-001.json']
    for recipe in ('strong19_dense_lr_keepall','strong19_dense_batch_keepall'):
        paths += [ROOT/'research/recipes'/recipe/n for n in ('recipe.json','README.md')]
    inputs.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    registration=dict(status='prepared',kind='bounded_dense_lr_batch_campaign',created=time.time(),
        initial_registration_sha256=sha(first),dataset_manifest_sha256=control['dataset_manifest_sha256'],
        initial_parameters_sha256=control['initial_parameters_sha256'],
        control=dict(attempt='pod-20260928T030525Z-44bb05aa',audit=str(control_path.relative_to(ROOT)),
                     audit_sha256=sha(control_path),peak_lr=.001,batch_games=128,eligible_retirement=False),
        inputs=inputs,max_additional_lr_probes=2,max_campaign_hours=30,batch_games=[64,128,256],
        confirmation_updates=256,screen_equivalent_updates=128,max_new_ram_endpoints=5,selected_disk_slots_per_peer=2,existing_checkpoint_deletion_allowed=False,production_defaults_changed=False,
        scope='Wait for the initial fixed grid; use pinned validation decisions to refine LR, confirm at a later horizon, '
              'and qualify/run matched-exposure physical batch comparisons. New trials use RAM; selected states get two disk copies. No prior checkpoint is deleted.')
    path=STUDY/'registration-001.json';publish(path,registration)
    print(__import__('json').dumps(dict(status='prepared',sha256=sha(path),inputs=len(inputs))))


if __name__=='__main__':main()
