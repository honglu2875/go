"""Qualify new-file-only RAM staging, replication and selected disk publication."""
import json
import os
from pathlib import Path
import subprocess
import sys
from execute_run import ROOT,STUDY,read,sha,publish,require
from gozero import checkpoints,checkpoint_stage,ram_checkpoints
from gozero.snapshots import canonical_json
from promote import promote


def main():
    import numpy as np
    recipe=ROOT/'research/recipes/strong19_dense_lr_keepall';parent=ROOT/'research/recipes/strong19_dense_lr'
    for p in recipe.glob('*.py'):
        if p.name!='train_config.py':require(sha(p)==sha(parent/p.name),'LR numerical code changed: '+p.name)
    sys.path.insert(0,str(recipe));import train_config
    for rate in (.00025,.001,.004):
        c=read(ROOT/'research/studies/strong19_dense_lr/lr06-config-001.json');c.pop('checkpoint_disk')
        c.update(checkpoint_temporary=True,checkpoint_minimum_free_bytes=64<<30,checkpoint_every=512)
        c['learner'].update(learning_rate=rate,end_learning_rate=.3*rate);train_config.validate(c)
    name='storage-fixture-keepall-20261005-001';attempt=ROOT/'runs'/name;attempt.mkdir(exist_ok=False)
    snapshot='3faa348db01fa08a0172f35f5415597bc90f8bf921356ff7608fefe3b25b4cc8'
    arrays=dict(fixture=np.arange(64,dtype=np.float32));owner=None;paths={};manifests={}
    for host in range(4):
        artifact=attempt/f'rank-{host}/artifacts';artifact.mkdir(parents=True)
        logical=artifact/'checkpoints/turn-000000001';state=dict(kind='storage_transport_fixture',host_rank=host,snapshot_id=snapshot)
        if host==0:
            path,identity,receipt=checkpoint_stage.write(logical,state=state,arrays=arrays,actors='{}',minimum_free_bytes=64<<30)
            owner=path
        else:
            path=logical;identity=checkpoints.write(path,state=state,arrays={},actors='{}')
        paths[host]=path;manifests[str(host)]=identity
    group=dict(kind='storage_transport_fixture_group',snapshot_id=snapshot,host_manifests=manifests,
               owner_checkpoint_path=str(owner),claims_learning=False)
    for host,path in paths.items():
        gp=path.with_suffix('.group.json');publish(gp,group)
        if host==0:ram_checkpoints.seal([gp])
        cp=dict(path=str(path),manifest_sha256=manifests[str(host)],group_sha256=sha(gp),temporary=True)
        publish(attempt/f'rank-{host}/artifacts/result.json',dict(status='passed',kind='storage_transport_fixture',claims_learning=False,
            snapshot_id=snapshot,latest_checkpoint=cp))
    publish(attempt/'result.json',dict(status='passed',kind='storage_transport_fixture',claims_learning=False,snapshot_id=snapshot))
    state,restored,actors=checkpoints.read(owner,expected_manifest_sha256=manifests['0'])
    np.testing.assert_array_equal(restored['fixture'],arrays['fixture']);require(actors=='{}','Actor fixture changed')
    subprocess.run([sys.executable,'-B',str(STUDY/'replicate.py'),'--workspace-root',str(ROOT),'--attempt',name,'--peer','1',
        '--output',str(STUDY/'storage-fixture-ram-replicas.json')],check=True,timeout=600)
    audit=STUDY/'storage-fixture-audit.json';publish(audit,dict(status='passed',kind='tiny_storage_fixture',claims_learning=False))
    record=dict(attempt=name,audit=str(audit.relative_to(ROOT)),audit_sha256=sha(audit))
    result=promote(record,'storage-fixture')
    require(len(result['copies'])==2 and all(x['status']=='passed' for x in result['copies']),'Disk publication failed')
    require(owner.exists() and sha(owner/'manifest.json')==manifests['0'],'Publication removed or changed the source')
    answer=dict(status='passed',kind='new_file_only_storage_qualification',claims_learning=False,
        checks=['unchanged_LR_numerical_source','explicit_RAM_configuration_guards','temporary_stage_read_roundtrip',
                'two_verified_readonly_RAM_replicas','all_rank_metadata_in_two_fsynced_disk_copies','source_payload_preserved'],
        fixture=name,operators={str((STUDY/n).relative_to(ROOT)):sha(STUDY/n) for n in ('replicate.py','promote.py','ram_copy.py','qualify_storage.py')},
        sources={str(p.relative_to(ROOT)):sha(p) for p in recipe.glob('*.py')},
        scope='Tiny labelled new fixture files only; no accelerator use or real-checkpoint mutation/deletion.')
    publish(STUDY/'storage-qualification-001.json',answer);print(json.dumps(dict(status='passed',checks=answer['checks'])))


if __name__=='__main__':main()
