"""Exercise the new RAM transport with tiny disposable files, never real state."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from execute_run import ROOT,STUDY,publish,sha,require
from gozero.pod import load_hosts,SSH_OPTIONS
from retention import BASE,REMOTE,remote


def main():
    hosts={h.rank:h for h in load_hosts(ROOT/'ops/hosts.json')}
    name='tuning-transport-qualification-20261005-001'
    relative=Path(name)/'rank-0/artifacts/checkpoints/turn-000000001';target=BASE/relative
    library=ROOT/'.gozero/snapshots/3faa348db01fa08a0172f35f5415597bc90f8bf921356ff7608fefe3b25b4cc8/packages/gozero/src'
    checks=[]
    with tempfile.TemporaryDirectory(prefix='dense-tuning-transport-') as td:
        path=Path(td)/'turn-000000001';path.mkdir()
        for member in ('manifest.json','state.json','actors.json','arrays.npz'):
            (path/member).write_bytes((member+'\n').encode()*16);(path/member).chmod(0o444)
        path.with_suffix('.group.json').write_text('{}\n');path.with_suffix('.group.json').chmod(0o444)
        files={n:dict(bytes=(path/n).stat().st_size,sha256=sha(path/n)) for n in ('manifest.json','state.json','actors.json','arrays.npz')}
        files['group.json']=dict(bytes=3,sha256=sha(path.with_suffix('.group.json')))
        settings=dict(path=str(target),files=files,bytes=sum(x['bytes'] for x in files.values()),library=str(library))
        for rank in (1,3):
            host=hosts[rank];remote(host,REMOTE,dict(settings,action='prepare'))
            subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),str(path)+'/',host.ssh+':'+str(target)+'/'],check=True,timeout=60)
            subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),str(path.with_suffix('.group.json')),
                            host.ssh+':'+str(target.with_suffix('.group.json'))],check=True,timeout=60)
            remote(host,REMOTE,dict(settings,action='seal'));remote(host,REMOTE,dict(settings,action='verify'))
            altered=json.loads(json.dumps(settings));altered['files']['arrays.npz']['sha256']='0'*64
            try:remote(host,REMOTE,dict(altered,action='verify'))
            except subprocess.CalledProcessError:pass
            else:raise AssertionError('Changed expected bytes accepted')
            restored=Path(td)/f'restored-{rank}'
            subprocess.run(['rsync','-a','-e',shlex.join(['ssh',*SSH_OPTIONS]),host.ssh+':'+str(target/'arrays.npz'),str(restored)],check=True,timeout=60)
            require(sha(restored)==files['arrays.npz']['sha256'],'Round-trip bytes differ')
            code="import json,sys;from pathlib import Path;c=json.load(sys.stdin);p=Path(c['path']);assert p.parts[-5:] == ('rank-0','artifacts','checkpoints','turn-000000001')[-5:] if False else p.name=='turn-000000001';assert p.parents[3].name=='tuning-transport-qualification-20261005-001';[q.unlink() for q in p.iterdir()];p.with_suffix('.group.json').unlink();p.rmdir();print(json.dumps(dict(status='passed')))"
            remote(host,code,dict(path=str(target)))
            checks.append(dict(peer_rank=rank,prepare_seal_verify_roundtrip=True,rejected_wrong_hash=True,fixture_payloads_removed=True))
    result=dict(status='passed',created=time.time(),checks=checks,retention_operator_sha256=sha(STUDY/'retention.py'),
                qualifier_sha256=sha(Path(__file__)),scope='Tiny labelled fixture bytes only. Prior and active checkpoint paths were never used.')
    publish(STUDY/'retention-transport-qualification-001.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
