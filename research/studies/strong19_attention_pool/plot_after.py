"""Render audited paired curves after closure; never reads models or targets."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

STUDY=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR','/tmp/gozero-attention-pool-matplotlib')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration-sha256',required=True)
    parser.add_argument('--wait-seconds',type=int,default=54000);args=parser.parse_args()
    started=time.time();deadline=time.monotonic()+args.wait_seconds
    report=dict(kind='audited_attention_pool_curves',started=started,
        operator_sha256=sha(Path(__file__)),registration_sha256=args.registration_sha256)
    try:
        if sha(STUDY/'registration-001.json')!=args.registration_sha256:raise ValueError('Registration differs')
        result=STUDY/'sequence-001/result.json'
        while not result.exists():
            if time.monotonic()>=deadline:raise TimeoutError('Training pair has not closed')
            time.sleep(30)
        closed=read(result)
        if closed['status']!='passed' or closed['registration_sha256']!=args.registration_sha256:
            report.update(status='not_rendered',reason='Training pair did not pass; inspect sequence-001/result.json')
            return
        comparison=STUDY/'sequence-001/comparison.json'
        if sha(comparison)!=closed['comparison_sha256'] or read(comparison)['status']!='passed':
            raise ValueError('Comparison identity differs')
        data=STUDY/'curves.csv'
        with data.open() as f:rows=list(csv.DictReader(f))
        if len(rows)!=68:raise ValueError('Incomplete curve table')
        groups={}
        for arm in ('flat','attention'):
            for split in ('validation','train_probe'):
                selected=[r for r in rows if r['arm']==arm and r['split']==split]
                if [int(r['turn']) for r in selected]!=list(range(0,257,16)):
                    raise ValueError('Missing/duplicate curve points')
                groups[arm,split]=[{k:float(v) for k,v in r.items() if k not in ('arm','split')} for r in selected]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        colors={'flat':'#64748b','attention':'#007f7b'}
        for axis,metric,minimum,xkey,title in (
            (axes[0,0],'expert_kl',0,'turn','Policy KL: complete curve'),
            (axes[0,1],'expert_kl',64,'turn','Policy KL: later updates'),
            (axes[1,0],'value_mse',0,'turn','Value MSE: complete curve'),
            (axes[1,1],'expert_kl',16,'learning_seconds','Policy KL versus learning time')):
            for (arm,split),values in groups.items():
                if xkey=='learning_seconds' and split!='validation':continue
                points=[p for p in values if p['turn']>=minimum]
                x=[p[xkey]/(60 if xkey=='learning_seconds' else 1) for p in points]
                label=arm+(' validation' if split=='validation' else ' train probe')
                axis.plot(x,[p[metric] for p in points],color=colors[arm],
                    linestyle='-' if split=='validation' else '--',linewidth=1.8,label=label)
            axis.set_title(title,loc='left');axis.grid(alpha=.2)
            axis.set_xlabel('Learning minutes (excludes compilation/evaluation)' if xkey=='learning_seconds' else 'Accepted updates')
            axis.set_ylabel('KL (nats)' if metric=='expert_kl' else 'Signed-value MSE')
        axes[0,0].legend(frameon=False,fontsize=9)
        fig.suptitle('19×19 single-token pooling • 256 updates of a 512-update schedule\n'
                     'One paired seed; identical draws and fixed validation population',fontsize=13)
        outputs=[]
        for suffix in ('png','pdf'):
            path=STUDY/('curves.'+suffix)
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,dpi=180);outputs.append(dict(path=path.name,sha256=sha(path)))
        plt.close(fig)
        report.update(status='passed',curve_sha256=sha(data),comparison_sha256=sha(comparison),outputs=outputs,
            scope='Descriptive paired learning curves. No confidence interval or playing-strength inference.')
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['finished']=time.time()
        with (STUDY/'plot-completion-001.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
