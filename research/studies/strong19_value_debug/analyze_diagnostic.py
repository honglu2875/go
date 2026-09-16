"""Check diagnostic closure and summarize probabilities and analytic gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):
    with np.load(path,allow_pickle=False) as f:d=dict(f)
    live=np.arange(d['values'].shape[1])[None,:]<d['counts'][:,None]
    out={k:v[live] for k,v in d.items() if v.shape[:2]==live.shape}
    for k in ('split','opponent'):
        out[k]=np.broadcast_to(d[k][:,None],live.shape)[live]
    return out


def summary(logits,target):
    z=logits.astype(np.float64);p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    v=p[:,0]-p[:,1];error=v-target
    dv=p*(np.asarray([1.,-1.,0.])[None,:]-v[:,None])
    mse_gradient=2*error[:,None]*dv
    q=np.stack(((1+target)/2,(1-target)/2,np.zeros_like(target)),axis=-1)
    ce_gradient=p-q
    norm=lambda x:np.linalg.norm(x,axis=-1)
    quantiles=lambda x:np.quantile(x,[0,.1,.5,.9,1]).tolist()
    return dict(positions=len(v),value_mse=float(np.mean(error**2)),target_mean=float(target.mean()),
        prediction_mean=float(v.mean()),probability_mean=p.mean(0).tolist(),
        probability_quantiles=[quantiles(p[:,i]) for i in range(3)],
        value_quantiles=quantiles(v),neutral_over_99_fraction=float(np.mean(p[:,2]>.99)),
        mse_logit_gradient_norm_mean=float(norm(mse_gradient).mean()),
        ce_logit_gradient_norm_mean=float(norm(ce_gradient).mean()),
        mse_logit_gradient_norm_quantiles=quantiles(norm(mse_gradient)),
        ce_logit_gradient_norm_quantiles=quantiles(norm(ce_gradient)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    closure=json.loads((a.attempt/'result.json').read_text())
    if closure['status']!='passed':raise ValueError('Attempt did not close successfully')
    records=[];paths={'evaluation':[],'training':[]};reports=[]
    for rank in range(4):
        folder=a.attempt/f'rank-{rank}'
        process=json.loads((folder/'result.json').read_text())
        report=json.loads((folder/'artifacts/result.json').read_text());reports.append(report)
        if process['status']!='passed' or not process['source_integrity'] or report['status']!='passed':
            raise ValueError('Rank did not pass source or runtime checks')
        for r in report['results']:
            path=folder/'artifacts'/r['artifact']
            if sha(path)!=r['sha256']:raise ValueError('Diagnostic artifact changed')
            paths[r['mode']].append(rows(path))
            records.append(dict(rank=rank,mode=r['mode'],sha256=r['sha256'],positions=r['positions']))
    for key in ('parameters_sha256','selection','snapshot_id'):
        if any(r[key]!=reports[0][key] for r in reports):raise ValueError('Rank input disagreement')
    packed={mode:{k:np.concatenate([r[k] for r in data]) for k in data[0]} for mode,data in paths.items()}
    expected=sum(x['length'] for x in reports[0]['selection'])
    for mode,data in packed.items():
        if len(data['values'])!=expected or not all(np.isfinite(x).all() for x in data.values()):
            raise ValueError('Missing or invalid diagnostic population')
    e,t=packed['evaluation'],packed['training']
    for key in ('values','opponent','split'):np.testing.assert_array_equal(e[key],t[key])
    metrics={}
    for mode,data in packed.items():
        for prefix in ('','aux_') if mode=='training' else ('',):
            for subset,mask in {'all':np.ones(expected,bool),'train':data['split']==0,'validation':data['split']==1,
                'positive':data['values']>.5,'negative':data['values']<-.5,'uncertain':abs(data['values'])<=.5}.items():
                if mask.any():metrics[f'{mode}/{prefix or "main"}/{subset}']=summary(data[prefix+'logits'][mask],data['values'][mask])
    result=dict(status='passed',kind='joint_value_diagnostic_analysis',attempt=a.attempt.name,
        snapshot=closure['snapshot_id'],parameters_sha256=reports[0]['parameters_sha256'],positions=expected,
        artifacts=records,metrics=metrics,training_evaluation_parity=dict(
            max_abs_logits=float(np.max(abs(e['logits']-t['logits']))),
            mean_abs_logits=float(np.mean(abs(e['logits']-t['logits']))),
            max_abs_latent=float(np.max(abs(e['latent']-t['latent'])))),
        gradient_convention='Per-position unweighted gradients with respect to three logits; CE target [(1+y)/2,(1-y)/2,0] is a signed-target surrogate, not recovered teacher WDL labels.',
        operator_sha256=sha(Path(__file__)))
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='passed',positions=expected,parity=result['training_evaluation_parity'],
        selected={k:v for k,v in metrics.items() if k in ('evaluation/main/all','evaluation/main/negative','evaluation/main/positive')})))


if __name__=='__main__':main()
