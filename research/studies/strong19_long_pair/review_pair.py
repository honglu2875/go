"""Produce endpoint/tail tables and complete curves after the paired queue closes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
METRICS=('expert_kl','family_kl','value_mse','value_family_mse')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    path=STUDY/'sequence-002/result.json';started=time.time()
    while not path.exists():
        if not args.wait:raise FileNotFoundError(path)
        if time.time()-started>140000:raise TimeoutError('Paired result did not close within the review window')
        time.sleep(30)
    result=json.loads(path.read_text())
    if result['registration_sha256']!='d343a8960ad09bca2ca58205d3d53b5fec1efeee0f53bb0cdb29e72b65254d08':
        raise ValueError('Unexpected paired registration')
    if result['status']!='passed':
        with (STUDY/'review-not-run-001.json').open('x') as f:
            json.dump(dict(status='blocked_by_failed_sequence',sequence_result_sha256=sha(path),error=result.get('error')),f,indent=2)
        return
    curves=[];arms={};inputs={str(path.relative_to(ROOT)):sha(path)}
    for arm in ('cnn','transformer'):
        folder=STUDY/(arm+'-long-002');audit_path=folder/'audit.json';a=json.loads(audit_path.read_text())
        if (sha(audit_path)!=result['arms'][arm]['audit_sha256'] or a['status']!='passed'
                or a['steps']!=512 or a['positions']!=27217367):raise ValueError('Audit changed')
        inputs[str(audit_path.relative_to(ROOT))]=sha(audit_path)
        attempt=result['arms'][arm]['attempt'];metrics_path=ROOT/'runs'/attempt/'rank-0/artifacts/metrics.jsonl'
        rows=[json.loads(x) for x in metrics_path.read_text().splitlines()];inputs[str(metrics_path.relative_to(ROOT))]=sha(metrics_path)
        if [r['turn'] for r in rows]!=list(range(1,513)):raise ValueError('Incomplete clock')
        summary={}
        for split in ('validation_history','training_probe_history'):
            history=a[split]
            if [r['turn'] for r in history]!=list(range(0,513,16)):raise ValueError('Evaluation cadence differs')
            summary[split]=dict(endpoint={k:history[-1]['metrics'][k] for k in METRICS},
                tail_three_mean={k:sum(r['metrics'][k] for r in history[-3:])/3 for k in METRICS})
            for row in history:
                t=row['turn'];curves.append(dict(arm=arm,split=split,turn=t,
                    learning_seconds=rows[t-1]['cumulative_learning_seconds'] if t else 0,
                    **{k:row['metrics'][k] for k in METRICS}))
        final=a['validation_history'][-1]['metrics']
        summary.update(learning_seconds=rows[-1]['cumulative_learning_seconds'],
            position_exposures=a['positions'],sustained_overfit_observations=[r for r in a['overfit_observations'] if r['sustained']],
            opponent_endpoints={str(i):dict(policy_kl=final[f'opponent_{i}_kl'],value_mse=final[f'value_opponent_{i}_mse']) for i in range(8)})
        arms[arm]=summary
    if arms['cnn']['position_exposures']!=arms['transformer']['position_exposures']:raise ValueError('Exposure budgets differ')
    comparison={aggregation:{metric:1-arms['transformer']['validation_history'][aggregation][metric]/arms['cnn']['validation_history'][aggregation][metric]
        for metric in METRICS} for aggregation in ('endpoint','tail_three_mean')}
    review=dict(kind='paired_long_joint19_complete_review',status='passed',created=time.time(),input_sha256=inputs,
        operator_sha256=sha(Path(__file__)),arms=arms,relative_transformer_gain=comparison,curves=curves,
        scope='One paired supervised seed; fixed endpoint and tails, complete validation/probe curves. No test targets, playing-strength or MFU claim.')
    with (STUDY/'RESULTS_001.json').open('x') as f:json.dump(review,f,indent=2,allow_nan=False);f.write('\n')
    with (STUDY/'curves-001.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    lines=['# Longer paired 19×19 results','',
        '| Arm | Endpoint policy KL | Last-three policy KL | Family KL | Value MSE | Learning minutes |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm,data in arms.items():
        m=data['validation_history'];e=m['endpoint'];tail=m['tail_three_mean']
        lines.append(f"| {arm} | {e['expert_kl']:.6f} | {tail['expert_kl']:.6f} | {e['family_kl']:.6f} | {e['value_mse']:.6f} | {data['learning_seconds']/60:.2f} |")
    lines+=['',review['scope'],'',
        'Positive relative transformer gains below mean lower loss; endpoint and tail must be interpreted together.','',
        '| Metric | Endpoint transformer gain | Last-three transformer gain |','| --- | ---: | ---: |']
    for k in METRICS:lines.append(f"| {k} | {100*comparison['endpoint'][k]:.2f}% | {100*comparison['tail_three_mean'][k]:.2f}% |")
    lines+=['',f"Registered sustained-overfit observations: CNN {len(arms['cnn']['sustained_overfit_observations'])}; transformer {len(arms['transformer']['sustained_overfit_observations'])}.",
        '', 'The fixed endpoint is reported without selecting a best validation checkpoint. Full curves and opponent-stratified endpoints are in RESULTS_001.json.',
        '', '![Full learning curves](curves-001.png)']
    with (STUDY/'RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    subprocess.run([str(ROOT/'.gozero/analysis-environments/plotting/bin/python'),'-B',str(STUDY/'plot_pair.py')],
        check=True,timeout=180,env={**__import__('os').environ,'MPLCONFIGDIR':'/tmp/go-long-pair-mpl'})
    print(json.dumps(dict(status='passed',report=str(STUDY/'RESULTS.md'),relative_transformer_gain=comparison)),flush=True)


if __name__=='__main__':main()
