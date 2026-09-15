"""Publish the SGD control against both original and LR-selected CNN controls."""
import argparse
import json
import os
from pathlib import Path
import sys
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE/'packages/gozero/src'))
from gozero.snapshots import read_json, verify, canonical_json
from gozero.checkpoints import sha256


def publish(path, value):
    with path.open('xb') as f:
        f.write(canonical_json(value)); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(),0o444)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--sgd-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); verify(SOURCE); root = a.workspace_root.resolve()
    if sha256(a.registration) != a.registration_sha256: raise ValueError('Registration changed')
    reg = read_json(a.registration); sgd = read_json(a.sgd_audit)
    if sgd['status'] != 'passed' or sgd['operator_snapshot'] != SOURCE.name: raise ValueError('Wrong SGD audit')
    for path,digest in sgd['input_files'].items():
        if sha256(root/path) != digest: raise ValueError('Audited SGD input changed')
    curves = [{'label':'SGD: historical256-position recipe','optimizer':'Nesterov SGD','curve':sgd['validation_curve'],
               'timing':sgd['timing'],'reserved_chip_hours':sgd['reserved_chip_hours'],'audit_sha256':sha256(a.sgd_audit)}]
    for ref in reg['comparison_references']:
        audit_path = root/ref['audit_path']
        if sha256(audit_path) != ref['audit_sha256']: raise ValueError('CNN reference changed')
        audit = read_json(audit_path); attempt = root/'runs'/audit['attempt']
        for name,digest in audit['input_files'].items():
            if sha256(root/name) != digest: raise ValueError('CNN input changed')
        records = [json.loads(x) for x in (attempt/'rank-0/artifacts/metrics.jsonl').read_text().splitlines()]
        exposures = {0:0}; elapsed = {0:0.}; total = 0
        for row in records:
            total += int(row['expert_positions']); exposures[row['turn']] = total; elapsed[row['turn']] = row['cumulative_learning_seconds']
        curve = [{**x,'step':x['turn'],'position_exposures':exposures[x['turn']],
                  'learning_seconds':elapsed[x['turn']]} for x in audit['validation_curve']]
        if total != sgd['position_exposures']: raise ValueError('Unequal final exposures')
        if any(x['episode_ids_sha256'] != sgd['validation_curve'][0]['episode_ids_sha256'] for x in curve): raise ValueError('Validation changed')
        curves.append({'label':ref['label'],'optimizer':'AdamW','curve':curve,'timing':audit['timing'],
                       'reserved_chip_hours':audit['reserved_chip_hours'],'audit_sha256':ref['audit_sha256']})
    a.output.mkdir(parents=True, exist_ok=False)
    result = {'kind':'historical_sgd_fixed_data_comparison','status':'passed','snapshot':SOURCE.name,
              'registration_sha256':a.registration_sha256,'sgd_audit_sha256':sha256(a.sgd_audit),'arms':curves,
              'scope':sgd['scope'],'test_split_closed':True,
              'selection_note':'AdamW1e-3 was selected from four tested rates; this is one pre-registered SGD recipe. Neither is established as a global optimum.'}
    publish(a.output/'comparison.json',result)
    lines = ['The historical SGD control and CNN AdamW controls used the same **11,469,333 position/symmetry exposures**, shuffled and regrouped into256-position SGD batches. All share the same model parameters at initialization and full validation population.\n',
             '| Recipe | Updates | Final validation KL | Top-move agreement | Learning minutes | Attempt chip-hours |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in curves:
        endpoint = arm['curve'][-1]
        lines.append(f"| {arm['label']} | {endpoint['step']:,} | {endpoint['metrics']['expert_kl']:.6f} | {100*endpoint['metrics']['expert_top1']:.2f}% | {arm['timing']['learning_seconds']/60:.2f} | {arm['reserved_chip_hours']:.2f} |")
    lines += ['', 'SGD uses Nesterov momentum0.9, per-sample LR2e-5 through the first5M exposures then6e-5, and coupled L2 coefficient3e-5 in the historical TensorFlow half-squared-norm convention. The final batch has21 positions. There is no gradient clipping, Lookahead, cosine decay or compressed imitation of the original19-day run’s final LR drop.',
              '', 'The current9x9 fixed weak-teacher corpus, modern norm-free main CNN, same-target training helper and BF16 arithmetic differ from the2019 self-play run. This checks transfer of its optimizer recipe. Batch grouping/order, optimizer, schedule and regularization differ from our AdamW controls; update count and training FLOPs are not equal. AdamW1e-3 is a selected endpoint from a four-rate grid; SGD is one pre-registered recipe. The test split remains closed. No Go-strength or RL-efficiency conclusion follows.',
              '', '[Exposure and wall-time curves](learning.png) · [Audited comparison](comparison.json)']
    path = a.output/'REPORT.md'; path.write_text('\n'.join(lines)+'\n'); path.chmod(0o444)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm in curves:
        y = [x['metrics']['expert_kl'] for x in arm['curve']]
        axes[0].plot([x['position_exposures']/1e6 for x in arm['curve']],y,marker='o',label=arm['label'])
        axes[1].plot([x['learning_seconds']/60 for x in arm['curve']],y,marker='o',label=arm['label'])
    for ax in axes:
        ax.set_ylabel('Validation policy KL'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_xlabel('Training position exposures (millions)'); axes[1].set_xlabel('Learning time (minutes)')
    for name in ('learning.png','learning.svg'):
        fig.savefig(a.output/name,dpi=160); (a.output/name).chmod(0o444)
    plt.close(fig)
    publish(a.output/'manifest.json',{'kind':result['kind'],'status':'passed','snapshot':SOURCE.name,
        'files':{name:sha256(a.output/name) for name in ('REPORT.md','comparison.json','learning.png','learning.svg')}})
    print(json.dumps({'status':'passed','publication':str(a.output)}),flush=True)

if __name__ == '__main__': main()
