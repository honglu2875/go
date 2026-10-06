"""Summarize a completed same-device tile/VJP screen without choosing a learner."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.audit.read_text())
    if report['status']!='passed':raise ValueError('A passed complete audit is required')
    groups={}
    for host in report['reports']:
        for case in host['cases']:
            key=tuple(case[k] for k in ('tokens','width','hidden','gated','collapsed'))
            groups.setdefault(key,[]).append((host['host_rank'],host['jax_rank'],case))
    summaries=[]
    for key,items in sorted(groups.items()):
        if len(items)!=7 or len({(h,r) for h,r,_ in items})!=1:
            raise ValueError('Expected seven candidates on one physical device per group')
        controls=[c for _,_,c in items if c['tiling']==[256,512,512] and not c['permutation_vjp']]
        if len(controls)!=1:raise ValueError('Unique current-tile control is required')
        control=controls[0];cases=[]
        for _,_,c in items:
            if c['status']!='passed' or max(e['relative_l2'] for e in c['errors'])>.01:
                raise ValueError('Numerical qualification failed')
            m=c['memory_bytes'];peak=m['argument_size_in_bytes']+m['output_size_in_bytes']+m['temp_size_in_bytes']-m['alias_size_in_bytes']
            cases.append(dict(tiling=c['tiling'],permutation_vjp=c['permutation_vjp'],
                              forward_seconds=c['sparse_forward_median_seconds'],
                              forward_backward_seconds=c['sparse_median_seconds'],
                              forward_speedup=control['sparse_forward_median_seconds']/c['sparse_forward_median_seconds'],
                              forward_backward_speedup=control['sparse_median_seconds']/c['sparse_median_seconds'],
                              measured_dense_forward_seconds=c['dense_forward_median_seconds'],
                              measured_dense_forward_backward_seconds=c['dense_median_seconds'],
                              forward_min_max_seconds=[min(c['sparse_forward_seconds']),max(c['sparse_forward_seconds'])],
                              forward_backward_min_max_seconds=[min(c['sparse_forward_backward_seconds']),max(c['sparse_forward_backward_seconds'])],
                              maximum_relative_l2=max(e['relative_l2'] for e in c['errors']),compiled_peak_bytes=peak))
        summaries.append(dict(tokens=key[0],width=key[1],hidden=key[2],gated=key[3],collapsed=key[4],
                              physical_host_rank=items[0][0],cases=cases,
                              fastest_forward=min(cases,key=lambda c:c['forward_seconds']),
                              fastest_forward_backward=min(cases,key=lambda c:c['forward_backward_seconds']),
                              dense_forward_backward_range_seconds=[min(c['measured_dense_forward_backward_seconds'] for c in cases),max(c['measured_dense_forward_backward_seconds'] for c in cases)]))
    if len(summaries)!=10:raise ValueError('Incomplete shape/routing coverage')
    result=dict(status='passed',snapshot=report['snapshot'],attempt=report['attempt'],groups=summaries,
                audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),
                operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Ten-repeat synchronized FFN microbenchmarks including dispatch and task gradients. Router balance/z losses, full-model attention, collectives and optimizer work are excluded. Candidates and controls within each group share a device. Fastest medians are observations, not full-model throughput, MFU, convergence, or strength claims.')
    a.output_directory.mkdir(exist_ok=False)
    (a.output_directory/'review.json').write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
    lines=['# Grouped-expert kernel screen','',result['scope'],'',
           'All 70 cases passed the coordinate and 1% complete-leaf relative-L2 gates.','',
           '| Tokens / expert hidden / activation | Routing | Current forward+backward ms | Fastest training tile / VJP | Training speedup | Fastest forward tile / VJP | Forward speedup |',
           '| --- | --- | ---: | --- | ---: | --- | ---: |']
    label=lambda c:'×'.join(map(str,c['tiling']))+(' / permutation' if c['permutation_vjp'] else ' / default')
    for g in summaries:
        base=next(c for c in g['cases'] if c['tiling']==[256,512,512] and not c['permutation_vjp'])
        train,forward=g['fastest_forward_backward'],g['fastest_forward']
        lines.append(f"| {g['tokens']} / {g['hidden']} / {'SwiGLU' if g['gated'] else 'GELU'} | {'concentrated' if g['collapsed'] else 'random'} | {base['forward_backward_seconds']*1000:.3f} | {label(train)} | {train['forward_backward_speedup']:.3f}× | {label(forward)} | {forward['forward_speedup']:.3f}× |")
    lines+=['','Small-batch results include Python dispatch/synchronization overhead. The dense controls are independently timed in every case; their spread and all sample ranges are retained in `review.json`.','',
            'No model or learning configuration is changed by this report. Any implementation selected from these measurements needs a complete learner memory/state/timing qualification.','']
    (a.output_directory/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='passed',groups=len(summaries),cases=70,output=str(a.output_directory))))


if __name__=='__main__':main()
