"""Render the unchanged gradient diagnostics with concise, unclipped labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOLDER=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((FOLDER/'readout_gradient_context_diagnostic.json').read_text())
b=json.loads((FOLDER/'trained_readout_gradient_context_diagnostic.json').read_text())
assert a['training_episode']==b['training_episode'] and a['common_endpoint']==b['common_endpoint']
out=FOLDER/'trained_gradient_review_v2';out.mkdir()
fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
for source,label in [(a,'At initialization'),(b,'After 1,024 updates')]:
    cases=[c for c in source['cases'] if c.get('pooling','learned')=='learned']
    values=[100*np.asarray(c['gradient_l2_by_frame'])[-1]/sum(c['gradient_l2_by_frame']) for c in cases]
    axes[0].plot([c['prefix_moves'] for c in cases],values,marker='o',label=label)
    norm=np.asarray(cases[-1]['gradient_l2_by_frame'])
    axes[1].semilogy(np.arange(1,129),100*norm/norm.sum(),label=label)
axes[0].set_xticks([16,64,128]);axes[0].set_ylim(0,40)
axes[0].set_xlabel('Observed history (moves)');axes[0].set_ylabel('Current-board gradient share (%)')
axes[1].set_xlabel('Frame in the 128-move history');axes[1].set_ylabel('Frame gradient share (%)')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('Gradient distribution in the original C128 transformer')
for name in ['gradients.png','gradients.svg']:
    fig.savefig(out/name,dpi=160,bbox_inches='tight');(out/name).chmod(0o444)
lines=['On this example, the original C128 model develops a stronger current-board gradient path during training. At 128 moves, its share of the sum of frame gradient norms rises from 0.805% at initialization to 25.090% after 1,024 updates. This limits what the initialization probe alone can establish.',
       '', '| History | At initialization | After 1,024 updates |','|---:|---:|---:|']
for c in b['cases']:
    original=next(x for x in a['cases'] if x['prefix_moves']==c['prefix_moves'] and x['pooling']=='learned')
    share=c['fraction_of_summed_frame_norms_in_last_n_frames']['1']
    lines.append(f"| {c['prefix_moves']} moves | {100*original['current_frame_fraction_of_summed_frame_norms']:.3f}% | {100*share:.3f}% |")
lines+=['','One fixed training example at the same endpoint, using the audited final control checkpoint. Shares use the sum of per-frame L2 gradient norms. They are not attention probabilities or a causal attribution. The shorter contexts truncate the original history and can change the input distribution. This is a mechanism diagnostic, not a held-out learnability comparison. Judge the actual readout intervention by its completed validation curve.',
        '','[Figure](gradients.png) · [Trained diagnostic](../trained_readout_gradient_context_diagnostic.json) · [Initialization diagnostic](../readout_gradient_context_diagnostic.json)',
        '','This publication corrects labels and clipping in the first figure; the diagnostic data are unchanged.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'REPORT.md').chmod(0o444)
manifest={'kind':'trained_readout_gradient_review','format_revision':2,
          'driver_sha256':sha(Path(__file__)),
          'diagnostics':{n:sha(FOLDER/n) for n in ['readout_gradient_context_diagnostic.json','trained_readout_gradient_context_diagnostic.json','diagnose_trained_readout.py']},
          'files':{n:sha(out/n) for n in ['REPORT.md','gradients.png','gradients.svg']}}
with (out/'manifest.json').open('xb') as f:f.write((json.dumps(manifest,sort_keys=True)+'\n').encode())
(out/'manifest.json').chmod(0o444);print(out,sha(out/'manifest.json'))
