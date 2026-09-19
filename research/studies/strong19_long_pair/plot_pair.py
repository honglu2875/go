"""Export the complete longer-pair curves; no interpolation or selected endpoint."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY=Path(__file__).resolve().parent
path=STUDY/'RESULTS_001.json';data=json.loads(path.read_text())
if data['status']!='passed':raise ValueError('Expected audited paired results')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
fig.get_layout_engine().set(rect=(0,.065,1,.87));handles=[]
for ri,(metric,title) in enumerate([('expert_kl','Policy KL'),('value_mse','Value MSE')]):
    for ci,(clock,label,scale) in enumerate([('turn','Accepted updates',1),('learning_seconds','Learning minutes',60)]):
        ax=axes[ri,ci]
        for arm,color in [('cnn','#7257a3'),('transformer','#19835c')]:
            for split,style in [('validation_history','-'),('training_probe_history','--')]:
                rows=[r for r in data['curves'] if r['arm']==arm and r['split']==split]
                line,=ax.plot([r[clock]/scale for r in rows],[r[metric] for r in rows],style,color=color,label=arm,linewidth=1.6)
                if ri==ci==0 and split=='validation_history':handles.append(line)
        ax.set(title=title,xlabel=label,ylabel='Lower is better',xlim=(0,None));ax.grid(alpha=.2)
fig.suptitle('Longer 19×19 comparison · AdamW and value CE for both models',fontsize=14)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
fig.text(.5,.022,'Solid: full fixed validation · Dashed: fixed training probe · Identical complete-game/D4 draws\n'
    'One paired seed, 512 updates each. Learning time excludes compilation, evaluation and checkpointing.',ha='center',fontsize=9)
outputs=[]
for suffix in ('.png','.pdf'):
    out=STUDY/('curves-001'+suffix)
    if out.exists():raise FileExistsError(out)
    fig.savefig(out,dpi=160);outputs.append(out)
plt.close(fig)
with (STUDY/'plot-receipt-001.json').open('x') as f:
    json.dump(dict(status='generated_not_yet_visually_reviewed',input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}),f,indent=2)
