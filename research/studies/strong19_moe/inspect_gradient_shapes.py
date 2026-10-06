"""Abstract-only reverse-mode shape inspection; allocates no full model/data."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'research/recipes/strong19_moe'),str(ROOT/'packages/gozero/src')]
import jax
import jax.numpy as jnp
import numpy as np
import joint,learner
from gozero.snapshots import canonical_json
base=json.loads((STUDY/'balance010-001-config.json').read_text());c=base['model'];t=512;b=8
p=jax.eval_shape(lambda:joint.initialize(0,c,base['value_model']))
spec=lambda shape,dtype=jnp.float32:jax.ShapeDtypeStruct(shape,dtype)
batch=dict(spatial=spec((b,t,19,19,22)),global_features=spec((b,t,19)),actions=spec((b,t),jnp.int32),counts=spec((b,),jnp.int32),policies=spec((b,t,362)),legal=spec((b,t,362),jnp.bool_),values=spec((b,t)))
fn=lambda p,b:learner.loss(p,b,c,value_weight=.7,path='bounded',chunk_frames=8,skip_padding=True)
jaxpr=jax.make_jaxpr(jax.value_and_grad(fn,has_aux=True))(p,batch)
rows=[]
def visit(value,path):
    if hasattr(value,'jaxpr'):value=value.jaxpr
    if hasattr(value,'eqns'):
        for i,eq in enumerate(value.eqns):
            label=path+'/'+str(i)+':'+eq.primitive.name
            for index,v in enumerate(eq.outvars):
                a=getattr(v,'aval',None)
                if a is not None and hasattr(a,'shape') and hasattr(a,'dtype'):
                    elements=int(np.prod(a.shape,dtype=np.int64));size=elements*np.dtype(a.dtype).itemsize
                    if size>=16<<20:rows.append(dict(path=label,output=index,shape=list(a.shape),dtype=str(a.dtype),bytes=size,primitive=eq.primitive.name))
            for k,v in eq.params.items():visit(v,label+'/'+k)
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):visit(v,path+'/'+str(i))
    elif isinstance(value,dict):
        for k,v in value.items():visit(v,path+'/'+str(k))
visit(jaxpr,'gradient')
rows.sort(key=lambda r:r['bytes'],reverse=True)
result=dict(status='passed',scope='Abstract JAXpr output extents; not live-buffer allocation or measured peak memory.',largest_outputs=rows[:120],largest_scan_outputs=[r for r in rows if r['primitive']=='scan'][:60])
with (STUDY/'gradient-shapes-001.json').open('xb') as f:f.write(canonical_json(result))
print(json.dumps(result['largest_outputs'][:16],indent=2))
