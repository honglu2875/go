"""Count dense arithmetic through static JAX loops, independent of XLA estimates.

FMA=2. Transcendentals are listed by operation, not assigned an invented
hardware-equivalent FLOP cost. Counts describe logical computation before
SPMD partitioning or accelerator tile padding, not memory transactions.
"""
from collections import Counter
import math
import numpy as np


def analyze(closed):
    counts=Counter(); unknown=Counter(); floating=Counter()
    def shape(v): return tuple(getattr(getattr(v,'aval',None),'shape',()))
    def elements(v): return math.prod(shape(v))
    def is_float(v): return str(getattr(getattr(v,'aval',None),'dtype','')) in ('float16','float32','float64','bfloat16')
    def walk(graph,multiplicity=1):
        graph=getattr(graph,'jaxpr',graph)
        for eq in graph.eqns:
            name=eq.primitive.name; p=eq.params
            if name=='scan':
                walk(p['jaxpr'],multiplicity*p['length']);continue
            if name in ('jit','pjit','remat2','custom_jvp_call','custom_vjp_call'):
                child=p.get('jaxpr',p.get('call_jaxpr',p.get('fun_jaxpr')))
                if child is None: raise ValueError('Cannot audit nested primitive '+name)
                walk(child,multiplicity);continue
            if name=='dot_general':
                dims=p['dimension_numbers'][0][0]
                contracting=math.prod(shape(eq.invars[0])[d] for d in dims)
                counts['multiply_add_flops']+=multiplicity*2*elements(eq.outvars[0])*contracting
                floating['multiply_add_flops']+=multiplicity*2*elements(eq.outvars[0])*contracting
                counts['dot_general_output_elements']+=multiplicity*elements(eq.outvars[0]);continue
            if name=='conv_general_dilated':
                kernel=shape(eq.invars[1]); dn=p['dimension_numbers'].rhs_spec
                fan=math.prod(kernel[d] for d in dn[1:])
                counts['multiply_add_flops']+=multiplicity*2*elements(eq.outvars[0])*fan
                floating['multiply_add_flops']+=multiplicity*2*elements(eq.outvars[0])*fan
                counts['convolution_output_elements']+=multiplicity*elements(eq.outvars[0]);continue
            if name in ('reduce_sum','reduce_max','reduce_min','reduce_or','reduce_and'):
                work=multiplicity*max(0,elements(eq.invars[0])-elements(eq.outvars[0]))
                counts[name+'_operations']+=work
                if is_float(eq.outvars[0]): floating[name]=floating[name]+work
                continue
            if name in ('add','sub','mul','div','neg','square','integer_pow','max','min','abs',
                        'exp','log','log1p','tanh','sqrt','rsqrt','logistic','pow','sin','cos','erf','erfc',
                        'rem','sign','eq','ne','lt','le','gt','ge','and','or','xor','not','select_n','convert_element_type'):
                counts[name+'_elements']+=multiplicity*sum(elements(v) for v in eq.outvars)
                if name not in ('select_n','convert_element_type'):
                    floating[name]+=multiplicity*sum(elements(v) for v in eq.outvars if is_float(v))
                continue
            if name in ('reshape','transpose','broadcast_in_dim','slice','squeeze','concatenate',
                        'pad','rev','iota','stop_gradient','copy','device_put','sharding_constraint','stack'):
                continue
            if name in ('dynamic_update_slice','dynamic_slice','gather','scatter','scatter-add'):
                counts[name+'_logical_output_elements']+=multiplicity*sum(elements(v) for v in eq.outvars);continue
            unknown[name]+=multiplicity
    walk(closed)
    return {'counts':dict(sorted(counts.items())),'unaccounted_primitives':dict(sorted(unknown.items())),
            'floating_operations_by_primitive':{k:v for k,v in sorted(floating.items()) if v},
            'floating_operations_unit_cost':sum(floating.values()),
            'scope':'Logical pre-SPMD arithmetic; static scan trip counts expanded; FMA=2',
            'transcendental_cost':'One scalar operation in unit-cost total; this is not a hardware-equivalent FLOP conversion'}
