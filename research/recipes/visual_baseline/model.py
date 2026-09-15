"""Common supervised objectives; architecture is an explicit frozen choice."""
import jax
if __package__:
    from . import cnn, transformer
    from .transformer import supervised_losses, exit_distillation
else:
    import cnn
    import transformer
    from transformer import supervised_losses, exit_distillation

def implementation(c):
    return cnn if c.get('architecture') == 'residual_cnn' else transformer

def initialize(seed,c): return implementation(c).initialize(seed,c)
def parameter_schema(c): return implementation(c).parameter_schema(c)
def forward(p,o,a,n,c,**kwargs): return implementation(c).forward(p,o,a,n,c,**kwargs)

def losses(p,batch,c,*,axis_name=None,exit_depths=(),exit_loss_weight=0.,exit_temperature=1.):
    prediction=forward(p,batch['observations'],batch['actions'],batch['counts'],c,return_exits=exit_depths)
    if not exit_depths:
        return supervised_losses(prediction,batch,c,axis_name=axis_name)
    teacher,students=prediction
    total,metrics=supervised_losses(teacher,batch,c,axis_name=axis_name)
    auxiliary,extra=exit_distillation(teacher,students,batch,axis_name=axis_name,temperature=exit_temperature)
    return total+exit_loss_weight*auxiliary,{**metrics,**extra,'exit_distillation_loss':auxiliary}

def layout(size,c): return transformer.layout(size,c)
def score_continuation(*args,**kwargs): return transformer.score_continuation(*args,**kwargs)
