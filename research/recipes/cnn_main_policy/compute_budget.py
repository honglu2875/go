"""Model-wide arithmetic accounting; FMA=2, padded dense spatial convolution.

This is algorithmic multiply-add work, not a claim about achieved MFU or an
XLA compiler estimate. The compiled executable is audited independently.
"""
import katago


def cnn(c,board_size=9,batch_size=128):
    schema=katago.parameter_schema(c); parts={}
    for s in schema:
        name=s['path']
        if not name.endswith('.weight') or name.startswith(('intermediate_','norm_intermediate_')): continue
        spatial=len(s['shape']) >= 4
        work=2*s['elements']*(board_size**2 if spatial else 1)*batch_size
        group='encoder' if name.startswith(('conv_spatial.','linear_global.')) else 'head' if name.startswith('policy_head.') else 'trunk'
        parts[group]=parts.get(group,0)+work
    return {'board_size':board_size,'batch_size':batch_size,'trainable_parameters':sum(s['elements'] for s in schema),
        'inference_parameters':katago.inference_parameter_count(c),'multiply_add_flops_by_component':parts,
        'multiply_add_flops_per_batch':sum(parts.values()),'multiply_add_flops_per_move':sum(parts.values())/batch_size,
        'fma_flops':2,'input_layout':'Dense actual board extent, including same-convolution boundary padding',
        'elementwise_operations':'Not included in dense MAC subtotal; compiler estimate reported separately',
        'training_only_helper':'Parameters counted above; compute excluded from inference'}
