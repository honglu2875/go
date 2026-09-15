"""Semantic CNN groups, checked against the pinned official model registration."""
import muon


def cnn(params):
    result = {}
    for name, p in params.items():
        if name in ('conv_spatial.weight', 'linear_global.weight'):
            group = 'input'
        elif name.startswith(('cycles.', 'tail.')):
            if name.endswith('.weight'):
                group = 'normal'
            elif name.endswith('.gamma'):
                group = 'normal_gamma'
            elif name.endswith('.beta'):
                group = 'noreg'
            else:
                raise ValueError('Unknown CNN trunk parameter role: ' + name)
        elif name.startswith(('policy_head.', 'value_head.', 'intermediate_policy_head.',
                              'intermediate_value_head.', 'norm_trunkfinal.', 'norm_intermediate_trunkfinal.')):
            if name.endswith(('.weight', '.gamma')):
                group = 'output'
            elif name.endswith(('.bias', '.beta')):
                group = 'output_noreg'
            else:
                raise ValueError('Unknown CNN output parameter role: ' + name)
        else:
            raise ValueError('Unknown CNN parameter prefix: ' + name)
        layout = None
        if group in muon.MUON_GROUPS:
            leading = int(name.startswith('cycles.'))
            if p.ndim == leading + 4:
                layout = 'conv'
            elif p.ndim == leading + 2:
                layout = 'linear'
            else:
                raise ValueError('Unknown stacked CNN matrix layout: ' + name)
        result[name] = dict(group=group, layout=layout)
    muon.validate(params, result)
    return result
