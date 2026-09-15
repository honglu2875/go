"""Export actual pinned Torch optimizer updates and official model groups."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REVISION = '92ee95c0a4b25fec214da00951ab69e97e207729'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pin_path = ROOT / 'research/studies/katago_muon/reference-source-001.json'
    pin = json.loads(pin_path.read_text())
    source = ROOT / pin['source']
    for name, expected in pin['files'].items():
        if sha(source / name) != expected:
            raise ValueError('Pinned official source changed')
    os.environ['KATAGO_MUON_BATCHED_NS'] = '0'
    os.environ['KATAGO_AUX_ADAM_FOREACH'] = '0'
    sys.path.insert(0, str(source / 'python'))
    import torch
    from muon import muon as official
    from katago.train import modelconfigs, model_pytorch
    torch.set_num_threads(2)
    torch.manual_seed(93117)
    definitions = [
        ('cycles.a.normactconvp.conv.weight', 'normal', 'conv', (1, 1, 12, 8), 3),
        ('cycles.b.blockstack.0.normactconv1.conv.weight', 'normal', 'conv', (3, 3, 8, 16), 3),
        ('cycles.g.blockstack.0.normactconv1.convpool.linear_g.weight', 'normal', 'linear', (12, 10), 3),
        ('tail.0.blockstack.0.normactconv1.conv.weight', 'normal', 'conv', (3, 3, 1, 12), 0),
        ('conv_spatial.weight', 'input', 'conv', (3, 3, 22, 16), 0),
        ('cycles.a.normactconvp.norm.gamma', 'normal_gamma', 'vector', (16,), 3),
        ('cycles.a.normactconvp.norm.beta', 'noreg', 'vector', (16,), 3),
        ('value_head.linear_valuehead.weight', 'output', 'linear', (7, 3), 0),
        ('value_head.linear_valuehead.bias', 'output_noreg', 'vector', (3,), 0),
    ]
    rates = dict(normal=1.7e-4, input=9e-5, normal_gamma=9e-5, noreg=1.1e-4, output=8e-5, output_noreg=1.2e-4)
    decays = dict(normal=.02, input=.006, normal_gamma=.00225, noreg=1e-6, output=.004, output_noreg=1e-6)
    rng = np.random.default_rng(631112)
    arrays, leaves, groups = {}, [], {}

    def convert(x, layout):
        if layout == 'conv':
            return np.transpose(x, (3, 2, 0, 1)).copy()
        if layout == 'linear':
            return x.T.copy()
        return x.copy()

    def restore(x, layout):
        x = x.detach().float().numpy().copy()
        if layout == 'conv':
            return np.transpose(x, (2, 3, 1, 0)).copy()
        if layout == 'linear':
            return x.T.copy()
        return x

    for i, (name, group, layout, shape, stacks) in enumerate(definitions):
        values = (rng.normal(size=(max(stacks, 1), *shape)) * .05).astype(np.float32)
        params = [torch.nn.Parameter(torch.tensor(convert(x, layout))) for x in values]
        groups.setdefault(group, []).extend(params)
        leaves.append(dict(name=name, group=group, layout=layout, shape=list(shape), stacks=stacks, params=params))
        arrays[f'p0.{i}'] = values if stacks else values[0]
    optimizer = official.SingleDeviceMuonWithAuxAdam([
        dict(params=values, use_muon=group == 'normal', lr=rates[group], weight_decay=decays[group])
        for group, values in groups.items()
    ])
    with torch.no_grad():
        for step in range(1, 5):
            for i, leaf in enumerate(leaves):
                values = (rng.normal(size=(max(leaf['stacks'], 1), *leaf['shape'])) * .15).astype(np.float32)
                if step == 4:
                    values[:] = 0
                arrays[f'g{step}.{i}'] = values if leaf['stacks'] else values[0]
                directions = []
                for value, p in zip(values, leaf['params']):
                    p.grad = torch.tensor(convert(value, leaf['layout']))
                    old = optimizer.state[p]
                    if leaf['group'] == 'normal':
                        direction = official.muon_update(p.grad.clone(), old.get('momentum_buffer', torch.zeros_like(p)).clone())
                        direction = direction.reshape(p.shape)
                    else:
                        direction = official.adam_update(p.grad.clone(), old.get('exp_avg', torch.zeros_like(p)).clone(),
                                                         old.get('exp_avg_sq', torch.zeros_like(p)).clone(), step, (.95, .995), 1e-6)
                    directions.append(restore(direction, leaf['layout']))
                arrays[f'u{step}.{i}'] = np.stack(directions) if leaf['stacks'] else directions[0]
            optimizer.step()
            for i, leaf in enumerate(leaves):
                for field, state_key in [('p', None), ('m', 'momentum_buffer' if leaf['group'] == 'normal' else 'exp_avg'), ('v', 'exp_avg_sq')]:
                    if field == 'v' and leaf['group'] == 'normal':
                        continue
                    values = [restore(p if state_key is None else optimizer.state[p][state_key], leaf['layout']) for p in leaf['params']]
                    arrays[f'{field}{step}.{i}'] = np.stack(values) if leaf['stacks'] else values[0]
        x = torch.tensor(rng.normal(size=(7, 11)).astype(np.float32)).bfloat16()
        arrays['bf16_input'] = x.float().numpy()
        for scalar in (3.4445, -4.775, 2.0315, .2 * np.sqrt(11)):
            arrays['bf16_scalar_' + str(scalar)] = (x * float(scalar)).float().numpy()
    config = copy.deepcopy(modelconfigs.config_of_name['b40c768nbt-fson-mish-rvglr-bnh'])
    config.update(trunk_num_channels=16, mid_num_channels=8, gpool_num_channels=2, p1_num_channels=4,
                  g1_num_channels=4, v1_num_channels=5, sbv2_num_channels=7, v2_size=7)
    config['block_kind'] = config['block_kind'][:4]
    config['intermediate_head_blocks'] = 4
    model = model_pytorch.Model(config, pos_len=19)
    reg = {}
    model.add_reg_dict(reg)
    named = {id(p): name for name, p in model.named_parameters()}
    registered = {}
    for group, parameters in reg.items():
        for p in parameters:
            name = named[id(p)]
            if name in registered:
                raise ValueError('Duplicate official model parameter group')
            registered[name] = dict(group=group, shape=list(p.shape))
    if len(named) != len(registered):
        raise ValueError('Official model grouping is incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'arrays.npz').open('xb') as f:
        np.savez(f, **arrays)
    report = dict(kind='official_katago_muon_reference', status='passed', source_revision=REVISION,
                  source_pin_sha256=sha(pin_path), operator_sha256=sha(Path(__file__)),
                  torch_version=torch.__version__, rates=rates, decays=decays, steps=4,
                  leaves=[{k: v for k, v in leaf.items() if k != 'params'} for leaf in leaves],
                  official_model_config=config, official_parameter_groups=registered,
                  arrays_sha256=sha(args.output / 'arrays.npz'),
                  scope='Actual scalar standard Muon/AuxAdam updates and complete small official model grouping; no learned-model performance claim.')
    with (args.output / 'manifest.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    for path in args.output.iterdir():
        path.chmod(0o444)
    print(json.dumps(dict(status='passed', manifest_sha256=sha(args.output / 'manifest.json'),
                          leaves=len(leaves), official_model_parameters=len(registered))), flush=True)


if __name__ == '__main__':
    main()
