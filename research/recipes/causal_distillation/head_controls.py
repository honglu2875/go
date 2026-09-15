#!/usr/bin/env python3
"""Compare fixed trained heads on held-out observed episodes, with raw game sums."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import read as checkpoint_read, sha256
from gozero.sequence_batches import Dataset
from gozero.snapshots import canonical_json, read_json, verify


def bootstrap_difference(rows, left, right, draws, seed):
    weights = np.asarray([r['tokens'] for r in rows], np.float64)
    difference = np.asarray([r[left] - r[right] for r in rows], np.float64)
    random = np.random.Generator(np.random.PCG64(seed))
    samples = []
    for begin in range(0, draws, 256):
        indices = random.integers(len(rows), size=(min(256, draws - begin), len(rows)))
        samples.extend((difference[indices].sum(axis=1) / weights[indices].sum(axis=1)).tolist())
    return {'token_weighted_nll_difference': float(difference.sum() / weights.sum()),
            'episode_bootstrap_95_interval': np.quantile(samples, [.025, .975], method='linear').tolist()}


def main():
    import jax
    import jax.numpy as jnp
    import model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify(SOURCE)
    protocol = read_json(args.protocol); root = args.workspace_root.resolve()
    if sha256(args.protocol) != args.expected_protocol_sha256 or jax.default_backend() != 'cpu' or jax.process_count() != 1:
        raise ValueError('Registered protocol or CPU execution differs')
    pilot_path = root / protocol['pilot_result']
    if sha256(pilot_path) != protocol['pilot_result_sha256']:
        raise ValueError('Trained pilot identity differs')
    pilot = read_json(pilot_path); training = root / '.gozero/snapshots' / pilot['source_snapshot']; manifest = verify(training)
    if sha256(training / manifest['recipe'] / 'model.py') != sha256(Path(__file__).with_name('model.py')):
        raise ValueError('Head implementation differs from training')
    config = read_json(training / 'resolved_config.json'); c = config['model']
    artifacts = root / 'runs' / pilot['attempt'] / 'rank-0/artifacts'
    checkpoint = artifacts / 'checkpoints' / 'turn-000004096'
    group = read_json(checkpoint.with_suffix('.group.json'))
    state = read_json(checkpoint / 'state.json')
    state, arrays, _ = checkpoint_read(checkpoint, expected_manifest_sha256=group['rank_manifests'][state['jax_rank']], array_prefix='p_')
    template, definition = jax.tree.flatten(model.initialize(config['seed'], c))
    if set(arrays) != {f'p_{i:04d}' for i in range(len(template))}:
        raise ValueError('Checkpoint parameter tree differs')
    if sha256(artifacts / 'model_export.npz') != pilot['model_export_sha256']:
        raise ValueError('Trained model export differs')
    with np.load(artifacts / 'model_export.npz', allow_pickle=False) as exported:
        if any(not np.array_equal(arrays[n], exported[n]) for n in arrays):
            raise ValueError('Checkpoint and export parameters differ')
    params = definition.unflatten([jnp.asarray(arrays[f'p_{i:04d}']) for i in range(len(template))])
    data = Dataset(config['dataset']['path'], protocol['dataset_manifest_sha256'])
    actions = c['size']**2 + 1
    frequency = np.full(actions, protocol['frequency_pseudocount'], np.float64)
    for shard, episode in data.indices['behavior', 0]:
        a = data.shards[shard]; begin, end = a['behavior_offsets'][episode:episode+2]
        frequency += np.bincount(a['behavior_actions'][begin:end], minlength=actions)
    frequency /= frequency.sum()

    def forward(batch):
        hidden, _ = model.prefill(params, batch['tokens'], batch['lengths'], c)
        counts = model.historical_counts(batch['tokens'], batch['lengths'], actions)
        context = jnp.take_along_axis(counts, (jnp.arange(batch['tokens'].shape[1]) % 2)[None, :, None, None], axis=2)[:, :, 0]
        play, behavior, _ = model.head_outputs(params, hidden, context)
        _, zero_counts, _ = model.head_outputs(params, hidden, jnp.zeros_like(context))
        def sums(logits):
            nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), batch['observed'][..., None], axis=-1)[..., 0]
            return jnp.sum(batch['behavior_mask'] * nll, axis=1)
        return {name: sums(logits) for name, logits in [('play', play), ('behavior', behavior), ('zero_counts', zero_counts)]}

    predict = jax.jit(forward)
    entries = [('behavior', *index) for index in data.indices['behavior', 2]]
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time(); rows = []
    for begin in range(0, len(entries), protocol['batch_games']):
        selected = entries[begin:begin+protocol['batch_games']]
        batch = data.batch(selected + [None] * (protocol['batch_games'] - len(selected)))
        values = jax.device_get(predict(batch))
        for row, (_, shard, episode) in enumerate(selected):
            n = int(batch['behavior_mask'][row].sum()); moves = batch['observed'][row, :n]
            record = {'shard': shard, 'game_id': int(data.shards[shard]['behavior_game_ids'][episode]), 'tokens': n,
                      **{key: float(value[row]) for key, value in values.items()}, 'frequency': float(-np.log(frequency[moves]).sum())}
            if not all(np.isfinite(record[k]) for k in ('play', 'behavior', 'zero_counts', 'frequency')):
                raise FloatingPointError('Nonfinite head comparison')
            rows.append(record)
    totals = sum(r['tokens'] for r in rows)
    comparisons = {other: bootstrap_difference(rows, 'behavior', other, protocol['bootstrap_draws'], protocol['bootstrap_seed'])
                   for other in ('play', 'zero_counts', 'frequency')}
    primary = comparisons['play']
    result = {'schema_version': 1, 'kind': 'causal_head_prediction_controls', 'status': 'passed', 'analysis_snapshot': SOURCE.name,
              'protocol_sha256': args.expected_protocol_sha256, 'model_export_sha256': pilot['model_export_sha256'],
              'games': len(rows), 'tokens': totals, 'frequency_train_games': len(data.indices['behavior', 0]),
              'token_weighted_nll': {key: sum(r[key] for r in rows) / totals for key in ('behavior', 'play', 'zero_counts', 'frequency')},
              'comparisons': comparisons, 'behavior_head_criterion_met': primary['token_weighted_nll_difference'] <= -protocol['minimum_nll_gain']
                  and primary['episode_bootstrap_95_interval'][1] < 0,
              'started_unix': started, 'finished_unix': time.time(), 'limitations': protocol['limitations']}
    if result['finished_unix'] - started > protocol['maximum_seconds']:
        raise RuntimeError('Head comparison exceeded registered CPU budget')
    verify(SOURCE)
    (args.output / 'games.json').write_bytes(canonical_json(rows))
    result['games_sha256'] = sha256(args.output / 'games.json')
    (args.output / 'result.json').write_bytes(canonical_json(result))
    for path in args.output.iterdir():
        path.chmod(0o444)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
