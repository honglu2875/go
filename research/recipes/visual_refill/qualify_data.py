#!/usr/bin/env python3
"""Compare training observations with independent native game execution."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.native import load_library
from gozero.snapshots import canonical_json, read_json, verify
from gozero.visual_sequence_batches import Dataset
import observations


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'native-receipt', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--dataset-sha256', required=True)
    p.add_argument('--native-receipt-sha256', required=True)
    a = p.parse_args(); verify(SOURCE)
    if sha256(a.native_receipt) != a.native_receipt_sha256:
        raise ValueError('Native observation reference differs')
    receipt = read_json(a.native_receipt)
    native = load_library(a.native_receipt.parent / receipt['filename'], receipt['binary_sha256'])
    data = Dataset(a.dataset, a.dataset_sha256)
    config = {**data.manifest['rules'], 'history': 1, 'simulations': 0, 'cpuct': 1., 'max_search_edges': data.size**2 + 1}
    selected = [(role, *entry) for role in ('expert', 'behavior') for split in (0, 1, 2)
                for entry in data.indices[role, split][:4]]
    boards, transitions, capture_moves, pass_inputs = 0, 0, 0, 0
    for entry in selected:
        batch = data.batch([entry]); game = native.Game(json.dumps(config)); count = int(batch['counts'][0])
        if count != batch['actions'].shape[1]:
            raise ValueError('Full-history loader omitted a target position')
        for turn in range(count):
            request, raw = game.start(17)
            actual = observations.from_native(raw.reshape(data.size, data.size, 6))
            np.testing.assert_array_equal(batch['observations'][0, turn], actual)
            boards += 1; pass_inputs += int(actual[0, 0, 4] > 0)
            if batch[entry[0] + '_mask'][0, turn] != 1 or batch[('behavior' if entry[0] == 'expert' else 'expert') + '_mask'][0, turn] != 0:
                raise ValueError('Expert and observed-behavior objectives were mixed')
            request, _ = game.evaluate(request, np.zeros(data.size**2 + 1, np.float32), 0.)
            if request is not None:
                raise ValueError('Zero-simulation native reference unexpectedly searched')
            game.finish(); old = game.state(); action = int(batch['actions'][0, turn])
            game.play(old[1], action); new = game.state()
            if action != data.size**2 and np.count_nonzero(new[4]) <= np.count_nonzero(old[4]):
                capture_moves += 1
            transitions += 1
        # A padding row contributes neither observation history nor targets.
        padded = data.batch([entry, None], positions=count + 1)
        np.testing.assert_array_equal(padded['observations'][0, :count], batch['observations'][0])
        if padded['counts'][1] != 0 or any(np.any(padded[k][1]) for k in padded if k.endswith('_mask')):
            raise ValueError('Padded sequence acquired training targets')
    verify(SOURCE)
    result = {'schema_version': 1, 'kind': 'visual_training_input_qualification', 'status': 'passed',
              'operator_snapshot': SOURCE.name, 'dataset_manifest_sha256': a.dataset_sha256,
              'native_receipt_sha256': a.native_receipt_sha256, 'selected_episodes': selected,
              'boards_compared_exactly': boards, 'transitions_replayed': transitions,
              'capture_moves_observed': capture_moves, 'after_pass_observations': pass_inputs,
              'scope': '24 whole real games across expert/behavior train/validation/test populations; exact six-channel comparison to independent native Game execution and objective/padding checks.'}
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_bytes(canonical_json(result)); print(json.dumps(result))


if __name__ == '__main__':
    main()
