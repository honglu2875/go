"""Prepare explicit three-arm scientific settings; does not launch training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packages/gozero/src'))
from gozero.snapshots import canonical_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value))
    path.chmod(0o444)


def main():
    plan_path = STUDY / 'cohort-plan-002.json'
    if sha(plan_path) != '9270f03a2080e19e3036d30d3d184dd6d56fdfeb4309824f87c64046350bc5f6':
        raise ValueError('Cohort plan changed')
    plan = json.loads(plan_path.read_text())
    prepared = json.loads((STUDY / 'cohort-result-002.json').read_text())
    if (sha(STUDY / 'cohort-result-002.json') != 'b06764ed4d2a8b495f0ffd8721fa180d9b9de7caaa4bf5404d7a3ea17e159783'
            or prepared['status'] != 'passed' or prepared['plan_sha256'] != sha(plan_path)
            or prepared['test_targets_read']):
        raise ValueError('Cohort did not pass')
    manifest = STUDY / 'cohort-manifest-002.json'
    if sha(manifest) != prepared['manifest_sha256']:
        raise ValueError('Packed manifest changed')
    parent = ROOT / 'research/studies/strong9_followups/scale1e-2-continuation/outcome-001.json'
    if sha(parent) != '681c55c07e741b9a6aee76b81576d099a7fb2bc4228b13b950c9eba6721fbbe8':
        raise ValueError('Encoder decision changed')
    populations = plan['populations']['train']
    expected = 128 * populations['positions'] / populations['games']
    if round(expected) != 53028:
        raise ValueError('Position batch reference differs')
    configs = {}
    recipes = dict(source_cnn='strong19_source_muon_learning', cnn_adamw='strong19_train', transformer='strong19_train')
    bases = dict(source_cnn='strong19_source_muon/tpu-full-size-config-002.json',
                 cnn_adamw='strong19_train/tpu-real19-cnn-qualification-001.json',
                 transformer='strong19_scaling/tpu-transformer-execution-config-001.json')
    for arm, base in bases.items():
        c = json.loads((ROOT / 'research/studies' / base).read_text())
        c.update(steps=108, checkpoint_every=108, eval_every=9, log_every=1)
        c['dataset'] = dict(path=prepared['target'], manifest_sha256=prepared['manifest_sha256'],
            buckets=plan['buckets'], warmup_buckets=plan['buckets'],
            bucket_probabilities=[populations['buckets'][str(b)] / populations['games'] for b in plan['buckets']])
        c['evaluation'] = dict(games_per_bucket=200000, training_probe_games=128, run_test=False)
        c['training'].update(purpose='learning', chunk_frames=16, value_weight=.7)
        if arm == 'source_cnn':
            runtime = c['learner']['source_runtime']
            runtime.update(reference_batch_positions=round(expected), initial_samples=0,
                no_lr_warmup=False, effective_lr_scale=1., epochs=[[18] for _ in range(6)],
                print_every=1, lookahead=dict(k=6, alpha=.5))
        else:
            c['learner'].update(learning_rate=.001, end_learning_rate=.0003, warmup_steps=40,
                beta1=.9, beta2=.95, epsilon=1e-8, weight_decay=.01, max_grad_norm=1.)
        configs[arm] = c
    if configs['source_cnn']['model'] != configs['cnn_adamw']['model']:
        raise ValueError('CNN optimizer control changed architecture')
    for key in ('seed', 'dataset', 'evaluation', 'steps', 'eval_every', 'value_model'):
        if any(c[key] != configs['source_cnn'][key] for c in configs.values()):
            raise ValueError('Paired scientific input differs: ' + key)
    if configs['transformer']['model']['encoder_layer_scale'] != .01:
        raise ValueError('Confirmed encoder was not inherited')
    # Independent metadata-only replay, before any model or target is read.
    import numpy as np
    entries = {b: [] for b in plan['buckets']}; info = {}
    for index, row in enumerate(plan['records']):
        key = ('expert', index // 32, index % 32); info[key] = row
        if row['split'] == 'train':
            entries[512 if row['rows'] <= 512 else 768].append(key)
    seed = configs['source_cnn']['seed']
    samplers = [np.random.Generator(np.random.PCG64(seed + 1 + 104729 * rank)) for rank in range(4)]
    augment = [np.random.Generator(np.random.PCG64(seed + 400003 + 104729 * rank)) for rank in range(4)]
    bucket_rng = np.random.Generator(np.random.PCG64(seed + 9143))
    draws = []; positions = 0; slots = 0
    for turn in range(1, 109):
        bucket = plan['buckets'][turn - 1] if turn <= 2 else int(bucket_rng.choice(plan['buckets'],
            p=configs['source_cnn']['dataset']['bucket_probabilities']))
        rows = []; count = 0
        for rank, (rng, symmetry_rng) in enumerate(zip(samplers, augment)):
            chosen = [entries[bucket][int(i)] for i in rng.integers(len(entries[bucket]), size=32)]
            symmetries = symmetry_rng.integers(0, 8, 32).tolist()
            count += sum(info[e]['rows'] for e in chosen)
            rows.append(dict(jax_rank=rank, local_entries_sha256=hashlib.sha256(canonical_json(chosen)).hexdigest(),
                             local_symmetries=symmetries))
        positions += count; slots += 128 * bucket
        draws.append(dict(turn=turn, bucket=bucket, positions=count, cumulative_positions=positions, ranks=rows))
    paths = {}
    for arm, c in configs.items():
        path = STUDY / ('pilot-' + arm.replace('_', '-') + '-config-001.json')
        publish(path, c); paths[arm] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                           recipe='research/recipes/' + recipes[arm])
    replay_path = STUDY / 'pilot-draw-replay-001.json'
    publish(replay_path, dict(kind='joint19_pilot_draw_replay', status='prepared', seed=seed,
        plan_sha256=sha(plan_path), model_or_targets_read=False, draws=draws, total_positions=positions,
        padded_position_slots=slots, training_population_equivalent_passes=positions/populations['positions']))
    result = dict(kind='joint19_pilot_settings', status='prepared', created=time.time(),
        operator_sha256=sha(Path(__file__)), cohort_manifest_sha256=prepared['manifest_sha256'],
        cohort_result_sha256=sha(STUDY / 'cohort-result-002.json'), encoder_decision_sha256=sha(parent),
        configs=paths, draw_replay_sha256=sha(replay_path), position_exposures=positions,
        expected_batch_positions=expected, rounded_batch_reference=round(expected),
        training_population_equivalent_passes=positions/populations['positions'],
        full_size_gates_pending=True, data_staging_pending=True, learning_jobs_started=False,
        order=['source_cnn', 'cnn_adamw', 'transformer'],
        checkpoint_policy='Final checkpoint retained and replicated. Intermediate validation has no stored model arrays; no retrospective best-checkpoint claim.',
        scope='Prepared first-seed joint19 learnability pilot, not a launch receipt or completed scientific result. Source-derived batch/cadence/epoch transfer is explicit; no historical-checkpoint hyperparameter equivalence. Full-size gates and storage/data staging must close before registration/launch.')
    publish(STUDY / 'pilot-settings-001.json', result)
    print(json.dumps(dict(status='prepared', configs=list(paths), positions=positions,
        training_passes=positions/populations['positions'], settings_sha256=sha(STUDY / 'pilot-settings-001.json'))))


if __name__ == '__main__':
    main()
