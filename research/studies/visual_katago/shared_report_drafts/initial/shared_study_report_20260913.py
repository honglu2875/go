"""Regenerate the completed shared-encoder study summary from pinned evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def publish(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing summary differs: ' + str(path))
        return
    with path.open('xb') as f:
        f.write(data)
    path.chmod(0o444)


def main():
    comparison = 'runs/shared-token-c83cca0a/final-analysis/comparison.json'
    contrast = 'runs/shared-token-c83cca0a/lr-comparison.json'
    assert sha(ROOT / comparison) == 'a12a24c504b254f5ccebf60513e4973f41f33c33eeccd1fad48cc60da8c72ca8'
    assert sha(ROOT / contrast) == '74c4f522c395ed1f2a08d73c568aae715d79a81b483cf5abed512bf013b3e322'
    report, lr = read(comparison), read(contrast)
    assert report['status'] == lr['status'] == 'passed'
    for name, expected in report['input_files'].items():
        assert sha(ROOT / name) == expected, name
    rows = {r['label']: r for r in report['arms']}
    control, best = rows['Shared 1:1, control'], rows['Shared 1:1, higher LR']
    c128, cnn = rows['C128 encoder + transformer'], rows['KataGo-derived CNN']
    kl = lambda r: r['curve'][-1]['expert_kl']
    findings = {
        'lr_kl_reduction_fraction': 1 - kl(best) / kl(control),
        'best_kl_increase_over_c128_fraction': kl(best) / kl(c128) - 1,
        'best_kl_increase_over_cnn_fraction': kl(best) / kl(cnn) - 1,
        'learning_speedup_over_c128': c128['timing']['learning_seconds'] / best['timing']['learning_seconds'],
        'initial_decode_speedup_over_c128': c128['decode_median_ms'] / best['decode_median_ms'],
        'best_trained_decode_ms': best['trained_decode_median_ms'],
    }
    attempts = [
        'pod-20260913T113940Z-cb529274', 'pod-20260913T114714Z-21177ace',
        'pod-20260913T121525Z-ef30e672', 'pod-20260913T122153Z-1c27ad98',
        'pod-20260913T125036Z-596df0c3',
    ]
    closures = [read(f'runs/{a}/result.json') for a in attempts]
    assert all(x['status'] == 'passed' for x in closures)
    cost = sum(x['reserved_chip_hours'] for x in closures)
    paths = [comparison, contrast, str(Path(__file__).relative_to(ROOT)),
             'runs/shared-token-c83cca0a/final-analysis/manifest.json',
             'runs/shared-token-c83cca0a/lr6-refinement.json',
             'runs/shared-token-4761ce55/equal-refinement.json',
             'runs/shared-token-ba7333c3/double-trunk-refinement.json',
             'runs/shared-token-ba7333c3/allocation-comparison.json',
             'runs/shared-token-ba7333c3/lr-audit-preflight/negative-check.json']
    paths += [f'runs/{a}/result.json' for a in attempts]
    for name in ('shared_equal_cpu_budget_v2.json', 'shared_double_trunk_cpu_budget.json',
                 'shared_training_arithmetic_comparison.json', 'shared_cpu_tests.log',
                 'shared_equal_tpu_qualification.json', 'shared_double_trunk_tpu_qualification.json',
                 'shared_equal_learning_registration.json', 'shared_double_trunk_learning_registration.json',
                 'shared_equal_lr6_learning_registration.json', 'shared_equal_endpoint_review_001.json',
                 'shared_allocation_endpoint_review_001.json'):
        paths.append('research/studies/visual_katago/' + name)
    record = {
        'schema_version': 1, 'kind': 'completed_shared_encoder_ablation', 'date': '2026-09-13',
        'status': 'complete', 'training_runs': 3, 'full_shape_qualifications': 2,
        'comparison_operator_snapshot': report['operator_snapshot'], 'findings': findings,
        'attempts': attempts, 'recorded_attempt_chip_hours': cost,
        'input_files': {p: sha(ROOT / p) for p in paths},
        'scope': 'One-seed fixed-data learnability and warm neural execution; no production or Go-strength promotion.',
        'next': 'Round closed. A separate LR 1e-3 trial on 1:1 would test the promising LR direction; none is queued.',
    }
    lines = [
        'The approved one-board-token, shared-encoder study is complete: two allocation runs followed by one LR-only follow-up, each trained for 1,024 updates. All five TPU attempts, including two full-shape qualifications, passed.',
        '',
        f"The 1:1 allocation at LR 6e-4 is the best shared model tested. Doubling LR reduces its final validation KL by {findings['lr_kl_reduction_fraction']:.1%}. Relative to C128, it trains {findings['learning_speedup_over_c128']:.2f}x faster and its warm neural decode is {findings['initial_decode_speedup_over_c128']:.2f}x faster, but final KL is {findings['best_kl_increase_over_c128_fraction']:.1%} higher. The CNN remains the accuracy reference. Moving from 1:1 to 1:2 did not improve KL at LR 3e-4; the small top-1 difference went in the other direction.",
        '',
        '| Model | Parameters | Peak AdamW LR | Validation KL | Top-1 | Learning min | Warm decode ms |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['arms']:
        last = row['curve'][-1]
        lines.append(f"| {row['label']} | {row['parameters']:,} | {row['peak_lr']:.0e} | {last['expert_kl']:.6f} | {last['expert_top1']:.4f} | {row['timing']['learning_seconds']/60:.2f} | {row['decode_median_ms']:.2f} |")
    lines += [
        '',
        f"Decode times are initial-weight medians, matching the historical measurement protocol: global batch 128, 9x9 boards and 128 prior moves. The best model also passed cached/full inference checks after training at {best['trained_decode_median_ms']:.2f} ms. These are warm neural graphs, including the encoder and pending action; CPU rules/features, transfers and pointer reset are excluded. Learning time excludes compilation, evaluation and checkpoint writing. This does not measure full-rollout throughput or MFU.",
        '',
        '![Learning curves by update and measured learning time](../../../runs/shared-token-c83cca0a/final-analysis/learning.png)',
        '',
        'Both new models use one width-768 observation embedding and one action embedding per move. A causal transformer predicts action_t from the output at observation_t, before action_t is visible. There is one tied policy head and no behavior, value or return-conditioning objective in this ablation.',
        '',
        'The encoder starts with a stride-one 3x3, 22-to-768 convolution. Each shared residual block has a stride-one 3x3 depthwise convolution, affine LayerNorm, pointwise 768-to-3072, GELU, pointwise 3072-to-768, and learned channel scaling. LayerScale starts at 1e-6. The residual stream is BF16 at block boundaries, with full gradient accumulation through every shared pass. The connector applies 768-to-64 channel compression, flattens 9x9x64, and projects 5184-to-768 once. All new models are batch-normalization-free.',
        '',
        '| Allocation | Unique encoder blocks x passes | Temporal layers | Encoder + connector parameters | Temporal parameters | Complete decode GFLOPs/move |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for label, budget_file in [('Shared 1:1, control', 'shared_equal_cpu_budget_v2.json'),
                               ('Shared 1:2', 'shared_double_trunk_cpu_budget.json')]:
        row = rows[label]
        result = read(f"runs/{row['attempt']}/rank-0/artifacts/result.json")
        schema = result['model_schema']
        enc = sum(x['elements'] for x in schema if x['path'].startswith('encoder.'))
        temporal = sum(x['elements'] for x in schema if x['path'].startswith('blocks.'))
        budget = read('research/studies/visual_katago/' + budget_file)
        flops = budget['cases']['128']['analytical']['multiply_add_flops_per_move'] / 1e9
        c = row['model']
        lines.append(f"| {label} | {c['encoder_blocks']} x {c['encoder_passes']} | {c['layers']} | {enc:,} | {temporal:,} | {flops:.4f} |")
    lines += [
        '',
        'Both execute 48 spatial blocks. Temporal layers use width 768, 12 query heads, four KV heads, head width 64, SwiGLU width 2048, RMSNorm and RoPE. The fixed 9x9 connector does not claim weight compatibility with other board sizes. Encoder/connector work is included in the traced budget. Total parameters and complete decoding arithmetic are within 1% of the frozen CNN reference (232,431,872 parameters; 37.3483 GFLOPs/move). Training arithmetic is separately recorded and is about 19% below the CNN reference; decoding equality does not imply equal training work.',
        '',
        'The compiler-accounted decode memory is 1.404 GiB/device for 1:1, 1.610 GiB for 1:2, and 16.749 GiB for C128. This is arguments + outputs + temporaries - aliases, not a measured peak-memory trace. The learned residual-scale median magnitude rises from about 0.0080 at LR 3e-4 to 0.0205 at 6e-4. Parameters are learning beyond initialization; this alone does not establish that repeated passes improve prediction.',
        '',
        'The data are the same fixed weak native-MCTS teacher samples used in the earlier comparisons, not new self-play or KataGo expert labels. There are 9,466 training games / 836,486 positions, and 1,170 validation games / 102,339 positions. Each full run consumes exactly 11,469,333 position exposures, using 128 complete games per update and identical game/D4 draws. AdamW beta=(0.9,0.95), epsilon=1e-8, weight decay=0.01, gradient clip=1, 64-update warmup and cosine decay to 0.3x peak are retained. The CNN reference retains its existing FSON/Mish and training helper head; the new models use a single policy objective. All are compared on the main policy KL against the same validation targets.',
        '',
        'This is a single-seed, finite-horizon learnability comparison. The allocation experiment changes encoder uniqueness, sharing and temporal depth together; it does not isolate weight sharing. The LR contrast verifies identical numerical source, library/lock hashes, architecture, initial weights and samples. The test split remains closed, and these runs do not establish playing strength, convergence or RL improvement.',
        '',
        'Six CPU model tests cover causality, cached/full inference, cache guards, rematerialized gradients, accumulation across shared passes and the fixed-board/BF16 contract. Abstract schemas and traced arithmetic passed. Each allocation completed full-batch TPU qualification for all 128/256/384 training buckets. All full runs passed finite-update, exact-draw, replica, persistent-checkpoint and trained-cache audits. The LR audit also rejected an allocation change mislabeled as an LR intervention.',
        '',
        '| New full run | Frozen configuration | Independent audit |',
        '|---|---|---|',
    ]
    for label, audit in [
        ('Shared 1:1, control', 'runs/shared-token-4761ce55/equal-audit.json'),
        ('Shared 1:2', 'runs/shared-token-ba7333c3/double-trunk-audit.json'),
        ('Shared 1:1, higher LR', 'runs/shared-token-c83cca0a/equal-lr6-audit.json'),
    ]:
        row = rows[label]
        lines.append(f"| {label} | [{row['snapshot'][:8]}](../../../.gozero/snapshots/{row['snapshot']}/resolved_config.json) | [audit](../../../{audit}) |")
    lines += [
        '',
        f"All new checkpoints, including optimizer and RNG state, are on persistent storage. The five attempts used {cost:.3f} recorded attempt chip-hours; no further job is queued. Historical six-layer-encoder metrics and audits remain available, but its checkpoint arrays were lost in the earlier RAM-storage interruption, as documented in the preceding handoff.",
        '',
        'The next useful isolated experiment would test LR 1e-3 on the 1:1 allocation. The current evidence supports keeping its speed benefits as a research reference; it does not support production promotion or calling the LR optimal.',
        '',
        'Implementation: [cloneable recipe](../../recipes/single_board_shared/README.md). Evidence: [machine-readable study index](shared_encoder_study_20260913.json), [complete comparison](../../../runs/shared-token-c83cca0a/final-analysis/REPORT.md), [LR contrast](../../../runs/shared-token-c83cca0a/lr-comparison.json), [SVG figure](../../../runs/shared-token-c83cca0a/final-analysis/learning.svg). Regenerate this summary with `python3 research/studies/visual_katago/shared_study_report_20260913.py`.',
    ]
    markdown = ('\n'.join(lines) + '\n').encode()
    record['summary_sha256'] = hashlib.sha256(markdown).hexdigest()
    publish(HERE / 'shared_encoder_study_20260913.md', markdown)
    publish(HERE / 'shared_encoder_study_20260913.json',
            (json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
    print(json.dumps({'status': 'passed', 'findings': findings, 'attempt_chip_hours': cost}))


if __name__ == '__main__':
    main()
