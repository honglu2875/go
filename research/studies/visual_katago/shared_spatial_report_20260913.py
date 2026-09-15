"""Publish the completed LR/spatial-readout round from pinned, audited evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INPUTS = HERE / 'shared_spatial_report_inputs_20260913.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def publish(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing publication differs: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(data)
    path.chmod(0o444)


def main():
    spec = read(INPUTS)
    inputs = dict(spec['input_files'])
    for name, expected in inputs.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Pinned evidence changed: ' + name)
    replication = read(spec['replication'])
    history = read(spec['historical_comparison'])
    lr = read(spec['lr_review'])
    budget = read(spec['cpu_budget'])
    qualification = read(spec['tpu_qualification'])
    for result in (replication, history, lr, budget, qualification):
        if result['status'] != 'passed':
            raise ValueError('Evidence did not pass its audit')
    # Follow the existing audit chain rather than trusting a copied endpoint.
    for result in (replication, history):
        for name, expected in result['input_files'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Audited evidence changed: ' + name)
            inputs[name] = expected
    pairs = replication['pairs']
    if len(pairs) != 2 or pairs[0]['seed'] == pairs[1]['seed']:
        raise ValueError('Expected both independent paired seeds')
    rows = {row['label']: row for row in history['arms']}
    spatial, control = rows['Spatial readout'], rows['One-token control']
    cnn, c128, six = (rows[label] for label in
                      ('CNN', 'C128 encoder transformer', 'Six-layer encoder transformer'))
    if (spatial['snapshot'] != pairs[0]['candidate']['snapshot']
            or control['snapshot'] != pairs[0]['control']['snapshot']):
        raise ValueError('Historical context is not the first paired seed')
    kl = lambda row: row['curve'][-1]['expert_kl']
    comparable_cnn = next(row for row in cnn['curve']
                          if row['expert_kl'] <= kl(spatial))
    attempts = [qualification['attempt']]
    attempts.extend(pair[role]['attempt'] for pair in pairs
                    for role in ('control', 'candidate'))
    if len(set(attempts)) != 5:
        raise ValueError('Expected four full runs and one qualification')
    closures = []
    for attempt in attempts:
        name = f'runs/{attempt}/result.json'
        closure = read(name)
        if closure['status'] != 'passed':
            raise ValueError('Unsuccessful or unclosed attempt: ' + attempt)
        inputs[name] = sha(ROOT / name)
        closures.append(closure)
    cost = sum(row['reserved_chip_hours'] for row in closures)
    diagnostics = [read(path) for path in spec['pass_diagnostics']]
    if any(d['status'] != 'passed' for d in diagnostics):
        raise ValueError('Incomplete pass diagnostic')
    if diagnostics[0]['selection'] != diagnostics[1]['selection']:
        raise ValueError('Pass diagnostics used different positions')
    findings = {
        'lr_6e4_to_1e3_relative_kl_reduction': 1-lr['candidates']['0.001']/lr['candidates']['0.0006'],
        'spatial_direction_replicated': replication['direction_replicated'],
        'mean_control_kl': replication['mean_control_kl'],
        'mean_spatial_kl': replication['mean_candidate_kl'],
        'relative_improvement_of_mean_kl': replication['relative_improvement_of_mean_kl'],
        'seed1_spatial_kl_increase_over_cnn': kl(spatial)/kl(cnn)-1,
        'seed1_learning_speedup_over_six_layer': six['timing']['learning_seconds']/spatial['timing']['learning_seconds'],
        'seed1_initial_decode_speedup_over_six_layer': six['decode_median_ms']/spatial['decode_median_ms'],
        'cnn_first_logged_checkpoint_at_or_below_spatial_endpoint': comparable_cnn,
    }
    direction = 'repeated in both seeds' if replication['direction_replicated'] else 'did not repeat in both seeds'
    lines = [
        f"The LR follow-up and minimal spatial-readout experiment are complete, including a fresh paired second seed. The spatial correction's improvement {direction}. Four full 1,024-update runs and one full-shape qualification passed their audits.",
        '',
        f"Increasing the one-token control's peak AdamW LR from 6e-4 to 1e-3 reduced first-seed validation KL from {lr['candidates']['0.0006']:.6f} to {lr['candidates']['0.001']:.6f} ({findings['lr_6e4_to_1e3_relative_kl_reduction']:.2%}). LR 1e-3 is the best of the three tested rates at this horizon; both spatial pairs use it. No larger LR or longer-horizon optimum is established.",
        '',
        '| Seed | Position exposures per arm | Control KL | Spatial KL | Relative KL reduction | Initial decode ratio |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for pair in pairs:
        lines.append(f"| {pair['seed']} | {pair['control']['position_exposures']:,} | {kl(pair['control']):.6f} | {kl(pair['candidate']):.6f} | {pair['relative_kl_improvement']:.2%} | {pair['warm_decode_latency_ratio']:.3f} |")
    lines += [
        '',
        f"Mean endpoint KL is {replication['mean_control_kl']:.6f} for control and {replication['mean_candidate_kl']:.6f} for spatial, a {replication['relative_improvement_of_mean_kl']:.2%} relative reduction. Direction replicated: {replication['direction_replicated']}. Both pairs satisfy the exploratory threshold of at least 1% KL improvement and no more than 15% initial decode latency regression: {replication['each_seed_passed_one_percent_latency_screen']}. Two seeds support reporting repeatability of direction, not precise statistical confidence. Both results are retained regardless of outcome.",
        '',
        '![Both paired seeds](../../../runs/shared-spatial-replication/two-seed/learning.png)',
        '',
        'The only model intervention adds `F[i] @ w + b` to each intersection logit, using the final encoder feature map F of shape 9x9x768. The same weight [768,1] and scalar bias are shared across all intersections and start at zero. Pass is unchanged. This adds 769 parameters and 124,416 multiply-add FLOPs per move, reuses the existing feature map, and retains one policy distribution and one training objective. There is no behavior or value objective. Common initial arrays and initial policies match within each pair; optimizer, data and D4 draws match exactly.',
        '',
        '```mermaid',
        'flowchart LR',
        '    B["Current board: 9 x 9 x 22"] --> E["Shared encoder: 24 blocks x 2 passes"]',
        '    E --> F["Spatial features: 9 x 9 x 768"]',
        '    F --> C["Connector: one board token"]',
        '    C --> T["18-layer causal transformer"]',
        '    H["Past board tokens and actions"] --> T',
        '    T --> L["Existing 82 policy logits"]',
        '    F --> S["Shared 768-to-1 projection"]',
        '    S --> A["Add to 81 intersection logits; pass unchanged"]',
        '    L --> A',
        '    A --> P["Single policy distribution"]',
        '```',
        '',
        'The original encoder and temporal network remain unchanged: a stride-one 3x3, 22-to-768 stem; 24 unique residual blocks executed twice; then channel compression 768-to-64 and a flattened 5184-to-768 projection producing one historical board token. Each encoder block has depthwise 3x3 convolution, LayerNorm, pointwise 768-to-3072, GELU, pointwise 3072-to-768, and learned channel scaling initialized at 1e-6. The 18-layer causal temporal transformer has width 768, 12 query heads, four KV heads, head width 64, SwiGLU width 2048, RMSNorm and RoPE. An observation token predicts its action before that action is visible. Everything is plain JAX and batch-normalization-free.',
        '',
        f"The spatial model has {spatial['parameters']:,} parameters: 117,780,544 encoder/connector, 113,273,856 temporal, and 65,281 other input/readout parameters. Complete cached decoding is {budget['cases']['128']['analytical']['multiply_add_flops_per_move']/1e9:.6f} GFLOPs per move at 9x9, batch 128 and 128 prior moves, versus {budget['cnn']['analytical']['multiply_add_flops_per_move']/1e9:.6f} for the frozen CNN. Parameter, dense-FLOP and unit-cost floating-operation differences all stay within 1%. Training/rematerialization arithmetic is recorded separately. Equal decoding arithmetic does not imply equal training work or equal hardware utilization.",
        '',
        'The historical comparison uses the first seed and identical actual samples, while retaining each architecture\'s previously selected LR:',
        '',
        '| Model | Parameters | Peak LR | Final KL | Top-1 | Learning minutes | Initial warm decode ms |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in history['arms']:
        endpoint = row['curve'][-1]
        lines.append(f"| {row['label']} | {row['parameters']:,} | {row['peak_lr']:.0e} | {kl(row):.6f} | {endpoint['expert_top1']:.4f} | {row['timing']['learning_seconds']/60:.2f} | {row['decode_median_ms']:.2f} |")
    lines += [
        '',
        f"The spatial result is close to the six-layer encoder's endpoint, with {findings['seed1_learning_speedup_over_six_layer']:.2f}x faster execution of the same training schedule and {findings['seed1_initial_decode_speedup_over_six_layer']:.2f}x faster initial warm decoding. It remains {findings['seed1_spatial_kl_increase_over_cnn']:.1%} worse than the CNN in final KL. The CNN already reaches {comparable_cnn['expert_kl']:.6f} at update {comparable_cnn['turn']} after {comparable_cnn['learning_seconds']/60:.2f} learning minutes, so this round does not establish better accuracy per training minute than CNN.",
        '',
        '![Historical context by updates and learning time](../../../runs/shared-spatial-fff3608e/historical-analysis-v2/learning.png)',
        '',
        'The CNN is the existing KataGo-derived nested CNN with its BN-free FSON/Mish implementation and training helper retained. This table compares main-policy KL; it is not a new exact reproduction of KataGo training. Initial decode timing is used for historical consistency. Trained-weight decode medians are included in the paired evidence. Warm neural timings include the encoder and pending action but exclude CPU rules/features, transfers and pointer reset; they are not complete rollout throughput or MFU. Learning minutes exclude compilation, evaluation and checkpoint writing.',
        '',
        'Endpoint effects by game phase, computed over the same full validation population in each pair:',
        '',
        '| Moves, zero-based | Validation positions | Spatial minus control KL, seed 1 | Spatial minus control KL, seed 2 |',
        '|---|---:|---:|---:|',
    ]
    for start, end in ((0,16),(16,64),(64,128),(128,256),(256,2048)):
        key = f'phase_{start}_{end}'
        differences = [pair['candidate']['curve'][-1][key+'_kl']-pair['control']['curve'][-1][key+'_kl'] for pair in pairs]
        count = int(pairs[0]['control']['curve'][-1][key+'_count'])
        lines.append(f"| [{start}, {end}) | {count:,} | {differences[0]:+.6f} | {differences[1]:+.6f} |")
    lines += [
        '',
        'The late tail has only 97 positions and is not a reliable basis for selecting an architecture. A successful spatial bypass supports retaining this readout as a research candidate. It does not isolate whether channel compression, one-token projection or temporal processing is the limiting stage, or separate spatial information from the correction\'s effect on pass calibration.',
        '',
        'A separate read-only CPU diagnostic compared one versus two encoder passes on the current board, preserving the exact two-pass history cache at 64 prior moves. The same eight validation positions were used for both trained first-seed models:',
        '',
        '| Model | Greedy agreement with two passes | Mean probability overlap | One-pass teacher KL | Two-pass teacher KL |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, result in zip(('Control','Spatial'), diagnostics):
        lines.append(f"| {label} | {result['greedy_agreement']:.1%} | {result['mean_one_step_policy_overlap']:.1%} | {result['metrics']['1']['mean_teacher_kl']:.4f} | {result['metrics']['2']['mean_teacher_kl']:.4f} |")
    lines += [
        '',
        'This is an eight-position compatibility diagnostic, not full validation, a draft-model training experiment, measured multi-step acceptance, MCTS equivalence or a TPU speed test. It motivates training an early-exit objective before claiming that dropping a pass provides a useful speculative draft.',
        '',
        'Training uses the same fixed weak native-MCTS teacher dataset as earlier studies, not KataGo expert labels or architecture-specific self-play. There are 9,466 training games / 836,486 positions and 1,170 validation games / 102,339 positions; the test split remains closed. Each update draws 128 complete games across four hosts. AdamW uses beta=(0.9,0.95), epsilon=1e-8, weight decay 0.01, clipping at 1, 64 warmup updates and cosine decay to 0.3 of peak LR. Initialization and sample sequences differ across seeds; each pair shares its actual draws and common initial weights. There is no playing-strength, convergence or RL-improvement claim.',
        '',
        'Nine CPU tests passed, covering causal and cached execution, nonzero spatial corrections, inactive/stale cache guards, shared/rematerialized gradients, zero-initialized equivalence and location/pass behavior. Full-shape TPU qualification covered global batch 128 and all 128/256/384 training buckets. Every full run passed finite-update, exact-sample, replica, persistent-checkpoint and trained-cache audits. Negative audit checks rejected a mislabeled LR contrast and a repeated seed presented as replication.',
        '',
        f"The five TPU attempts used {cost:.3f} recorded attempt chip-hours, including launch/compile/checkpoint overhead. All four new full checkpoints are retained on persistent storage. Qualification arrays have two hash-verified persistent remote replicas, with a restore receipt and locator; temporary RAM was used only for validation. Previous immutable study artifacts are preserved. This round is closed and no further TPU job is queued.",
        '',
        'The next architecture candidate is spatial attention inside the encoder before compression, retaining one historical board token and the local readout. A four-token connector is a separate alternative. Both have only analytic budget proposals so far: full graph FLOPs, padding, memory, causality and runtime must be qualified before either is trained. Neither proposal nor early-exit/speculative training was silently added to this round.',
        '',
        'Implementation: [cloneable recipe](../../recipes/single_board_spatial/README.md). Evidence: [study index](shared_spatial_study_20260913.json), [paired replication](../../../runs/shared-spatial-replication/two-seed/REPORT.md), [historical comparison](../../../runs/shared-spatial-fff3608e/historical-analysis-v2/REPORT.md), [LR selection](shared_spatial_lr_endpoint_review.json), [spatial-attention proposal](shared_spatial_attention_encoder_feasibility.json), [four-token proposal](shared_spatial_four_token_feasibility.json). Regenerate with `.venv/bin/python -B research/studies/visual_katago/shared_spatial_report_20260913.py`.',
    ]
    inputs[str(INPUTS.relative_to(ROOT))] = sha(INPUTS)
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    markdown = ('\n'.join(lines)+'\n').encode()
    record = {
        'schema_version': 1, 'kind': 'completed_shared_spatial_readout_study',
        'date': '2026-09-13', 'status': 'complete', 'findings': findings,
        'training_runs': 4, 'full_shape_qualifications': 1, 'attempts': attempts,
        'recorded_attempt_chip_hours': cost, 'pairs': pairs,
        'input_files': inputs, 'summary_sha256': hashlib.sha256(markdown).hexdigest(),
        'scope': 'Two paired fixed-data seeds; historical context uses seed 1. No Go-strength or production promotion.',
        'next': 'Round closed. Review spatial-attention and four-token analytic alternatives before any further training.',
    }
    publish(HERE/'shared_spatial_study_20260913.md', markdown)
    publish(HERE/'shared_spatial_study_20260913.json',
            (json.dumps(record,sort_keys=True,separators=(',',':'))+'\n').encode())
    print(json.dumps({'status':'passed','findings':findings,'attempt_chip_hours':cost}))


if __name__ == '__main__':
    main()
