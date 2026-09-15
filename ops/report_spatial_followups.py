"""Assemble the completed four-intervention study from audited immutable inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.snapshots import canonical_json, read_json, verify
from gozero.checkpoints import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True)
    p.add_argument('--registration', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    p.add_argument('--cnn-contrast', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    p.add_argument('--context-result', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    p.add_argument('--attention-result', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    p.add_argument('--attention-addendum', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    p.add_argument('--auxiliary-result', nargs=2, required=True, metavar=('PATH', 'SHA256'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE)
    inputs = {}; arms = {}; pairs = {}; executions = {}; replications = {}; common = None; draws = None
    checkpoint_inventory = {}
    def document(path, wanted=None):
        path = (root / path).resolve(); path.relative_to(root)
        actual = sha256(path)
        if wanted is not None and actual != wanted:
            raise ValueError('Input changed: ' + str(path))
        inputs[str(path.relative_to(root))] = actual
        return read_json(path)
    def checkpoint(attempt):
        if attempt in checkpoint_inventory: return
        record = document(root / 'runs' / attempt / 'rank-0/artifacts/result.json')
        saved = record['latest_checkpoint']; owner = Path(saved['owner_checkpoint_path'])
        owner.relative_to(root)
        group = document(Path(saved['path']).with_suffix('.group.json'), saved['group_sha256'])
        manifest = document(owner / 'manifest.json', group['host_manifests']['0'])
        files = manifest['files']
        for name in ('state.json', 'actors.json'):
            document(owner / name, files[name]['sha256'])
        array = owner / 'arrays.npz'
        item = {'original_owner': str(owner), 'manifest_sha256': group['host_manifests']['0'],
                'array_sha256': files['arrays.npz']['sha256'], 'array_bytes': files['arrays.npz']['bytes']}
        if array.exists():
            if array.is_symlink() or array.stat().st_mode & 0o222 or array.stat().st_size != item['array_bytes']:
                raise ValueError('Local checkpoint differs from its immutable storage contract')
            item.update(placement='local', checkpoint=str(owner))
        elif (owner / 'arrays.remote-replicas.json').exists():
            path = owner / 'arrays.remote-replicas.json'; locator = document(path)
            if (locator['array_sha256'] != item['array_sha256']
                    or locator['original_path'] != str(array.relative_to(root))):
                raise ValueError('Replica locator differs from checkpoint identity')
            item.update(placement='remote_complete_checkpoint', locator=str(path), locator_sha256=sha256(path),
                        hosts=locator['retained_hosts'], checkpoint=str(root / locator['retained_directory']))
        elif owner.with_suffix('.partitioned.json').exists():
            path = owner.with_suffix('.partitioned.json'); locator = document(path)
            descriptor = document(locator['descriptor'], locator['descriptor_sha256'])
            if descriptor['files']['arrays.npz'] != files['arrays.npz'] or descriptor['manifest_sha256'] != item['manifest_sha256']:
                raise ValueError('Partition composition differs from checkpoint identity')
            item.update(placement='persistent_byte_parts', locator=str(path), locator_sha256=sha256(path),
                        descriptor=locator['descriptor'], descriptor_sha256=locator['descriptor_sha256'],
                        parts=descriptor['parts'], restore_contract=locator['restore_contract'])
        else:
            raise ValueError('Checkpoint has neither local arrays nor an explicit persistent locator')
        item['verification_scope'] = 'Current metadata, location and local size/mode checks; full array hashes and ordinary-reader restoration are established by the prior pinned training/storage audits, not recomputed by this report.'
        checkpoint_inventory[attempt] = item
    def draw_signature(row):
        signature = {}
        for host in range(4):
            folder = root / 'runs' / row['attempt'] / f'rank-{host}/artifacts'
            record = document(folder / 'result.json')
            path = folder / 'metrics.jsonl'; inputs[str(path.relative_to(root))] = sha256(path)
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            signature[record['jax_rank']] = [{k: x[k] for k in
                ('turn', 'bucket', 'local_entries_sha256', 'local_symmetries', 'expert_positions')} for x in rows]
        return hashlib.sha256(canonical_json(signature)).hexdigest()
    def contrast(label, path, wanted, *, first_seed=True):
        nonlocal common, draws
        contrast_path = str((root / path).resolve())
        c = document(path, wanted)
        kinds = {'CNN main policy': 'cnn_main_only_contrast', 'Context readout': 'context_spatial_readout_contrast',
                 'Encoder attention': 'encoder_attention_contrast', 'First-pass auxiliary': 'first_pass_auxiliary_contrast'}
        if c['status'] != 'passed' or c['kind'] != kinds[label]:
            raise ValueError('Contrast integrity failed')
        r = document(c['comparison'], c['comparison_sha256'])
        if r['status'] != 'passed' or not r['all_game_and_augmentation_draws_equal']:
            raise ValueError('Comparison integrity failed')
        for name, digest in r['input_files'].items():
            path = root / name
            if sha256(path) != digest: raise ValueError('Audited evidence changed: ' + name)
            inputs[str(path.relative_to(root))] = digest
        by_label = {x['label']: x for x in r['arms']}
        control = by_label[c['control']] if 'control' in c else r['arms'][0]
        candidate = by_label[c['candidate']] if 'candidate' in c else r['arms'][1]
        for row in (control, candidate): checkpoint(row['attempt'])
        if first_seed:
            settings = {**r['common_non_lr_settings'], 'end_to_peak_lr': r['common_end_to_peak_lr']}
            if common is None: common = settings
            elif common != settings: raise ValueError('First-seed training settings differ')
            for row in (control, candidate):
                source = root / '.gozero/snapshots' / row['snapshot']; verify(source)
                inputs[str((source / 'manifest.json').relative_to(root))] = sha256(source / 'manifest.json')
                if row['snapshot'] in arms: continue
                if row['curve'][-1]['turn'] != 1024: raise ValueError('Incomplete first-seed learning')
                if arms:
                    validation = lambda item: [(x['turn'],x['validation_ids_sha256'],x['expert_count']) for x in item['curve']]
                    if validation(row) != validation(next(iter(arms.values()))):
                        raise ValueError('Validation populations differ across interventions')
                signature = draw_signature(row)
                if draws is None: draws = signature
                elif draws != signature: raise ValueError('First-seed actual draws differ across interventions')
                if row['peak_lr'] != .001 or row['position_exposures'] != 11469333:
                    raise ValueError('Unexpected first-seed exposure/LR contract')
                record = document(root / 'runs' / row['attempt'] / 'rank-0/artifacts/result.json')
                profile = record.get('decode_profile')
                final = record['validation_history'][-1]['metrics']
                phases = {phase: {'count': final['phase_' + phase + '_count'], 'kl': final['phase_' + phase + '_kl']}
                          for phase in ('0_16', '16_64', '64_128', '128_256', '256_2048')}
                if (sum(x['count'] for x in phases.values()) != final['expert_count']
                        or abs(sum(x['count'] * x['kl'] for x in phases.values()) / final['expert_count'] - final['expert_kl']) > 1e-6):
                    raise ValueError('Phase metrics do not reproduce the full validation population/KL')
                arms[row['snapshot']] = {**row, 'decode_gflops_per_move':
                    profile['jaxpr']['counts']['multiply_add_flops'] / profile['batch_size'] / 1e9 if profile else None,
                    'final_phase_metrics': phases,
                    'historical_profile_procedure': ('Repeated first-board input with automatic SPMD, initialization weights only.'
                        if row['model']['architecture'] == 'katago_nested_policy' else
                        'Real full-history prefixes, explicit shard_map, observed donated KV outputs; initial and trained weights.')}
        result = {'contrast': contrast_path, 'contrast_sha256': wanted,
                  'control': control, 'candidate': candidate, 'scientific_result': c}
        if first_seed: pairs[label] = result
        return result
    registration = document(*a.registration)
    attention_addendum = document(*a.attention_addendum)
    contrast('CNN main policy', *a.cnn_contrast)
    for label, pin in [('Context readout', a.context_result), ('Encoder attention', a.attention_result),
                       ('First-pass auxiliary', a.auxiliary_result)]:
        execution = document(*pin)
        if execution['status'] != 'passed':
            raise ValueError('Intervention has not completed and passed its audits: ' + label)
        executions[label] = execution
        launch = document((root / pin[0]).parent / 'launch.json')
        if label == 'First-pass auxiliary':
            budget_pin = launch['specification']['cpu_budget']
            auxiliary_budget = document(budget_pin['path'], budget_pin['sha256'])
        if label == 'Encoder attention':
            launch = document((root / pin[0]).parent / 'launch.json')
            followup = document(launch['registration'], execution['registration_sha256'])
            if not any(x['sha256'] == a.attention_addendum[1] for x in followup['prerequisite_evidence']):
                raise ValueError('Attention follow-up does not pin the adaptive addendum')
            if not execution.get('exploratory_followup') or followup['minimum_replicated_quality_gain'] != .05:
                raise ValueError('Attention quality follow-up identity differs')
        contrast(label, execution['first_contrast'], execution['first_contrast_sha256'])
        folder = (root / pin[0]).parent; replication = folder / 'replication/replication.json'
        if replication.exists():
            r = document(replication)
            if r['status'] != 'passed': raise ValueError('Replication audit failed')
            for name, digest in r['input_files'].items():
                path = root / name
                if sha256(path) != digest: raise ValueError('Replication evidence changed')
                inputs[str(path.relative_to(root))] = digest
            for pair in r['pairs']:
                contrast(label, pair['contrast'], pair['contrast_sha256'], first_seed=False)
            replications[label] = r
    if not executions['Context readout']['replicated_gain']:
        raise ValueError('This study report expects the reviewed replicated context parent')
    for row in arms.values():
        if row['decode_gflops_per_move'] is None:
            if row['model']['architecture'] != 'katago_nested_policy':
                raise ValueError('Missing non-CNN decoding arithmetic')
            row['decode_gflops_per_move'] = auxiliary_budget['cnn']['analytical']['multiply_add_flops_per_batch'] / 128 / 1e9
    best = 'Context readout'
    for label in ('Encoder attention', 'First-pass auxiliary'):
        if executions[label]['replicated_gain'] or (label == 'Encoder attention' and executions[label].get('replicated_quality_gain')):
            best = label
    new_attempts = []
    for path in sorted((root / 'runs').glob('pod-*/launch.json')):
        launch = read_json(path)
        if launch['start_unix_time'] < registration['created_unix']: continue
        closed = path.parent / 'result.json'
        if not closed.exists(): raise ValueError('A new study attempt is still open')
        r = document(closed)
        if r['snapshot_id'] != launch['snapshot_id'] or r['attempt_id'] != path.parent.name:
            raise ValueError('Attempt identity differs')
        expected = (r['end_unix_time'] - r['start_unix_time']) * r['reserved_chips'] / 3600
        if abs(expected - r['reserved_chip_hours']) > 1e-9: raise ValueError('Attempt cost arithmetic differs')
        if r['status'] == 'passed': checkpoint(r['attempt_id'])
        new_attempts.append({k: r[k] for k in ('attempt_id', 'snapshot_id', 'status', 'elapsed_seconds', 'reserved_chip_hours')})
    result = {'status': 'passed', 'kind': 'completed_spatial_followups_study', 'operator_snapshot': SOURCE.name,
        'created_unix': time.time(), 'registration_sha256': a.registration[1], 'common_first_seed_settings': common,
        'all_first_seed_actual_draws_sha256': draws, 'pairs': pairs, 'replications': replications,
        'strongest_replicated_transformer': best, 'first_seed_arms': list(arms.values()),
        'selection_scope': 'Strongest research model with replicated validation improvement; attention uses the explicitly adaptive quality-only follow-up, with the original latency failure retained.',
        'attention_adaptive_addendum': attention_addendum,
        'execution_outcomes': {label: {key: r.get(key) for key in ('replicated_gain', 'replicated_quality_gain', 'replicated_draft_utility', 'exploratory_followup')}
                               for label, r in executions.items()},
        'new_attempts': new_attempts, 'new_attempt_chip_hours': sum(x['reserved_chip_hours'] for x in new_attempts),
        'input_files': inputs,
        'checkpoint_inventory': checkpoint_inventory,
        'scope': 'Same fixed weak native-MCTS teacher corpus and closed test split. Quality is validation-policy learnability, not Go strength or RL sample efficiency. Neural latency excludes rules, features, transfers and search; arithmetic is not measured MFU. Learning clocks exclude compile/evaluation/checkpoint/engineering. Actual allocation cost is not inferred from attempt chip-hours.'}
    a.output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg'); matplotlib.rcParams['svg.hashsalt'] = SOURCE.name
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for row in arms.values():
        curve = row['curve'][1:]
        for ax, x in zip(axes, ([x['turn'] for x in curve], [x['learning_seconds'] / 60 for x in curve])):
            ax.plot(x, [r['expert_kl'] for r in curve], marker='o', markersize=3, linewidth=1.5, label=row['label'])
    axes[0].set_xlabel('Optimizer updates'); axes[1].set_xlabel('Learning minutes')
    for ax in axes: ax.set_ylabel('Validation KL'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Fixed-data 9×9 policy learning: all first-seed outcomes')
    fig.savefig(a.output / 'learning.png', dpi=170)
    fig.savefig(a.output / 'learning.svg', metadata={'Date': None}); plt.close(fig)
    fmt = lambda x: '—' if x is None else f'{x:.2f}'
    lines = [f"The strongest research transformer with replicated validation improvement is **{best}**.", '',
        '| First-seed model | Parameters | GFLOPs/move | Final KL | Top-1 | Learning min | Initial / trained neural ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in arms.values():
        final = row['curve'][-1]
        lines.append(f"| {row['label']} | {row['parameters']:,} | {row['decode_gflops_per_move']:.6f} | {final['expert_kl']:.6f} | {final['expert_top1']:.4f} | {row['timing']['learning_seconds']/60:.2f} | {fmt(row['decode_median_ms'])} / {fmt(row['trained_decode_median_ms'])} |")
    lines += ['', 'All first-seed runs used exactly the same game and D4 draws, 11,469,333 live-position exposures and LR schedule. Different seeds have independent initializations and draws. GFLOPs count complete logical decoding at batch 128, FMA=2, including the encoder; transformers include 128 past moves and cache advancement. Parameter counts include the CNN control’s 297,088 retained but unused helper slots.',
        '', 'The neural timings are historical profiles with different procedures: CNN initialization profiles repeat one board and use automatic SPMD; transformer profiles use real full-history prefixes, explicit shard_map and observed donated KV outputs. Both use board 9×9 and global batch 128 with dynamic runtime inputs. No trained CNN profile was recorded. These numbers are not a controlled CNN-versus-transformer latency comparison. Paired transformer interventions retain the same profiling procedure.', '',
        '| Intervention | First-seed relative KL reduction | Initial decode ratio | Primary screen |',
        '|---|---:|---:|---|']
    for label, pair in pairs.items():
        r = pair['scientific_result']; screen = r.get('replication_screen_passed')
        lines.append(f"| {label} | {r['relative_endpoint_kl_improvement']:.2%} | {fmt(r.get('warm_decode_latency_ratio'))} | {'n/a: control' if screen is None else screen} |")
    for label, r in replications.items():
        effects = ', '.join(f"{x['relative_kl_improvement']:.2%}" for x in r['pairs'])
        lines += ['', f"{label}: paired KL effects {effects}; mean control/candidate KL {r['mean_control_kl']:.6f}/{r['mean_candidate_kl']:.6f}. Both primary screens passed: {r['each_seed_passed_one_percent_latency_screen']}."]
    lines += ['', 'The attention model narrowly failed the original first-seed latency screen. After reviewing that endpoint, an explicit exploratory addendum selected one independent second seed, requiring at least 5% paired KL improvement in each seed. The original joint quality/latency classifications remain unchanged. Its research-parent selection is therefore adaptive and does not establish a production latency or MFU improvement.',
              '', f"Attention quality-only improvement replicated: {executions['Encoder attention'].get('replicated_quality_gain', False)}. The CNN reference has only one matching seed in this study."]
    auxiliary = pairs['First-pass auxiliary']['scientific_result']
    lines += ['', 'The following diagnostic uses the same first-seed endpoint, split by zero-based move index. It was examined after seeing the endpoint, rather than used for selecting runs.',
              '', '| Model | Moves 0–15 | 16–63 | 64–127 | 128–255 | 256+ |',
              '|---|---:|---:|---:|---:|---:|']
    for row in arms.values():
        lines.append('| ' + row['label'] + ' | ' + ' | '.join(f"{x['kl']:.4f}" for x in row['final_phase_metrics'].values()) + ' |')
    lines += ['', 'Position counts in these bins are 18,720 / 53,317 / 21,994 / 8,211 / 97. The last bin is too small to support a broad conclusion.']
    lines += ['', '| First-pass draft metric | Paired parent | Auxiliary trained |',
        '|---|---:|---:|',
        f"| Teacher KL | {auxiliary['parent_draft_kl']:.6f} | {auxiliary['trained_draft_kl']:.6f} |",
        f"| Probability overlap with own full policy | {auxiliary['parent_draft_full_policy_overlap']:.4f} | {auxiliary['trained_draft_full_policy_overlap']:.4f} |",
        f"| Greedy agreement with own full policy | {auxiliary['parent_draft_full_argmax_agreement']:.4f} | {auxiliary['trained_draft_full_argmax_agreement']:.4f} |",
        f"| Warm draft decode ms | {auxiliary['parent_draft_decode_ms']:.2f} | {auxiliary['trained_draft_decode_ms']:.2f} |",
        '', f"Auxiliary draft-utility screen passed: {auxiliary['draft_utility_screen_passed']}. Its trained draft/full neural latency ratio is {auxiliary['trained_draft_over_full_neural_latency']:.3f}. Additional training arithmetic is retained in the pinned CPU qualifications; actual training time appears above.",
        '', 'The draft uses full-pass historical boards/actions and a one-pass current board. This is teacher-forced one-step draft evaluation. A draft cache must be verified or recomputed before replacing full-history state; these metrics do not measure multi-step speculative acceptance.',
        '', 'The CNN control jointly removes the helper objective and its batch-statistics path under the same optimizer. It does not separate those effects or retune the learning rate. The context readout mixes the temporal representation with current uncompressed spatial features; its gain does not isolate history from the current board’s global representation.',
        '', 'Training uses 9,466 games / 836,486 positions; validation uses 1,170 games / 102,339 positions. Labels are from the fixed weak native-MCTS teacher, not KataGo expert checkpoints. Every candidate includes its encoder in the approximately 233M-parameter and complete-decoding budget. Test data remain closed.',
        '', f"This round recorded {len(new_attempts)} TPU attempts and {result['new_attempt_chip_hours']:.6f} attempt chip-hours, including failed attempts. The separate reservation ledger also counts engineering and idle allocation time; billing is unknown.",
        '', result['scope'], '', '![All first-seed learning curves](learning.png)', '',
        'Exact configurations, every paired contrast, both replication outcomes, input hashes and attempt identities are in [result.json](result.json).']
    (a.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (a.output / 'result.json').write_bytes(canonical_json(result))
    files = {p.name: sha256(p) for p in a.output.iterdir() if p.is_file()}
    (a.output / 'manifest.json').write_bytes(canonical_json({'operator_snapshot': SOURCE.name, 'files': files}))
    for path in a.output.iterdir(): path.chmod(0o444)
    print(json.dumps({'status': 'passed', 'best': best, 'output': str(a.output),
                      'attempt_chip_hours': result['new_attempt_chip_hours']}), flush=True)


if __name__ == '__main__':
    main()
