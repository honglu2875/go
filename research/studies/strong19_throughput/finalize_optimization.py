"""Select only an audited runtime; keep historical configurations untouched."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    report_path = STUDY / 'VARIED_001.json'
    report = read(report_path)
    completion = read(STUDY / 'varied-completion-001.json')
    if (completion['status'] != 'passed' or completion['report_sha256'] != sha(report_path)
            or report['status'] != 'passed' or not report['selected_for_research']):
        raise ValueError('No audited runtime qualified for selection')
    for relative, expected in report['input_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Audited evidence changed: ' + relative)
    reg = read(STUDY / 'varied-registration-001.json')
    recipe = ROOT / 'research/recipes/strong19_padding_varied'
    for name, expected in {**reg['qualified_sources'], **reg['driver_sources']}.items():
        if sha(recipe / name) != expected:
            raise ValueError('Qualified working source changed: ' + name)
    config = read(STUDY / 'varied-config-001.json')['reference_config']
    original = json.loads(json.dumps(config))
    config['training']['skip_padding'] = True
    check = json.loads(json.dumps(config))
    del check['training']['skip_padding']
    if check != original:
        raise ValueError('Scientific configuration changed')
    config_path = STUDY / 'selected-training-config-001.json'
    write(config_path, config)
    folder = ROOT / 'runs' / report['attempt']
    comparisons = []
    for host in range(4):
        for path in (folder / f'rank-{host}/artifacts').glob('conditional_skip-vs-original_dense-update-*.json'):
            comparisons.append(read(path))
    if len(comparisons) != 16:
        raise ValueError('Incomplete drift evidence')
    drift = {part: max(v['relative_l2'] for c in comparisons for k, v in c['groups'].items()
                       if k.split('/')[0] == part) for part in ('params', 'first', 'second')}
    drift.update(min_first_moment_cosine=min(v['cosine'] for c in comparisons for k, v in c['groups'].items()
                                           if k.split('/')[0] == 'first' and v['cosine'] is not None),
                 maximum_parameter_abs=max(c['groups']['params']['max_abs'] for c in comparisons),
                 maximum_old_coordinate_failures=max(c['groups']['all']['outside_screen001_tolerance'] for c in comparisons))
    attempts = ['pod-20260920T233339Z-9d392c05', 'pod-20260921T000933Z-6cb573ad', report['attempt']]
    chip_hours = sum(read(ROOT / 'runs' / a / 'result.json')['reserved_chip_hours'] for a in attempts)
    result = dict(status='selected_for_research', created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  recipe=str(recipe.relative_to(ROOT)), qualification_snapshot=report['snapshot'],
                  configuration=str(config_path.relative_to(ROOT)), configuration_sha256=sha(config_path),
                  only_training_configuration_change={'training.skip_padding': True},
                  review_sha256=sha(report_path), drift=drift, comparisons=report['comparisons'],
                  original_screen_status='rejected', bounded_attempt_chip_hours=chip_hours,
                  source_binding=reg['qualified_sources'], training_started=False,
                  limitations='Exact padding invariance within the conditional executable; bounded but nonzero drift from historical compiler. '
                              'Two varied samples per length. No long-run learning, strength or achieved MFU claim.')
    write(STUDY / 'selection-001.json', result)
    lines = ['# Fixed-model transformer throughput optimization', '',
             '**Selected for subsequent research runs:** skip encoder chunks containing only padded frames. '
             'The 232,011,540-parameter architecture, full histories, losses, optimizer and sampled examples are unchanged.', '',
             '| Sequence length | Original update | Optimized update | Less time | Compiled peak, original → optimized |',
             '|---|---:|---:|---:|---:|']
    for bucket in (512, 768):
        old = next(r for r in report['cases'] if r['variant'] == 'original_dense' and r['bucket'] == bucket)
        new = next(r for r in report['cases'] if r['variant'] == 'conditional_skip' and r['bucket'] == bucket)
        gain = report['comparisons'][str(bucket)]['latency_reduction']
        lines.append(f"| {bucket} | {old['median_seconds']:.3f} s | {new['median_seconds']:.3f} s | {100*gain:.2f}% | {old['max_compiled_peak_bytes']/1e9:.2f} → {new['max_compiled_peak_bytes']/1e9:.2f} GB |")
    lines += ['', 'Timings are medians of two varied batches per length, taking the slowest synchronized host for each update. '
              'Compiled memory is a compiler estimate. Compilation and host audit work are outside timed updates.', '',
              '## Numerical evidence', '',
              'The initial strict coordinate-wise TPU screen **failed** and remains rejected. A separately registered diagnostic '
              'showed that its discrepancy appears when introducing conditional execution even with every board still computed. '
              'The same compiled program with padding enabled versus skipped matched exactly.', '',
              'The final qualification carried state through four different real batches spanning both lengths. All parameters, '
              'both AdamW moments and scalar metrics matched the conditional reference exactly at every update on every rank. '
              'Independent comparisons with the original compiler passed the prospectively registered group/metric bounds. '
              'CPU checks also cover full gradients, FP32/BF16 and empty/partial shards.', '',
              f"Maximum relative L2 across semantic groups: parameters {drift['params']:.3g}, first moments {drift['first']:.3g}, "
              f"second moments {drift['second']:.3g}. Minimum first-moment cosine: {drift['min_first_moment_cosine']:.10f}. "
              f"Maximum absolute parameter difference from the old compiler: {drift['maximum_parameter_abs']:.3g}. "
              'This is not bitwise reproduction of the historical compiler.', '',
              'The memory-summary overflow discovered in the first harness was corrected in subsequent drivers using ceiling-KiB '
              'int32 collectives. Historical failed receipts and original per-host measurements are retained.', '',
              '## Reuse and limits', '',
              'Use the cloned recipe with [selected-training-config-001.json](selected-training-config-001.json); its only training '
              'configuration change is `training.skip_padding: true`. The old configurations retain their default dense path. '
              'Freeze this configuration as a new run before training. Historical source-bound checkpoints require an audited '
              'migration or initialization step. No long training job was started by this optimization pass.', '',
              'Normal validation and checkpoint audits are still required in the next long run. These short systems tests establish '
              'neither long-run learning quality nor playing strength or achieved MFU. See [implementation details](IMPLEMENTATION.md), '
              '[the rejected screen](SCREEN_001.md), [the diagnostic](DIAGNOSTIC_001.md), '
              '[the prospective protocol](VARIED_PROTOCOL_001.md) and [the final audit](VARIED_001.md).', '',
              '## Separate architecture ideas', '',
              'Next, test a learned attention pool producing one board token while retaining the spatial policy readout. '
              'Also diagnose the contribution of history, then consider a distinct head for actual opponent behavior. '
              'These are unscheduled model/objective changes, described with primary sources in [FUTURE_IDEAS.md](FUTURE_IDEAS.md).', '',
              f"The three bounded attempts used {chip_hours:.2f} reserved chip-hours; this is allocation accounting, not utilization."]
    with (STUDY / 'RESULTS.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'comparisons', 'drift', 'bounded_attempt_chip_hours')}))


if __name__ == '__main__':
    main()
