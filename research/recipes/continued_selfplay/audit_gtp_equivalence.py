#!/usr/bin/env python3
"""Audit registered real-KataGo traces of identical parent/fork parameters."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE / 'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.model_artifacts import validate_candidate
from gozero.snapshots import canonical_json, read_json, verify


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root', type=Path, required=True); p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.workspace_root.resolve(); verify(SOURCE); spec = read_json(a.spec)
    report = {'schema_version': 1, 'kind': 'continued_selfplay_gtp_equivalence', 'status': 'failed',
              'analysis_snapshot': SOURCE.name, 'spec_sha256': sha256(a.spec), 'comparisons': [],
              'scope': 'Exact fixed-visit GTP/search and real-KataGo game equivalence for identical parameters. Caps remain unscored. This is adapter qualification, not evidence of strength improvement.'}
    try:
        require(spec['snapshot'] == SOURCE.name and spec['maximum_attempts'] == 1, 'Wrong registration')
        for item in spec['prerequisites']:
            require(sha256(root / item['path']) == item['sha256'] and read_json(root / item['path'])['status'] == 'passed',
                    'Prerequisite failed or changed')
        directory = root / spec['directory']; panel = read_json(directory / 'result.json')
        require(panel['snapshot_id'] == SOURCE.name and panel['spec_sha256'] == sha256(SOURCE / spec['panel']), 'Panel identity differs')
        children = {item['id']: item for item in panel['matches']}
        require(set(children) == {'reference', 'fork'} and all(not r['timed_out'] for r in children.values()), 'Incomplete or timed-out panel')
        results = {}; configs = {}; candidates = {}
        for name, child in children.items():
            path = directory / name / 'result.json'; result = read_json(path)
            require(sha256(path) == child['result_sha256'] and child['games'] == result['games'], 'Panel/raw games differ')
            cfg = read_json(SOURCE / spec['matches'][name]); candidate = read_json(SOURCE / cfg['candidate'])
            identity = validate_candidate(root, candidate)
            require(result['spec_sha256'] == sha256(SOURCE / spec['matches'][name])
                    and result['candidate_sha256'] == sha256(SOURCE / cfg['candidate'])
                    and result['candidate'] == candidate and result['candidate_adapter'] == 'eval/learned_gtp.py'
                    and result['inference_native_snapshot'] == spec['native_snapshot']
                    and result['native_receipt_sha256'] == spec['native_receipt_sha256'], 'Candidate or native identity differs')
            if name == 'fork':
                require(identity['native_receipt'] is not None, 'Fork native dependency was not verified')
            results[name] = result; configs[name] = cfg; candidates[name] = candidate
        require(candidates['reference']['model_export_sha256'] == candidates['fork']['model_export_sha256'], 'Parameters differ')
        stripped = [{k: v for k, v in configs[name].items() if k not in ('candidate', 'candidate_cpus', 'katago_cpus')}
                    for name in ('reference', 'fork')]
        require(stripped[0] == stripped[1], 'Match conditions differ beyond candidate identity and isolated CPUs')
        require(len(results['reference']['games']) == len(results['fork']['games']) == 4, 'Wrong scheduled game count')
        identity_fields = ('pair', 'candidate_color', 'opening', 'status', 'candidate_version', 'katago_version', 'katago_rules',
                           'katago_config_sha256', 'candidate_search_totals', 'checked_positions', 'score', 'independent_raw_score',
                           'katago_adjudicated_score', 'score_margin_agrees_with_katago', 'winner_agrees_with_katago', 'candidate_points')
        for left, right in zip(results['reference']['games'], results['fork']['games'], strict=True):
            require(left['status'] in ('completed', 'truncated') and right['status'] == left['status'], 'Process failure or unequal game status')
            require({k: left[k] for k in identity_fields if k in left} == {k: right[k] for k in identity_fields if k in right},
                    'Game identity, search totals or outcome differs')
            require(len(left['moves']) == len(right['moves']) == left['checked_positions'] == right['checked_positions'], 'Board coverage differs')
            for x, y in zip(left['moves'], right['moves'], strict=True):
                def strip(move):
                    return {k: ({a: b for a, b in v.items() if a != 'seconds'} if k == 'search' else v)
                            for k, v in move.items() if k != 'engine_seconds'}
                require(strip(x) == strip(y), 'Move or neural/search result differs')
            game = f"pair-{left['pair']:03d}-{left['candidate_color']}"
            for name, expected in (('reference', left), ('fork', right)):
                require(read_json(directory / name / game / 'result.json') == expected, 'Raw game differs from match receipt')
            require((directory / 'reference' / game / 'game.sgf').read_bytes() == (directory / 'fork' / game / 'game.sgf').read_bytes(), 'SGF differs')
            report['comparisons'].append({'game': game, 'moves_and_search_exact': len(left['moves']), 'status': left['status'],
                                          'score': left.get('score')})
        report.update(status='passed', scheduled_games=8,
                      checked_boards=sum(2 * r['moves_and_search_exact'] for r in report['comparisons']),
                      completed_scores=sum(2 * (r['status'] == 'completed') for r in report['comparisons']),
                      caps_unscored=sum(2 * (r['status'] == 'truncated') for r in report['comparisons']),
                      panel_result_sha256=sha256(directory / 'result.json'))
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        with a.output.open('xb') as stream:
            stream.write(canonical_json(report))
        print(canonical_json(report).decode(), flush=True)


if __name__ == '__main__':
    main()
