"""Native/transcript replay helpers with explicit online-model identities.

Origin: source ad223b589aa206074724b73c6b34452b9c7cc3d0db4d614e26865a8efccd522b.
Kept inside this clone because other research recipes are excluded from snapshots.
"""
import json
import numpy as np
from gozero.checkpoints import sha256
from gozero.snapshots import read_json
from learned_gtp import action
from qualify_katago import kata_cells
from match import summarize

def require(value,message):
    if not value:raise ValueError(message)


def responses(path,command):
    pending={};found=[]
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['kind']=='command':pending[row['id']]=row['text']
        elif row['kind']=='response'and pending[row['id']]==command:
            require(row['success'],'Failed retained transcript response');found.append(row['text'])
    return found


def played_boards(path,count,convert):
    pending={};boards={};ply=0
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['kind']=='command':pending[row['id']]=row['text']
        elif row['kind']=='response':
            command=pending[row['id']].split()[0]
            if command in ('play','genmove'):
                require(row['success'],'Failed retained move');ply+=1
            elif command=='showboard':
                require(row['success']and 1<=ply<=count,'Invalid retained board position')
                board=convert(row['text'])
                require(ply not in boards or boards[ply]==board,'Repeated board response changed its position')
                boards[ply]=board
    require(ply==count and set(boards)==set(range(1,count+1)),'Transcript position coverage differs')
    return [boards[i]for i in range(1,count+1)]


def loaded_identity(folder, game, source, spec, descriptor, trained, native):
    ready = [json.loads(line) for line in (folder / 'candidate/stderr.log').read_text().splitlines()
             if line.startswith('{') and json.loads(line).get('kind') == 'engine_ready']
    require(len(ready) == 1, 'Engine readiness absent or duplicated')
    ready = ready[0]
    require(game['candidate_version'] == 'gozero-native-gumbel-v1'
            and ready['candidate_sha256'] == sha256(source / spec['candidate'])
            and ready['training_snapshot'] == descriptor['training_snapshot']
            and ready['weights_sha256'] == descriptor['model_export_sha256']
            and ready['native_sha256'] == native['binary_sha256'] == trained['native_binary_sha256']
            and ready['backend'] == 'cpu', 'Actual loaded candidate identity differs')
    require(ready['simulations_excluding_root'] == spec['candidate_simulations_excluding_root']
            and ready['cpuct'] == spec['candidate_cpuct'], 'Actual inference budget differs')


def audit_transcript(path, game, side):
    pending = {}; seen = set(); moves = []; stats = []; starts = []; exits = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'start':
            starts.append(row['argv'])
        elif row['kind'] == 'exit':
            exits.append(row['returncode'])
        elif row['kind'] == 'transport_error':
            raise ValueError('Retained GTP transport failure')
        elif row['kind'] == 'command':
            require(row['id'] not in seen, 'Duplicated command ID'); seen.add(row['id'])
            pending[row['id']] = row['text']
        elif row['kind'] == 'response':
            require(row['id'] in pending and row['success'], 'Unmatched or failed GTP response')
            command = pending.pop(row['id']).split()
            if command[0] in ('play', 'genmove'):
                vertex = command[2] if command[0] == 'play' else row['text'].strip()
                moves.append((command[1].upper(), action(vertex, 9)))
            elif command[0] == 'gozero-search-stats':
                stats.append(json.loads(row['text']))
    require(not pending and exits == [0] and starts == [game[side + '_argv']],
            'Incomplete transcript, engine exit or actual process argv differs')
    require(moves == [(m['color'], action(m['vertex'], 9)) for m in game['moves']], 'Transcript moves differ')
    require(responses(path, 'version') == [game[side + '_version']], 'Transcript engine version differs')
    if side == 'candidate':
        require(stats == [m['search'] for m in game['moves'] if 'search' in m], 'Transcript search evidence differs')
    else:
        require([json.loads(r) for r in responses(path, 'kata-get-rules')] == [game['katago_rules']], 'Transcript rules differ')
    require(responses(path, 'final_score') == ([game['score']] if game['status'] == 'completed' else []),
            'Transcript final score or cap adjudication differs')


def replay_game(folder, game, native, spec):
    boards = played_boards(folder / 'candidate/gtp.jsonl', len(game['moves']), lambda b: b)
    other = played_boards(folder / 'katago/gtp.jsonl', len(game['moves']), lambda b: kata_cells(b, 9))
    require(boards == other, 'Candidate/KataGo board transcripts differ')
    for side in ('candidate', 'katago'):
        audit_transcript(folder / side / 'gtp.jsonl', game, side)
    position = native.Game(json.dumps({'size': 9, 'komi': 7.5, 'history': 4, 'scoring': 'pass_alive_area',
                                       'simulations': 0, 'cpuct': 0., 'max_search_edges': 1000000}))
    actions = []
    for ply, (board, move) in enumerate(zip(boards, game['moves'])):
        chosen = action(move['vertex'], 9); actions.append(chosen)
        require(chosen in position.legal(), 'Recorded move illegal in native replay')
        position.play(1 + ply % 2, chosen)
        expected = np.asarray(['.XO'.index(x) for x in board.replace('\n', '')], np.uint8)
        require(np.array_equal(position.state()[4], expected), 'Native replay differs from actual KataGo board')
        if 'search' in move:
            stats = move['search']
            require(all(type(stats[k]) is int and stats[k] >= 0 for k in ('simulations', 'neural_evaluations', 'terminal_evaluations'))
                    and stats['neural_evaluations'] + stats['terminal_evaluations'] == stats['simulations'] + 1,
                    'Neural/terminal work does not cover root plus simulations')
    _, _, terminal, white_score, _ = position.state()
    require(terminal == (game['status'] == 'completed'), 'Native terminal flag differs')
    if terminal:
        score = game['score']; value = 0. if score == '0' else float(score[2:]) * (1 if score[0] == 'W' else -1)
        require(white_score == value, 'Native final score differs from real KataGo')
    seed = spec['seed_prefix'] + '-' + str(game['pair'])
    return seed


def replay_timeout_prefix(folder, game, native):
    """Verify only jointly observed boards; never adjudicate a timed-out game."""
    require(game['status'] == 'failed' and game.get('error', '').startswith('TimeoutError(')
            and game.get('candidate_points') is None and 'score' not in game, 'Unsupported or assigned failure')
    count = game['checked_positions']
    require(0 <= count <= len(game['moves']) <= count + 1, 'Timeout prefix coverage differs')
    def prefix(path, convert):
        pending = {}; boards = {}; ply = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['kind'] == 'command': pending[row['id']] = row['text']
            elif row['kind'] == 'response' and row['success']:
                command = pending[row['id']].split()[0]
                if command in ('play', 'genmove'): ply += 1
                elif command == 'showboard' and ply <= count:
                    require(ply not in boards or boards[ply] == convert(row['text']), 'Timeout prefix board changed')
                    boards[ply] = convert(row['text'])
        require(set(boards) == set(range(1, count + 1)), 'Timeout transcript lacks observed boards')
        return [boards[i] for i in range(1, count + 1)]
    boards = prefix(folder / 'candidate/gtp.jsonl', lambda b: b)
    require(boards == prefix(folder / 'katago/gtp.jsonl', lambda b: kata_cells(b, 9)), 'Timeout prefix board transcripts differ')
    position = native.Game(json.dumps({'size': 9, 'komi': 7.5, 'history': 4, 'scoring': 'pass_alive_area',
                                       'simulations': 0, 'cpuct': 0., 'max_search_edges': 1000000}))
    for ply, board in enumerate(boards):
        chosen = action(game['moves'][ply]['vertex'], 9)
        require(chosen in position.legal(), 'Timeout prefix contains an illegal move')
        position.play(1 + ply % 2, chosen)
        expected = np.asarray(['.XO'.index(x) for x in board.replace('\n', '')], np.uint8)
        require(np.array_equal(position.state()[4], expected), 'Timeout prefix native replay differs')
    return count


def diagnostics(games):
    opportunities = passes = pessimistic_refusals = 0
    search = []
    for game in games:
        for previous, move in zip(game['moves'], game['moves'][1:]):
            if move['opening'] or move['color'] != game['candidate_color']:
                continue
            search.append(move['search'])
            if previous['vertex'].lower() == 'pass':
                opportunities += 1; passes += move['vertex'].lower() == 'pass'
                pessimistic_refusals += move['vertex'].lower() != 'pass' and move['search']['root_value'] <= -.95
    return {'summary': summarize(games), 'candidate_turns_after_opponent_pass': opportunities,
            'responded_with_pass': passes, 'continued_with_play': opportunities - passes,
            'continued_with_play_at_root_value_at_most_minus_0_95': pessimistic_refusals,
            'search_totals': {k: sum(g['candidate_search_totals'][k] for g in games)
                              for k in ('simulations', 'neural_evaluations', 'terminal_evaluations')},
            'search_quantiles_0_25_50_75_100': {k: np.quantile([s[k] for s in search], [0., .25, .5, .75, 1.]).tolist()
                                             for k in ('neural_evaluations', 'terminal_evaluations', 'root_value')},
            'scope': 'Secondary endgame and search descriptions. More passes may concede losses; these counts do not change the strength criterion.'}
