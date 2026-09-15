"""Pinned positions from previously audited real-KataGo transcripts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from gozero.checkpoints import sha256
from gozero.model_artifacts import artifact
from gozero.snapshots import canonical_json,read_json
from learned_gtp import action
from qualify_katago import kata_cells


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


def build(root,c,native):
    audit_path=artifact(root,c['source_audit']);require(sha256(audit_path)==c['source_audit_sha256'],'Original match audit changed')
    audit=read_json(audit_path);require(audit['status']=='passed'and not audit['registered_combined_criterion_met'],'Unexpected prior study')
    arm=audit['arms']['exact'];base=artifact(root,arm['directory']);files={};games=[];eligible=[]
    for name,expected in sorted(arm['raw_files_sha256'].items()):
        if '/pair-'not in name or not name.endswith('/result.json'):continue
        path=artifact(base,name);require(sha256(path)==expected,'Retained game changed')
        game=read_json(path);require(game['status']in ('completed','truncated')and game['checked_positions']==len(game['moves']),'Incomplete game evidence')
        paths=[path,path.parent/'candidate/gtp.jsonl',path.parent/'katago/gtp.jsonl']
        for item in paths:
            relative=str(item.relative_to(base));require(sha256(item)==arm['raw_files_sha256'][relative],'Retained transcript changed')
            files[str(item.relative_to(root))]=sha256(item)
        boards=played_boards(paths[1],len(game['moves']),lambda b:b)
        other=played_boards(paths[2],len(game['moves']),lambda b:kata_cells(b,9))
        require(boards==other,'Retained candidate/KataGo boards differ')
        moves=[action(m['vertex'],9)for m in game['moves']]
        require(all(m['color']==('B'if i%2==0 else'W')for i,m in enumerate(game['moves'])),'Unexpected color sequence')
        index=len(games);games.append({'path':str(path.relative_to(root)),'data':game,'moves':moves,'boards':boards})
        for ply,move in enumerate(game['moves']):
            if move['color']!=game['candidate_color']or move['opening']:continue
            category='after_pass'if ply and moves[ply-1]==81 else'control'
            identity=f'{name}:{ply}';key=hashlib.sha256((str(c['selection_seed'])+':'+identity).encode()).hexdigest()
            eligible.append({'case_id':key,'game_index':index,'ply':ply,'category':category})
    require(len(games)==72,'Expected all 72 exact-student games')
    post=sorted((r for r in eligible if r['category']=='after_pass'),key=lambda r:r['case_id'])
    controls=sorted((r for r in eligible if r['category']=='control'),key=lambda r:r['case_id'])[:c['control_positions']]
    require(len(post)==899 and len(controls)==c['control_positions'],'Fixed source population differs')
    offsets=np.asarray([0,*np.cumsum([len(g['moves'])for g in games])],np.int64)
    actions=np.asarray([a for g in games for a in g['moves']],np.int32)
    stones,legal,outcomes=native.replay_observations(json.dumps({'size':9,'komi':7.5,'scoring':'pass_alive_area'}),actions,offsets)
    stones=stones.reshape(-1,81);legal=legal.reshape(-1,82);outcomes=json.loads(outcomes);starts=[];cursor=0;checked=0
    for index,g in enumerate(games):
        starts.append(cursor);count=len(g['moves'])
        game=native.Game(json.dumps({'size':9,'komi':7.5,'history':4,'scoring':'pass_alive_area',
            'simulations':0,'cpuct':0.,'max_search_edges':1000000}))
        for ply,(board,move)in enumerate(zip(g['boards'],g['moves'])):
            require(np.array_equal(stones[cursor+ply],game.state()[4])
                and np.flatnonzero(legal[cursor+ply]).tolist()==game.legal(),'Bulk pre-action replay differs from per-move replay')
            game.play(1+ply%2,move)
            expected=np.asarray(['.XO'.index(x)for x in board.replace('\n','')],np.uint8)
            require(np.array_equal(game.state()[4],expected),'New native replay differs from real KataGo board');checked+=1
        require(outcomes[index]['terminal']==(g['data']['status']=='completed'),'Native terminal flag differs')
        if outcomes[index]['terminal']:
            score=g['data']['katago_adjudicated_score'];value=0. if score=='0'else float(score[2:])*(1 if score[0]=='W'else -1)
            require(outcomes[index]['white_score']==value,'Native terminal score differs from real KataGo')
        cursor+=count
    require(cursor==len(stones),'Replay row coverage differs')
    cases=[]
    for row in sorted(post+controls,key=lambda r:r['case_id']):
        game=games[row['game_index']];ply=row['ply'];move=game['data']['moves'][ply];board_index=starts[row['game_index']]+ply
        cases.append({**row,'game_path':game['path'],'game_sha256':files[game['path']],
            'panel':game['path'].split('/')[-3],'game_status':game['data']['status'],'candidate_color':game['data']['candidate_color'],
            'prefix':game['moves'][:ply],'stones':stones[board_index].tolist(),'legal':np.flatnonzero(legal[board_index]).tolist(),
            'recorded_action':action(move['vertex'],9),'recorded_search':{k:v for k,v in move['search'].items()if k!='seconds'}})
    return {'schema_version':1,'kind':'retained_katago_endgame_positions','source_audit_sha256':c['source_audit_sha256'],
        'selection_seed':c['selection_seed'],'control_positions':c['control_positions'],'source_game_count':len(games),
        'source_boards_checked':checked,'source_scores_checked':sum(o['terminal']for o in outcomes),
        'eligible_after_pass':len(post),'eligible_controls':sum(r['category']=='control'for r in eligible),
        'files_sha256':files,'cases':cases}


def qualified_cases(corpus,limit):
    groups={}
    for row in corpus['cases']:groups.setdefault((row['panel'],row['category']),[]).append(row)
    selected=[]
    # Deterministic spread across panels and categories, independent of outputs.
    for index in range(limit):
        for key in sorted(groups):
            if index<len(groups[key]):selected.append(groups[key][index])
            if len(selected)==limit:return selected
    return selected
