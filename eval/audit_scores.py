#!/usr/bin/env python3
"""Audit retained GTP transcripts without changing the original match outcomes or logs."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'packages/gozero/src'))
from gozero.checkpoints import sha256
from gozero.scoring import direct_area,score_string
from gozero.snapshots import canonical_json,read_json,verify
from qualify_katago import kata_cells


def last_response(path,command):
    pending={};found=[]
    for line in path.read_text().splitlines():
        event=json.loads(line)
        if event['kind']=='command':pending[event['id']]=event['text']
        elif event['kind']=='response' and pending[event['id']]==command:
            if not event['success']:raise ValueError('Requested transcript command failed')
            found.append(event['text'])
    if not found:raise ValueError('Transcript command absent: '+command)
    return found[-1]


def audit(path):
    verify(SOURCE);path=path.resolve()
    original=read_json(path/'result.json');results=[]
    for game in original['games']:
        folder=path/f"pair-{game['pair']:03d}-{game['candidate_color']}"
        ours=folder/'candidate/gtp.jsonl';kata=folder/'katago/gtp.jsonl'
        if game['status'] not in ('completed','failed') or (game['status']=='failed' and not game.get('error','').startswith('AssertionError("Terminal scores differ:')):
            raise ValueError('A game failed for a reason other than the audited scoring assumption')
        if len(game['moves'])<2 or any(m['vertex'].lower()!='pass' for m in game['moves'][-2:]):raise ValueError('Game did not end in two passes')
        if game['checked_positions']!=len(game['moves']):raise ValueError('Not every played position was checked')
        board=last_response(ours,'showboard');kata_board=kata_cells(last_response(kata,'showboard'),9)
        if board!=kata_board:raise ValueError('Terminal boards differ')
        raw=score_string(direct_area(board,7.5));native=last_response(ours,'final_score');external=last_response(kata,'final_score')
        if raw!=native:raise ValueError('Independent raw area and native scores disagree')
        results.append({'pair':game['pair'],'candidate_color':game['candidate_color'],'plies':len(game['moves']),
                        'board':board,'raw_area_score':raw,'native_score':native,'katago_adjudicated_score':external,
                        'winner_agrees':raw[0]==external[0],'candidate_points':0.5 if raw=='0' else float(raw[0]==game['candidate_color']),
                        'source_status':game['status'],'candidate_transcript_sha256':sha256(ours),'katago_transcript_sha256':sha256(kata)})
    report={'schema_version':1,'kind':'independent_terminal_score_audit','audit_snapshot':SOURCE.name,
            'original_match_result_sha256':sha256(path/'result.json'),'original_match_snapshot':original['snapshot_id'],
            'original_status':original['status'],'games':results,'all_terminal_boards_matched':True,'native_raw_scores_independently_verified':True,
            'all_winners_agree':all(r['winner_agrees'] for r in results),'candidate_total_points':sum(r['candidate_points'] for r in results),
            'score_margin_disagreements':sum(r['raw_area_score']!=r['katago_adjudicated_score'] for r in results),
            'interpretation':'Raw Tromp-Taylor scoring was the declared native contract. The initial harness incorrectly required exact agreement with KataGo pass-alive area adjudication. Original failure records remain unchanged; this audit does not establish general rules compatibility or strength improvement.'}
    verify(SOURCE);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--matches',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.matches)
    with args.output.open('xb') as stream:stream.write(canonical_json(report))
    print(json.dumps({k:v for k,v in report.items() if k!='games'},sort_keys=True))
