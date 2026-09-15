"""Independent validation-only direct Tromp-Taylor area scoring.

This deliberately uses simple grid flood fills, independently of native chain
storage. It neither classifies nor removes dead stones. Rollouts remain in Rust.
"""
import math


def direct_area(board: str, komi: float) -> float:
    rows=board.splitlines();size=len(rows)
    if not size or any(len(row)!=size or set(row)-set('.XO') for row in rows) or not math.isfinite(komi):
        raise ValueError('Invalid square board or komi')
    cells=''.join(rows);counts={'X':cells.count('X'),'O':cells.count('O')};seen=set()
    for start,c in enumerate(cells):
        if c!='.' or start in seen:continue
        region={start};seen.add(start);todo=[start];borders=set()
        while todo:
            p=todo.pop();x=p%size;y=p//size
            for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if not 0<=nx<size or not 0<=ny<size:continue
                n=ny*size+nx
                if cells[n]!='.':borders.add(cells[n])
                elif n not in seen:seen.add(n);region.add(n);todo.append(n)
        if len(borders)==1:counts[next(iter(borders))]+=len(region)
    return float(counts['O']-counts['X']+komi)


def score_string(white_minus_black):
    if not math.isfinite(white_minus_black):raise ValueError('Nonfinite score')
    return '0' if white_minus_black==0 else ('W' if white_minus_black>0 else 'B')+'+'+str(abs(white_minus_black))
