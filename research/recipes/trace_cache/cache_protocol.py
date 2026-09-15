"""Host commit guards for a fixed-order native game batch and device KV.

Rust remains authoritative for legality, branch validation and state advancement.
These guards bind retained history KV to the next native ticket and canonical
tape. They perform no Go transitions. Their cost belongs in packet-loop timing.
"""
import copy
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


class CacheRebuildRequired(ValueError):
    def __init__(self,games):
        self.games=games
        super().__init__('Canonical-history cache reconstruction required for games '+str(games))


class CacheCoordinator:
    def __init__(self,games,horizon,samples,actions,max_tokens,models,contexts=(0,0)):
        require(games>0 and 1<=horizon<=64 and 1<=samples<=64,'Invalid cache dimensions')
        self.games=games;self.horizon=horizon;self.samples=samples;self.actions=actions;self.max_tokens=max_tokens
        self.models=tuple(models);self.contexts=tuple(contexts);self.entries=[None]*games;self.pending=None

    def invalidate(self,models=None,contexts=None):
        """Caller must also cancel any pending native request before another start."""
        self.entries=[None]*self.games;self.pending=None
        if models is not None:self.models=tuple(models)
        if contexts is not None:self.contexts=tuple(contexts)

    def frame(self,tickets,tokens,lengths):
        require(len(tickets)==self.games and tokens.shape==(self.games,self.max_tokens)
                and tokens.dtype==np.int32 and lengths.shape==(self.games,)and lengths.dtype==np.int32,
                'Cache/native frame dimensions differ')
        tapes=[]
        for game,ticket in enumerate(tickets):
            length=int(lengths[game]);require(0<=length<self.max_tokens and tokens[game,0]==self.actions,'Malformed canonical tape')
            if ticket is None:
                require(length==0,'Inactive native frame is not a dummy');tapes.append(None);continue
            require(set(ticket)==set('generation move_number models contexts episode'.split())
                    and type(ticket['generation'])is int and ticket['generation']>0
                    and type(ticket['episode'])is int and ticket['episode']>=0
                    and ticket['move_number']==length and tuple(ticket['models'])==self.models
                    and tuple(ticket['contexts'])==self.contexts,'Cache model/context/ticket identity differs')
            require(((0<=tokens[game,1:length+1])&(tokens[game,1:length+1]<self.actions)).all(),'Invalid canonical action')
            tapes.append(tokens[game,:length+1].tobytes())
        return tapes

    def adopt_rebuilt(self,tickets,tokens,lengths):
        """Call after rebuilding all active roots with the current parameters.

        A fresh coordinator/invalidation is required because the whole device
        cache was replaced. Inactive roots remain invalid and rebuild on reuse.
        """
        require(self.pending is None and all(e is None for e in self.entries),'Rebuild requires an invalidated coordinator')
        tapes=self.frame(tickets,tokens,lengths)
        for game,ticket in enumerate(tickets):
            if ticket is not None:
                self.entries[game]={'generation':ticket['generation']-1,'episode':ticket['episode'],
                    'move_number':ticket['move_number'],'actions':(),'selector':0,'tape':tapes[game]}

    def prepare(self,tickets,tokens,lengths):
        require(self.pending is None,'A cache request is already pending')
        tapes=self.frame(tickets,tokens,lengths)
        # [active, reset, flattened view/sample, actual root token, prior count,
        #  prior accepted actions padded with -1]. One compact host transfer.
        controls=np.full((self.games,5+self.horizon),-1,np.int32);controls[:,:5]=0
        controls[:,1]=1;controls[:,3]=self.actions;misses=[]
        for game,ticket in enumerate(tickets):
            if ticket is None:continue
            position=ticket['move_number'];entry=self.entries[game]
            controls[game,0]=1;controls[game,3]=tokens[game,position]
            if entry is None:
                if position:misses.append(game)
                continue
            require(ticket['generation']==entry['generation']+1,'Stale or skipped cache generation; cancel and invalidate')
            if ticket['episode']!=entry['episode']:
                require(ticket['episode']==entry['episode']+1 and position==0,'Unexpected episode transition')
                continue
            require(position==entry['move_number']+len(entry['actions']) and tapes[game]==entry['tape'],
                    'Committed cache prefix differs from the native canonical tape')
            controls[game,1]=0;controls[game,2]=entry['selector'];controls[game,4]=len(entry['actions'])
            controls[game,5:5+len(entry['actions'])]=entry['actions']
        if misses:raise CacheRebuildRequired(misses)
        self.pending=(copy.deepcopy(tickets),tapes)
        return controls

    def commit(self,resolutions,proposals):
        require(self.pending is not None,'No cache request to commit')
        require(len(resolutions)==self.games and proposals.shape==(self.games,2,self.samples,self.horizon)
                and proposals.dtype==np.int32,'Cache commit dimensions differ')
        tickets,tapes=self.pending;updated=list(self.entries);certificates=[]
        for game,(ticket,row)in enumerate(zip(tickets,resolutions)):
            actual=row['actions'];samples=row['selected_samples'];count=len(actual)
            if ticket is None:
                require(not actual and not samples and row['stop']=='work_limit'and row['terminal_white_score']is None,
                        'Inactive cache acquired a commit or outcome')
                certificates.append(None);continue
            require(1<=count<=self.horizon and len(samples)==count
                    and all(type(x)is int and 0<=x<self.actions for x in actual)
                    and all(type(x)is int and 0<=x<self.samples for x in samples),'Invalid active cache commit')
            view=(ticket['move_number']+count-1)%2;sample=samples[-1]
            require(proposals[game,view,sample,:count-1].tolist()==actual[:-1],
                    'Selected cache branch does not contain the committed prior actions')
            selector=view*self.samples+sample
            updated[game]={'generation':ticket['generation'],'episode':ticket['episode'],
                'move_number':ticket['move_number'],'actions':tuple(actual),'selector':selector,
                'tape':tapes[game]+np.asarray(actual,np.int32).tobytes()}
            certificates.append({'selector':selector,'valid_through':ticket['move_number']+count-1,
                'committed_ply':ticket['move_number']+count,'last_actual_action':actual[-1]})
        self.entries=updated;self.pending=None
        return certificates
