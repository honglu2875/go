"""Host-owned source schedule, print-norm snapshots and epoch progress.

Actual loss positions advance samples. The schedule's reference batch is
explicit; variable complete histories do not reproduce fixed-position sampling.
"""
import copy
import math
import source_schedule

GROUPS=('input','normal','normal_gamma','noreg','output','output_noreg')


def settings(config,samples,ratios):
    return source_schedule.settings(samples=samples,global_batch=config['reference_batch_positions'],
        effective_lr_scale=config['effective_lr_scale'],factors=config['factors'],norm_ratios=ratios,
        lookahead_alpha=config['lookahead']['alpha'],no_lr_warmup=config['no_lr_warmup'])


def validate(config,horizon):
    fields={'reference_batch_positions','effective_lr_scale','factors','initial_samples',
            'no_lr_warmup','lookahead','epochs','print_every'}
    if set(config)!=fields or type(horizon) is not int or horizon<1:
        raise ValueError('Explicit source runtime settings and horizon required')
    epochs=config['epochs']
    if (not isinstance(epochs,list) or not epochs or any(not isinstance(e,list) or not e
            or any(type(n) is not int or n<1 for n in e) for e in epochs)
            or sum(map(sum,epochs))!=horizon):raise ValueError('Subepochs must cover the horizon')
    if type(config['print_every']) is not int or config['print_every']<1:raise ValueError('Positive print cadence required')
    look=config['lookahead']
    if set(look)!={'k','alpha'}:raise ValueError('Explicit Lookahead settings required')
    if look['k'] is None:
        if look['alpha'] is not None:raise ValueError('Disabled Lookahead alpha differs')
    elif (type(look['k']) is not int or not 1<=look['k']<=1000000
            or type(look['alpha']) not in (int,float) or not 0<look['alpha']<1):
        raise ValueError('Invalid Lookahead period/alpha')
    settings(config,config['initial_samples'],None)
    if config['factors']['gnorm_clip_scale']<=0:raise ValueError('Positive clipping factor required')


def vector(value):
    import numpy as np
    return np.asarray([value['rates'][k] for k in GROUPS]+[value['decays'][k] for k in GROUPS]
                      +[value['source_sum_gradient_clip_cap']],dtype=np.float32)


def _norms(value,baseline=False):
    if (set(value)!={'input','normal'} or any(type(x) not in (int,float) or not math.isfinite(x)
            or x<0 or (baseline and x==0) for x in value.values())):raise ValueError('Finite source norms required')


def initialize(config,horizon,baseline):
    validate(config,horizon);_norms(baseline,True)
    return dict(version=1,turn=0,samples=config['initial_samples'],epoch=0,batch_in_epoch=0,
        subepoch=0,batch_in_subepoch=0,baseline=dict(baseline),latest_norms=None,norm_snapshot_turn=None,
        settings=settings(config,config['initial_samples'],None),refresh_count=0,last_refresh_turn=0,
        last_refresh_samples=config['initial_samples'],last_refresh_norms=None)


def _refresh(state,config):
    ratios=None if state['latest_norms'] is None else {k:state['latest_norms'][k]/state['baseline'][k] for k in ('input','normal')}
    state['settings']=settings(config,state['samples'],ratios)
    state['refresh_count']+=1;state['last_refresh_turn']=state['turn']
    state['last_refresh_samples']=state['samples'];state['last_refresh_norms']=copy.deepcopy(state['latest_norms'])


def validate_state(state,config,horizon,positions=None):
    validate(config,horizon)
    expected={'version','turn','samples','epoch','batch_in_epoch','subepoch','batch_in_subepoch',
        'baseline','latest_norms','norm_snapshot_turn','settings','refresh_count','last_refresh_turn',
        'last_refresh_samples','last_refresh_norms'}
    if set(state)!=expected or state['version']!=1:raise ValueError('Runtime state coverage differs')
    for name in ('turn','samples','epoch','batch_in_epoch','subepoch','batch_in_subepoch','refresh_count','last_refresh_turn','last_refresh_samples'):
        if type(state[name]) is not int or state[name]<0:raise ValueError('Invalid runtime counter')
    if not 0<=state['last_refresh_turn']<=state['turn']<=horizon:raise ValueError('Progress outside horizon')
    if state['samples']<config['initial_samples'] or (positions is not None and state['samples']-config['initial_samples']!=positions):
        raise ValueError('Samples disagree with actual loss positions')
    if not config['initial_samples']<=state['last_refresh_samples']<=state['samples']:
        raise ValueError('Refresh sample progress differs')
    remaining=state['turn'];epoch=subepoch=batch_epoch=batch_sub=0
    for e,counts in enumerate(config['epochs']):
        if remaining>=sum(counts):remaining-=sum(counts);epoch=e+1;continue
        epoch=e;batch_epoch=remaining
        for j,count in enumerate(counts):
            if remaining>=count:remaining-=count;continue
            subepoch=j;batch_sub=remaining;break
        break
    if (state['epoch'],state['subepoch'],state['batch_in_epoch'],state['batch_in_subepoch'])!=(epoch,subepoch,batch_epoch,batch_sub):
        raise ValueError('Epoch counters disagree with accepted updates')
    _norms(state['baseline'],True)
    if state['latest_norms'] is None:
        if state['norm_snapshot_turn'] is not None:raise ValueError('Missing norm snapshot')
    else:
        _norms(state['latest_norms'])
        if type(state['norm_snapshot_turn']) is not int or not 0<state['norm_snapshot_turn']<=state['turn']:
            raise ValueError('Invalid norm snapshot turn')
    value=state['settings']
    if (set(value)!={'rates','decays','warmup_scale','per_sample_lr','source_sum_gradient_clip_cap','mean_to_source_sum_multiplier'}
            or value['mean_to_source_sum_multiplier']!=config['reference_batch_positions']):
        raise ValueError('Settings schema or batch reference differs')
    for key in ('rates','decays'):
        if set(value[key])!=source_schedule.MUON_GROUPS|source_schedule.ADAM_GROUPS:raise ValueError('Group coverage differs')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value[key].values()):raise ValueError('Invalid group setting')
    for key in ('warmup_scale','per_sample_lr','source_sum_gradient_clip_cap'):
        if type(value[key]) not in (int,float) or not math.isfinite(value[key]) or value[key]<=0:raise ValueError('Invalid scalar setting')
    prior=state['last_refresh_norms']
    if prior is not None:_norms(prior)
    ratios=None if prior is None else {k:prior[k]/state['baseline'][k] for k in ('input','normal')}
    if value!=settings(config,state['last_refresh_samples'],ratios):raise ValueError('Settings disagree with their refresh inputs')


def before_step(state,config,horizon):
    validate_state(state,config,horizon)
    if state['turn']==horizon:raise ValueError('Runtime complete')
    result=copy.deepcopy(state);epoch_entry=result['batch_in_epoch']==0;subepoch_entry=result['batch_in_subepoch']==0
    if epoch_entry:_refresh(result,config)
    return result,dict(epoch_entry=epoch_entry,subepoch_entry=subepoch_entry,
                       is_print_batch=(result['batch_in_epoch']+1)%config['print_every']==0)


def after_step(state,config,horizon,*,positions,pre_update_norms):
    if type(positions) is not int or positions<=0:raise ValueError('Actual global loss position count required')
    result=copy.deepcopy(state);result['turn']+=1;result['samples']+=positions
    result['batch_in_epoch']+=1;result['batch_in_subepoch']+=1
    is_print=result['batch_in_epoch']%config['print_every']==0
    if is_print:
        if pre_update_norms is None:raise ValueError('Pre-update print norms missing')
        _norms(pre_update_norms);result['latest_norms']=dict(pre_update_norms);result['norm_snapshot_turn']=result['turn']
    elif pre_update_norms is not None:raise ValueError('Unexpected non-print snapshot')
    interval=5 if result['samples']<=200000000 else 50
    refresh=result['batch_in_epoch']%interval==0
    if refresh:_refresh(result,config)
    counts=config['epochs'][result['epoch']];subepoch_end=result['batch_in_subepoch']==counts[result['subepoch']];epoch_end=False
    if subepoch_end:
        result['subepoch']+=1;result['batch_in_subepoch']=0
        if result['subepoch']==len(counts):epoch_end=True;result['epoch']+=1;result['subepoch']=0;result['batch_in_epoch']=0
    validate_state(result,config,horizon)
    return result,dict(refreshed=refresh,norm_snapshot=is_print,subepoch_end=subepoch_end,epoch_end=epoch_end)
