"""Explicit active and issued temporal work, including the unchanged encoder."""
import math
import joint,encoder

def budget(c,v):
    schema=joint.parameter_schema(c,v);d,L,f=c['width'],c['layers'],c['mlp_hidden']
    kv=c['kv_heads']*(d//c['heads']);points=c['max_board_size']**2
    attention_projections=4*L*(2*d*d+2*d*kv)
    base=6*d*f;active=issued=base;fuzzy=c.get('fuzzy');pad=lambda v:128*math.ceil(v/128)
    padded=6*pad(d)*pad(f)
    if fuzzy:
        g,k=fuzzy['groups'],fuzzy['selected'];h=g*k
        active=2*d*(h+k);issued=4*d*h
        padded=2*pad(d)*(pad(h)+g*pad(k))
    context=c['policy_context_dim'];w=c['encoder_width']
    policy=2*((points+1)*d+points*w+d*context+points*w*context+points*context)
    value=2*(d*v['hidden']+v['hidden']*3);decode={}
    for history in (0,128,512,768,1534):
        work=dict(encoder=encoder.flops(c,c['max_board_size']),temporal_projection=attention_projections,
            temporal_attention=4*L*d*(2*history+3),policy=policy,value=value)
        common=sum(work.values())
        decode[str(history)]={**work,'temporal_ffn_active':2*L*active,'temporal_ffn_issued':2*L*issued,
            'total_active_matrix_flops':common+2*L*active,'total_issued_matrix_flops':common+2*L*issued}
    return dict(parameters=sum(x['elements'] for x in schema),encoder_parameters=sum(x['elements'] for x in schema if x['path'].startswith('encoder.')),
        temporal_ffn_parameters=sum(x['elements'] for x in schema if x['path'].startswith(('blocks.fuzzy.','blocks.gate.','blocks.up.','blocks.down.'))),
        ffn_token_active_matrix_flops=active,ffn_token_issued_matrix_flops=issued,
        ffn_token_128_tile_rounding_estimate=padded,ffn_active_ratio_vs_swiglu=active/base,ffn_issued_ratio_vs_swiglu=issued/base,
        training_ffn_issued_matrix_flops_per_live_position=3*L*3*issued,
        decode=decode,schema=schema,
        scope='Cached move includes both action and board tokens, encoder passes, attention and heads. Training has three temporal tokens per position. Issued training excludes rematerialization and padding. Tile estimate is illustrative, not measured HLO or MFU. Selection, masks, reductions, statistics, nonlinearities are additional work.')
