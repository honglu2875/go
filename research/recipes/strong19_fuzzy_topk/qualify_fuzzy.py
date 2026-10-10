"""CPU gates for grouped dictionaries and their causal policy/value integration."""
import copy
import json
from pathlib import Path
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from gozero.fuzzy_topk import choicewise, reference, diagnostics
import joint, learner, adamw, optimizer_io


def close(a,b,atol=2e-5,rtol=2e-5):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)


class Qualification(unittest.TestCase):
    def test_dictionary_oracle_forward_and_all_gradients(self):
        rng=np.random.default_rng(713)
        for g in (1,2,4,8):
            d,k=8,5;h=g*k
            args=tuple(jnp.asarray(rng.normal(size=shape),jnp.float32) for shape in ((2,3,d),(d,h),(h,),(h,d),(d,)))
            direction=jnp.asarray(rng.normal(size=(2,3,d)),jnp.float32)
            for dtype in (jnp.float32,jnp.bfloat16):
                fast=lambda *v:choicewise(*v,groups=g,dtype=dtype)[0]
                oracle=lambda *v:reference(*v,groups=g,dtype=dtype)
                close(fast(*args),oracle(*args),atol=2e-5,rtol=2e-5)
                actual=jax.jit(jax.grad(lambda *v:jnp.sum(fast(*v)*direction),argnums=tuple(range(5))))(*args)
                expected=jax.jit(jax.grad(lambda *v:jnp.sum(oracle(*v)*direction),argnums=tuple(range(5))))(*args)
                # BF16 transpose matmuls round before their sum in the grouped
                # implementation. Compare relative gradient error, not bits.
                if dtype==jnp.float32:close(actual,expected)
                else:
                    for a,b in zip(actual,expected,strict=True):
                        self.assertLess(float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(b),1e-8)),.02)

    def test_ties_nonpositive_features_and_live_statistics(self):
        x=jnp.zeros((2,3,4));up=jnp.zeros((4,8));bias=jnp.asarray([2,2,-1,-1,0,0,3,1.])
        down=jnp.arange(32,dtype=jnp.float32).reshape(8,4);db=jnp.zeros(4)
        live=jnp.asarray([[True,False,False],[True,True,False]])
        y,stats=choicewise(x,up,bias,down,db,groups=2,valid=live)
        close(y,jnp.broadcast_to(2*down[0]+3*down[6],y.shape))
        np.testing.assert_array_equal(stats[:8],[3,0,3,0,3,0,3,0])
        np.testing.assert_array_equal(stats[8:16],[3,0,0,0,0,0,3,0])
        m=diagnostics(stats[None],groups=2,selected=4)
        self.assertEqual(float(m['fuzzy_positive_group_fraction']),.5)
        all_y,all_stats=choicewise(x,up,bias,down,db,groups=2)
        close(y,all_y);self.assertEqual(float(all_stats[-1]),6.)
        grad=lambda mask:jax.grad(lambda u:jnp.sum(choicewise(x,u,bias,down,db,groups=2,valid=mask)[0]))(up)
        close(grad(live),grad(jnp.zeros_like(live)))
        empty=diagnostics(jnp.zeros((2,17)),groups=2,selected=4)
        self.assertTrue(all(np.isfinite(v) for v in empty.values()))

    def fixture(self):
        c=json.loads((Path(__file__).parent/'dense.json').read_text())
        n=copy.deepcopy(c['model']);n.update(width=16,layers=2,mlp_hidden=24,heads=2,kv_heads=1,max_board_size=3,
            max_positions=8,dtype='float32',attention_backend='xla',encoder_width=16,encoder_blocks=2,encoder_passes=2,
            encoder_attention_blocks=1,encoder_attention_heads=2,encoder_attention_mlp_hidden=24,encoder_expansion=2,
            connector_channels=4,encoder_layer_scale=.1,policy_context_dim=4)
        n['fuzzy']=dict(groups=4,selected=13,implementation='choicewise',bias=True)
        v=dict(hidden=7,spatial_channels=5);rng=np.random.default_rng(916);b,t,size=2,3,3
        counts=jnp.asarray([3,1]);live=jnp.arange(t)[None,:]<counts[:,None]
        batch=dict(spatial=jnp.asarray(rng.normal(size=(b,t,size,size,22)),jnp.float32),
            global_features=jnp.asarray(rng.normal(size=(b,t,19)),jnp.float32),
            actions=jnp.asarray([[1,2,3],[4,5,6]],jnp.int32),counts=counts,
            policies=jnp.full((b,t,10),.1),legal=jnp.ones((b,t,10),bool),values=jnp.where(live,.4,jnp.nan))
        return n,v,batch

    def test_shared_initialization_and_bounded_gradients(self):
        c,v,b=self.fixture();p=joint.initialize(912,c,v)
        dense=joint.initialize(912,{k:w for k,w in c.items() if k!='fuzzy'},v)
        for name in p.keys()&dense.keys():np.testing.assert_array_equal(p[name],dense[name])
        fn=lambda p:joint.losses(p,b,c,value_weight=.7,chunk_frames=2,skip_padding=True)
        reference_loss=lambda p:learner.totals(learner.materialized(p,b,c),b,c,value_weight=.7)
        result=jax.jit(jax.value_and_grad(fn,has_aux=True))(p)
        expected=jax.jit(jax.value_and_grad(reference_loss,has_aux=True))(p)
        close(result,expected,atol=2e-5,rtol=5e-4)
        self.assertEqual(float(result[0][1]['fuzzy_live_tokens_per_layer']),12.)
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(result)))
        for prefix in ('encoder.','blocks.fuzzy.','value_head.'):
            self.assertGreater(sum(float(jnp.sum(g*g)) for k,g in result[1].items() if k.startswith(prefix)),0.)

    def test_causal_forward_cache_and_invalid_append(self):
        c,v,b=self.fixture();p=joint.initialize(913,c,v)
        out=joint.forward(p,b,c)
        first,cache=joint.first_move(p,b['spatial'][:,0],b['global_features'][:,0],c)
        for name in ('policy','value_logits'):close(first[name],out[name][:,0])
        for t in (1,2):
            current,cache=joint.append_move(p,cache,b['actions'][:,t-1],b['spatial'][:,t],b['global_features'][:,t],c,
                attention_positions=t+1,active=b['counts']>t)
            for name in ('policy','value_logits'):close(current[name],out[name][:,t])
        bad=copy.copy(b);bad['spatial']=b['spatial'].at[:,2].set(19.);bad['actions']=b['actions'].at[:,1:].set(0)
        altered=joint.forward(p,bad,c)
        close(out['policy'][:,:2],altered['policy'][:,:2])
        invalid,unchanged=joint.append_move(p,cache,jnp.full((2,),-1),b['spatial'][:,0],b['global_features'][:,0],c,attention_positions=4)
        close(invalid['policy'],jnp.zeros_like(invalid['policy']));close(cache['keys'],unchanged['keys'])

    def test_optimizer_state_round_trip(self):
        c,v,b=self.fixture();p=joint.initialize(914,c,v)
        g=jax.tree.map(jnp.ones_like,p);state=adamw.initialize(p)
        p,state,metrics=adamw.apply_gradient(p,state,g,jnp.asarray(1.),learning_rate=.001,architecture=c['architecture'])
        self.assertEqual(float(metrics['accepted']),1.)
        identity=dict(configuration_sha256='a'*64,source_sha256='b'*64)
        metadata,arrays=optimizer_io.flatten(p,state,**identity)
        schema=[{k:x[k] for k in ('path','shape','dtype')} for x in joint.parameter_schema(c,v)]
        restored=optimizer_io.restore(metadata,arrays,schema=schema,**identity)
        close((p,state),restored)

    def test_first_update_lr_scaling(self):
        p={'blocks.fuzzy.up.weight':jnp.arange(35,dtype=jnp.float32).reshape(5,7)/30,
           'blocks.fuzzy.up.bias':jnp.zeros(7)}
        g=jax.tree.map(lambda x:jnp.ones_like(x)*.3,p);state=adamw.initialize(p)
        _,_,base=adamw.apply_gradient(p,state,g,jnp.asarray(1.),learning_rate=.001,architecture='causal_visual_policy')
        for rate in (.0004,.0006,.0015,.00225):
            _,_,m=adamw.apply_gradient(p,state,g,jnp.asarray(1.),learning_rate=rate,architecture='causal_visual_policy')
            for name,value in m.items():
                close(value,base[name]*(rate/.001 if name.startswith('update_norm_') else 1.),atol=2e-6,rtol=2e-6)


if __name__=='__main__':
    if jax.default_backend()!='cpu':raise ValueError('CPU-only qualification')
    unittest.main()
