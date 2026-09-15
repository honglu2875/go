"""Independent convolution and visibility oracles; full/cached model contracts."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import causal
import capacity_reference
import readout_reference
from observation_attention import ObservationMask
from test_causal import C, fixture


def allowed_keys(query, stride, length):
    # Enumerate complete previous frames, then the known part of this frame.
    frame, offset = divmod(query, stride)
    end = frame * stride + (stride if offset == stride - 1 else stride - 1)
    return list(range(min(end, length)))


def stem_reference(p, spatial, depth, xp):
    x = spatial
    for i in range(depth):
        w = p[f'encoder.stem{i}.weight']
        padded = xp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
        out = xp.zeros((*x.shape[:-1], w.shape[-1]), dtype=xp.float32)
        for y in range(3):
            for z in range(3):
                out = out + padded[:, :, y:y+x.shape[2], z:z+x.shape[3]] @ w[y, z]
        out = out / xp.sqrt(xp.mean(out * out, -1, keepdims=True) + C['norm_epsilon'])
        out = out * p[f'encoder.stem{i}.scale']
        x = out / (1 + xp.exp(-out))
    return x


class EncoderChecks(unittest.TestCase):
    def test_default_model_and_gradients_are_bitwise_unchanged(self):
        s, g, a, n = map(jnp.asarray, fixture())
        p = jax.jit(lambda: capacity_reference.initialize(91, C))()
        objective = lambda module, cfg, p: jnp.sum(module.forward(p, s, g, a, n, cfg)**2)
        expected = jax.jit(jax.value_and_grad(lambda p: objective(capacity_reference, C, p)))(p)
        for config in (C, {**C, 'encoder_depth': 2, 'observation_attention': 'causal','readout_pooling':'learned','policy_spatial_bias':False}):
            actual_p = jax.jit(lambda: causal.initialize(91, config))()
            for key in p:
                np.testing.assert_array_equal(actual_p[key], p[key], err_msg=key)
            actual = jax.jit(jax.value_and_grad(lambda p: objective(causal, config, p)))(actual_p)
            for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(x, y)

    def test_deeper_stem_against_independent_forward_and_gradient(self):
        s = jnp.asarray(fixture()[0][:1, :1])
        config = {**C, 'encoder_depth': 4}
        p = jax.jit(lambda: causal.initialize(91, config))()
        stem = {k: v for k, v in p.items() if k.startswith('encoder.stem')}
        expected = stem_reference(jax.tree.map(np.asarray, stem), np.asarray(s), 4, np)
        np.testing.assert_allclose(causal.spatial_features(stem, s, config), expected, atol=5e-6, rtol=4e-5)
        weights = jnp.asarray(np.random.default_rng(611).normal(size=expected.shape).astype(np.float32))
        actual = jax.jit(jax.value_and_grad(lambda p, x: jnp.sum(causal.spatial_features(p, x, config)*weights), argnums=(0, 1)))(stem, s)
        reference = jax.jit(jax.value_and_grad(lambda p, x: jnp.sum(stem_reference(p, x, 4, jnp)*weights), argnums=(0, 1)))(stem, s)
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
            np.testing.assert_allclose(x, y, atol=2e-5, rtol=4e-4)
        for gradient in actual[1][0].values():
            self.assertGreater(float(jnp.linalg.norm(gradient)), 1e-6)

    def test_optional_local_policy_refactor_preserves_qualified_readout_graph(self):
        s,g,a,n=map(jnp.asarray,fixture())
        for mode in ('causal','board'):
            c={**C,'readout_pooling':'board_mean','observation_attention':mode}
            p=jax.jit(lambda:readout_reference.initialize(91,c))()
            objective=lambda module,p:jnp.sum(module.forward(p,s,g,a,n,c)**2)
            expected=jax.jit(jax.value_and_grad(lambda p:objective(readout_reference,p)))(p)
            actual=jax.jit(jax.value_and_grad(lambda p:objective(causal,p)))(p)
            for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
                np.testing.assert_array_equal(x,y)

    def test_board_visibility_against_frame_enumeration(self):
        for stride in (6, 38, 198):
            length = stride * 3 + 3
            mask = ObservationMask((length+8, length+8), stride, length)
            expected = np.zeros(mask.shape, bool)
            for q in range(mask.shape[0]):
                expected[q, allowed_keys(q, stride, length)] = True
            np.testing.assert_array_equal(mask[:, :], expected)
            np.testing.assert_array_equal(mask[2:9, 1:11], expected[2:9, 1:11])
            self.assertEqual(mask, ObservationMask(mask.shape, stride, length))
            self.assertEqual(hash(mask), hash(ObservationMask(mask.shape, stride, length)))
            self.assertNotEqual(mask, ObservationMask(mask.shape, stride, length-1))

    def test_board_attention_against_numpy_and_forbidden_key_gradients(self):
        r = np.random.default_rng(11)
        q = r.normal(size=(2, 7, 4, 8)).astype(np.float32)
        k = r.normal(size=(2, 19, 2, 8)).astype(np.float32)
        v = r.normal(size=k.shape).astype(np.float32)
        positions = np.asarray([[5,6,7,8,9,10,11], [11,12,13,14,15,16,17]])
        lengths = np.asarray([11,17])
        kk, vv = np.repeat(k, 2, axis=2), np.repeat(v, 2, axis=2)
        expected = np.zeros_like(q)
        for b in range(2):
            for t, position in enumerate(positions[b]):
                indices = allowed_keys(int(position), 6, int(lengths[b]))
                scores = np.einsum('hd,shd->hs', q[b,t], kk[b,indices])/np.sqrt(8)
                w = np.exp(scores-scores.max(-1,keepdims=True)); w /= w.sum(-1,keepdims=True)
                expected[b,t] = np.einsum('hs,shd->hd', w, vv[b,indices])
        got = causal.dense_attention(*map(jnp.asarray,(q,k,v,positions,lengths)), 6)
        np.testing.assert_allclose(got, expected, atol=3e-6, rtol=3e-6)
        def target(k, v):
            return causal.dense_attention(jnp.asarray(q), k, v, jnp.asarray(positions), jnp.asarray(lengths), 6)[0,1].sum()
        grad_k, grad_v = jax.jit(jax.grad(target, argnums=(0,1)))(jnp.asarray(k), jnp.asarray(v))
        # Query at frame1 patch0 sees all its board/readout but not action11.
        self.assertGreater(float(jnp.linalg.norm(grad_v[0,10])), 1e-5)
        np.testing.assert_array_equal(grad_k[0,11:], 0)
        np.testing.assert_array_equal(grad_v[0,11:], 0)
        np.testing.assert_array_equal(grad_v[1], 0)

    def test_full_model_causality_and_ragged_cache_for_each_intervention(self):
        s, g, a, n = map(jnp.asarray, fixture())
        for depth, mode, pool, local in ((4,'causal','learned',False), (2,'board','learned',False), (4,'board','learned',False),
                                  (2,'causal','board_mean',False), (4,'board','board_mean',False),
                                  (2,'causal','board_mean',True), (4,'board','board_mean',True)):
            with self.subTest(depth=depth, mode=mode,pool=pool,local=local):
                c = {**C, 'encoder_depth': depth, 'observation_attention': mode,'readout_pooling':pool,'policy_spatial_bias':local}
                p = jax.jit(lambda: causal.initialize(91, c))()
                if local:p={**p,'head.local.weight':jnp.linspace(-.2,.25,causal.encoder_channels(c))[:,None],
                            'head.local.bias':jnp.asarray(.125,jnp.float32)}
                run = jax.jit(lambda s,g,a: causal.forward(p,s,g,a,n,c))
                base = run(s,g,a)
                changed = run(s.at[0,3:].add(.3).at[1].set(0), g.at[0,3:].add(1).at[1].set(0), a.at[0,2:].set(0).at[1].set(9))
                np.testing.assert_array_equal(base[0,:3],changed[0,:3])
                self.assertGreater(float(jnp.max(jnp.abs(run(s.at[0,2,1,1,1].add(1),g,a)[0,2]-base[0,2]))),1e-6)
                counts = jnp.asarray([2,3])
                _, cache = causal.forward(p,s[:,:3],g[:,:3],a[:,:3],counts,c,with_cache=True,network_version=9)
                got, updated = causal.append_move(p,cache,jnp.asarray([a[0,1],a[1,2]]),jnp.stack([s[0,2],s[1,3]]),jnp.stack([g[0,2],g[1,3]]),c,attention_positions=4,network_version=9)
                full, expected = causal.forward(p,s[:,:4],g[:,:4],a[:,:4],counts+1,c,with_cache=True,network_version=9)
                np.testing.assert_allclose(got,jnp.stack([full[0,2],full[1,3]]),atol=2e-5,rtol=2e-5)
                for key in expected:
                    np.testing.assert_allclose(updated[key],expected[key],atol=2e-5,rtol=2e-5,err_msg=key)
                opening, _ = causal.first_move(p,s[:,0],g[:,0],c,network_version=9)
                np.testing.assert_allclose(opening,base[:,0],atol=2e-5,rtol=2e-5)
                frozen = causal.append_move(p,cache,jnp.asarray([a[0,1],a[1,2]]),s[:,0],g[:,0],c,attention_positions=4,active=jnp.zeros(2,bool),network_version=9)[1]
                for key in cache:
                    np.testing.assert_array_equal(cache[key],frozen[key])

    def test_pooled_readout_gives_current_observation_a_direct_residual_path(self):
        s,g,a,n=map(jnp.asarray,fixture())
        p=jax.jit(lambda:causal.initialize(91,C))()
        # Disable attention/FFN residual outputs: any board information must
        # now enter directly through the observation-conditioned readout.
        p={**p,'blocks.out.weight':jnp.zeros_like(p['blocks.out.weight']),
           'blocks.down.weight':jnp.zeros_like(p['blocks.down.weight'])}
        base=causal.forward(p,s,g,a,n,C)
        changed=causal.forward(p,s.at[0,2].set(0),g,a,n,C)
        np.testing.assert_array_equal(base,changed)
        c={**C,'readout_pooling':'board_mean'}
        pooled=causal.forward(p,s,g,a,n,c)
        changed=causal.forward(p,s.at[0,2].set(0),g,a,n,c)
        self.assertGreater(float(jnp.max(jnp.abs(pooled[0,2]-changed[0,2]))),1e-5)
        np.testing.assert_array_equal(pooled[0,:2],changed[0,:2])
        np.testing.assert_array_equal(pooled[1],changed[1])
        gradient=jax.jit(jax.grad(lambda spatial:jnp.sum(causal.forward(p,spatial,g,a,n,c)[0,2]**2)))(s)
        self.assertGreater(float(jnp.linalg.norm(gradient[0,2])),1e-6)
        np.testing.assert_array_equal(gradient[0,3:],0)

    def test_local_policy_matches_spatial_oracle_and_preserves_initial_policy(self):
        s,g,a,n=map(jnp.asarray,fixture());c={**C,'readout_pooling':'board_mean'}
        parent=jax.jit(lambda:causal.initialize(91,c))();base=causal.forward(parent,s,g,a,n,c)
        local={**c,'policy_spatial_bias':True};p=jax.jit(lambda:causal.initialize(91,local))()
        self.assertEqual(set(p)-set(parent),{'head.local.weight','head.local.bias'})
        for key in parent:np.testing.assert_array_equal(p[key],parent[key])
        np.testing.assert_array_equal(causal.forward(p,s,g,a,n,local),base)
        p={**p,'head.local.weight':jnp.linspace(-.2,.25,64)[:,None],'head.local.bias':jnp.asarray(.125)}
        stem={k:np.asarray(v) for k,v in p.items() if k.startswith('encoder.stem')}
        features=stem_reference(stem,np.asarray(s),2,np)
        expected=(features@np.asarray(p['head.local.weight']))[...,0]+.125
        expected=np.concatenate([expected.reshape(2,5,9),np.zeros((2,5,1),np.float32)],-1)
        expected=np.where(np.arange(5)[None,:,None]<np.asarray(n)[:,None,None],expected,0)
        got=causal.forward(p,s,g,a,n,local)-base
        np.testing.assert_allclose(got,expected,atol=2e-6,rtol=2e-5)
        np.testing.assert_array_equal(got[...,-1],0)

    def test_local_policy_closed_form_gradients_and_point_correspondence(self):
        c={**C,'policy_spatial_bias':True};p=jax.jit(lambda:causal.initialize(91,c))()
        rng=np.random.default_rng(741);x=jnp.asarray(rng.normal(size=(2,32)).astype(np.float32))
        features=jnp.asarray(rng.normal(size=(2,3,3,64)).astype(np.float32))
        weights=jnp.asarray(rng.normal(size=(2,10)).astype(np.float32))
        w=jnp.linspace(-.2,.25,64)[:,None];bias=jnp.asarray(.125,jnp.float32)
        def objective(w,bias,features):
            return jnp.sum(causal.policy({**p,'head.local.weight':w,'head.local.bias':bias},x,3,c,features)*weights)
        dw,db,df=jax.jit(jax.grad(objective,argnums=(0,1,2)))(w,bias,features)
        expected_w=np.einsum('bpc,bp->c',np.asarray(features).reshape(2,9,64),np.asarray(weights[:,:9]))[:,None]
        expected_f=np.asarray(weights[:,:9]).reshape(2,3,3,1)*np.asarray(w[:,0])
        np.testing.assert_allclose(dw,expected_w,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(db,np.asarray(weights[:,:9]).sum(),atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(df,expected_f,atol=2e-6,rtol=2e-6)
        # Zero initialization keeps the old logits but permits learning this
        # projection immediately; no auxiliary policy or extra loss is used.
        initial_dw=jax.grad(objective)(jnp.zeros_like(w),jnp.zeros_like(bias),features)
        self.assertGreater(float(jnp.linalg.norm(initial_dw)),1e-4)
        initial=causal.policy({**p,'head.local.weight':w,'head.local.bias':bias},x,3,c,features)
        changed=causal.policy({**p,'head.local.weight':w,'head.local.bias':bias},x,3,c,features.at[0,1,2,0].add(1))
        expected=np.zeros((2,10),np.float32);expected[0,5]=float(w[0,0])
        np.testing.assert_allclose(changed-initial,expected,atol=2e-6,rtol=2e-6)

    def test_encoder_recomputation_preserves_full_model_and_gradients(self):
        s,g,a,n=map(jnp.asarray,fixture())
        for local,dtype in ((False,'float32'),(True,'float32'),(False,'bfloat16'),(True,'bfloat16')):
            c={**C,'encoder_depth':6,'policy_spatial_bias':local,'readout_pooling':'board_mean','dtype':dtype}
            p=jax.jit(lambda:causal.initialize(91,c))()
            if local:p={**p,'head.local.weight':jnp.linspace(-.2,.25,64)[:,None]}
            def objective(p,s,config):return jnp.sum(causal.forward(p,s,g,a,n,config)**2)
            expected=jax.jit(jax.value_and_grad(lambda p,s:objective(p,s,c),argnums=(0,1)))(p,s)
            remat={**c,'encoder_rematerialize':True}
            actual=jax.jit(jax.value_and_grad(lambda p,s:objective(p,s,remat),argnums=(0,1)))(p,s)
            np.testing.assert_allclose(actual[0],expected[0],atol=1e-4 if dtype=='bfloat16' else 0,rtol=1e-4 if dtype=='bfloat16' else 0)
            for x,y in zip(jax.tree.leaves(actual[1]),jax.tree.leaves(expected[1])):
                np.testing.assert_allclose(x,y,atol=.01 if dtype=='bfloat16' else 2e-5,rtol=.01 if dtype=='bfloat16' else 3e-5)


if __name__ == '__main__':
    unittest.main()
