"""Position isolation, native-state inputs and separated objective gradients."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import model


class BoardHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c={'size':3,'komi':.5,'width':16,'heads':4,'blocks':1,'max_tokens':8,'dtype':'float32',
               'board_width':8,'board_blocks':1,'board_mode':'exact','expert_temperature':1.,'behavior_temperature':1.}
        cls.p=model.initialize(271,cls.c)

    def batch(self):
        tokens=np.array([[10,0,1,9,11,11],[10,8,1,9,11,11]],np.int32)
        shape=tokens.shape;stones=np.zeros((*shape,9),np.uint8)
        stones[0,1:,0]=1;stones[0,2:,1]=2;stones[1,1:,8]=1;stones[1,2:,1]=2
        expert=np.zeros(shape,np.float32);expert[0,:4]=1
        observed=np.zeros(shape,np.float32);observed[1,:4]=1
        return {'tokens':tokens,'lengths':np.array([3,3],np.int32),'stones':stones,
                'policies':np.eye(10,dtype=np.float32)[np.array([[0,2,9,9,0,0],[8,1,9,9,0,0]])],
                'observed':np.array([[0,1,9,9,0,0],[8,1,9,9,0,0]],np.int32),
                'values':np.zeros(shape,np.float32),'legal':np.ones((*shape,10),bool),
                'expert_mask':expert,'behavior_mask':observed,'value_mask':expert.copy()}

    def test_future_tokens_states_and_other_games_cannot_change_a_prefix(self):
        batch=self.batch();first=model.predictions(self.p,batch,self.c)
        batch['tokens'][0,3:]=7;batch['tokens'][1,:]=5
        batch['stones'][0,3:]=2;batch['stones'][1,:]=1
        second=model.predictions(self.p,batch,self.c)
        for a,b in zip(first,second):np.testing.assert_array_equal(a[0,:3],b[0,:3])

    def test_board_observation_affects_only_its_own_position(self):
        batch=self.batch();first=model.predictions(self.p,batch,self.c)
        batch['stones'][0,1,4]=2
        second=model.predictions(self.p,batch,self.c)
        for a,b in zip(first,second):
            np.testing.assert_array_equal(np.asarray(a)[0,[0,2,3,4,5]],np.asarray(b)[0,[0,2,3,4,5]])
            np.testing.assert_array_equal(a[1],b[1])
        self.assertFalse(np.array_equal(first[0][0,1],second[0][0,1]))

    def test_behavior_cannot_train_either_encoder_or_expert_heads(self):
        batch=self.batch();batch['expert_mask'][:]=0;batch['value_mask'][:]=0
        grad=jax.grad(lambda p:model.losses(p,batch,self.c)[0])(self.p)
        private=0.
        for key,tree in grad.items():
            if key.startswith('behavior'):private+=sum(float(np.abs(a).sum()) for a in jax.tree.leaves(tree))
            else:self.assertTrue(all(np.count_nonzero(a)==0 for a in jax.tree.leaves(tree)),key)
        self.assertGreater(private,0.)
        self.assertGreater(float(np.abs(grad['behavior_board_hidden']).sum()),0.)

    def test_expert_trains_both_encoders_without_behavior_or_capped_outcomes(self):
        batch=self.batch();batch['behavior_mask'][:]=0
        first=model.losses(self.p,batch,self.c)[0];batch['values'][1,:]=100
        np.testing.assert_array_equal(first,model.losses(self.p,batch,self.c)[0])
        grad=jax.grad(lambda p:model.losses(p,batch,self.c)[0])(self.p)
        for key,tree in grad.items():
            if key.startswith('behavior'):self.assertTrue(all(np.count_nonzero(a)==0 for a in jax.tree.leaves(tree)),key)
        for key in ('tokens','play','board_embedding','board_query'):
            self.assertGreater(sum(float(np.abs(a).sum()) for a in jax.tree.leaves(grad[key])),0.,key)

    def test_single_leaf_evaluation_uses_the_matching_player_and_board(self):
        batch=self.batch();all_outputs=model.predictions(self.p,batch,self.c)
        h,_=model.prefill(self.p,batch['tokens'],batch['lengths'],self.c)
        counts=model.historical_counts(batch['tokens'],batch['lengths'],10)
        for t in (0,1,3):
            board=model.board_features(self.p,batch['stones'][0:1,t:t+1],jnp.array([[1+t%2]]),self.c)
            outputs=model.head_outputs(self.p,h[0:1,t:t+1],counts[0:1,t:t+1,t%2],board)
            for actual,expected in zip(outputs,all_outputs):
                np.testing.assert_allclose(actual[0,0],expected[0,t],rtol=2e-5,atol=2e-5)


if __name__=='__main__':unittest.main()
