import gc
import hashlib
import os
from pathlib import Path
import unittest

LIBRARY = os.environ.get('GOZERO_NATIVE_LIBRARY')
if LIBRARY:
    import numpy as np
    from gozero.native import Actors, load_library


def config(size=1):
    return {'size':size,'komi':0.5,'history':2,'games':2,'workers':2,'worker_cpus':[],
            'simulations':4,'cpuct':1.5,'max_search_edges':10000,'max_game_moves':32,
            'dirichlet_alpha':0.3,'dirichlet_fraction':0.25,'temperature_early':1.0,
            'temperature_late':0.0,'temperature_moves':8,'seed':27,'actor_offset':0}


@unittest.skipUnless(LIBRARY, 'Set GOZERO_NATIVE_LIBRARY to a qualified native extension')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=Path(LIBRARY)
        with cls.library.open('rb') as stream:
            cls.sha=hashlib.file_digest(stream,'sha256').hexdigest()

    def finish_search(self,actors,network):
        batch=actors.start(network)
        while batch.active_count:
            batch=actors.evaluate(batch,np.zeros((2,actors.config['size']**2+1),np.float32),np.zeros(2,np.float32))
        return actors.commit()

    def test_bulk_board_observations_match_the_real_katago_scoring_fixture(self):
        import json
        module=load_library(self.library,self.sha)
        self.assertEqual(module.OBSERVATION_REPLAY_ABI_VERSION,1)
        fixture=json.loads((Path(__file__).resolve().parents[1]/'eval/fixtures/pass_alive_area.json').read_text())
        columns='ABCDEFGHJKLMNOPQRSTUVWXYZ';vertices=fixture['action_vertices']
        actions=np.array([81 if v.lower()=='pass' else (9-int(v[1:]))*9+columns.index(v[0].upper()) for v in vertices],np.int32)
        config={'size':9,'komi':7.5,'scoring':'pass_alive_area'}
        stones,legal,outcomes=module.replay_observations(json.dumps(config),actions,np.array([0,len(actions)],np.int64))
        self.assertEqual(stones.dtype,np.uint8);self.assertEqual(legal.dtype,np.bool_)
        stones=stones.reshape(-1,81);legal=legal.reshape(-1,82)
        game=module.Game(json.dumps({**config,'history':2,'simulations':0,'cpuct':1.5,'max_search_edges':1000}))
        for ply,a in enumerate(actions):
            np.testing.assert_array_equal(stones[ply],game.state()[4])
            self.assertEqual(np.flatnonzero(legal[ply]).tolist(),game.legal())
            game.play(1+ply%2,int(a))
        self.assertEqual(json.loads(outcomes),[{'terminal':True,'white_score':fixture['katago_adjudicated_white_minus_black']}])
        board='\n'.join(''.join('.XO'[int(x)] for x in row) for row in stones[-1].reshape(9,9))
        self.assertEqual(board,fixture['board'])
        _,_,capped=module.replay_observations(json.dumps(config),actions[:1],np.array([0,1],np.int64))
        self.assertEqual(json.loads(capped),[{'terminal':False,'white_score':None}])

    def test_causal_leaf_history_reconstructs_each_search_and_rejects_stale_tickets(self):
        import json
        module=load_library(self.library,self.sha)
        self.assertEqual(module.CAUSAL_GAME_ABI_VERSION,1)
        for gumbel in (None,{'max_considered_actions':16,'value_scale':0.1,'maxvisit_init':50.,'rescale_values':True,'gumbel_scale':0.}):
            c={'size':3,'komi':0.5,'history':2,'simulations':64,'cpuct':1.5 if gumbel is None else 0.,'max_search_edges':10000,'gumbel':gumbel}
            game=module.Game(json.dumps(c));game.play(1,0);game.play(2,1)
            request,features=game.start(27);old=request;deepest=0;requests=0
            while request is not None:
                tape=game.request_history(request);self.assertEqual(tape.dtype,np.int32)
                self.assertEqual(tape[:2].tolist(),[0,1]);deepest=max(deepest,len(tape)-2)
                replay=module.Game(json.dumps({**c,'simulations':0,'gumbel':None}))
                for ply,a in enumerate(tape):replay.play(1+ply%2,int(a))
                _,expected=replay.start(0);np.testing.assert_array_equal(features,expected)
                tape[0]=9;self.assertEqual(game.request_history(request)[0],0)
                wrong=(request[0],request[1]+1,request[2],request[3])
                with self.assertRaises(ValueError):game.request_history(wrong)
                if requests:
                    with self.assertRaises(ValueError):game.request_history(old)
                request,features=game.evaluate(request,np.zeros(10,np.float32),0.);requests+=1
            chosen,_,_,simulations,_,_=game.finish();self.assertEqual(simulations,64);self.assertGreaterEqual(deepest,2)
            with self.assertRaises(ValueError):game.request_history(old)
            game.play(1,chosen);request,_=game.start(28)
            self.assertEqual(game.request_history(request).tolist(),[0,1,chosen])

    def test_completed_search_inspection_preserves_finish_and_terminal_backup(self):
        import json
        module=load_library(self.library,self.sha)
        self.assertEqual(module.SEARCH_INSPECTION_ABI_VERSION,1)
        for rescale in (False,True):
            c={'size':3,'komi':7.5,'history':2,'simulations':16,'cpuct':0.,'max_search_edges':10000,
               'scoring':'pass_alive_area','gumbel':{'max_considered_actions':16,'maxvisit_init':50.,
                    'value_scale':.1,'rescale_values':rescale,'gumbel_scale':0.}}
            game=module.Game(json.dumps(c));control=module.Game(json.dumps(c))
            with self.assertRaises(ValueError):game.inspect_search()
            results=[]
            for current in (game,control):
                current.play(1,9);before=current.state();request,_=current.start(13)
                with self.assertRaises(ValueError):current.inspect_search()
                while request is not None:request,_=current.evaluate(request,np.zeros(10,np.float32),-.875)
                if current is game:
                    a=current.inspect_search();self.assertEqual(a,current.inspect_search());d=json.loads(a)
                    self.assertEqual(d['root_visits'],16);self.assertEqual(sum(d['visits']),16)
                    self.assertIsNotNone(d['terminal_child_values'][9]);self.assertEqual(d['terminal_child_values'][9],-1.)
                    self.assertEqual(d['value_sums'][9],float(d['visits'][9]))
                    self.assertEqual(d['action_values'][9],1.)
                    for action,n in enumerate(d['visits']):
                        if n:self.assertEqual(d['action_values'][action],float(np.float32(d['value_sums'][action]/n)))
                    np.testing.assert_array_equal(current.state()[4],before[4]);self.assertEqual(current.state()[:4],before[:4])
                results.append(current.finish())
                with self.assertRaises(ValueError):current.inspect_search()
            self.assertEqual(d['chosen'],results[0][0]);np.testing.assert_array_equal(np.asarray(d['policy'],np.float32),results[0][1])
            self.assertEqual(results[0][0],results[1][0]);np.testing.assert_array_equal(results[0][1],results[1][1]);self.assertEqual(results[0][2:],results[1][2:])

    def test_dual_trace_batch_rejection_is_atomic_and_terminal_reset_has_new_identity(self):
        import json
        module=load_library(self.library,self.sha)
        engine=module.DualTraceActors(json.dumps({'size':3,'komi':0.5,'history':2,
            'scoring':'pass_alive_area','max_game_moves':36}),2)
        tickets,tokens,lengths,legal=engine.start([7,11],[0,0],4)
        self.assertEqual(tokens.shape,(2,41));np.testing.assert_array_equal(tokens[:,0],[10,10])
        np.testing.assert_array_equal(lengths,[0,0])
        self.assertEqual(legal.shape,(2,10));self.assertTrue(legal[:,-1].all())
        actions=np.full((2,2,1,4),9,np.int32)
        logits=np.zeros((2,2,1,4,10),np.float32);logits[...,9]=1.
        noise=np.zeros((2,4,10),np.float32)
        wrong=json.loads(tickets);wrong[1]['contexts'][0]=1
        with self.assertRaisesRegex(ValueError,'stale trace'):
            engine.resolve(json.dumps(wrong),actions,logits,noise)
        self.assertEqual(engine.tapes(),[[],[]])
        broken=logits.copy();broken[1,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):engine.resolve(tickets,actions,broken,noise)
        self.assertEqual(engine.tapes(),[[],[]])
        result=json.loads(engine.resolve(tickets,actions,logits,noise))
        self.assertTrue(all(x['actions']==[9,9] and x['stop']=='terminal' and x['terminal_white_score']==0.5 for x in result))
        with self.assertRaises(ValueError):engine.resolve(tickets,actions,logits,noise)
        reset,tokens,lengths,_=engine.start([7,11],[0,0],4)
        old=json.loads(tickets);new=json.loads(reset)
        self.assertTrue(all(b['episode']==a['episode']+1 and b['generation']>a['generation'] for a,b in zip(old,new)))
        np.testing.assert_array_equal(lengths,[0,0]);self.assertEqual(engine.tapes(),[[],[]])
        engine.cancel()
        with self.assertRaises(ValueError):engine.resolve(reset,actions,logits,noise)

    def test_dual_trace_cap_is_not_a_terminal_result_and_dimensions_are_strict(self):
        import json
        module=load_library(self.library,self.sha)
        engine=module.DualTraceActors(json.dumps({'size':3,'komi':0.5,'history':2,
            'scoring':'pass_alive_area','max_game_moves':1}),1)
        tickets,_,_,_=engine.start([1,1],[0,0],2)
        actions=np.zeros((1,2,1,2),np.int32)
        logits=np.zeros((1,2,1,2,10),np.float32);logits[...,0]=1.
        noise=np.zeros((1,2,10),np.float32)
        with self.assertRaises(ValueError):engine.resolve(tickets,actions,logits[:,:,:,:,:5],noise)
        self.assertEqual(engine.tapes(),[[]])
        result=json.loads(engine.resolve(tickets,actions,logits,noise))[0]
        self.assertEqual(result['actions'],[0]);self.assertEqual(result['stop'],'move_limit')
        self.assertIsNone(result['terminal_white_score'])

    def test_dual_trace_root_mask_matches_exact_legality_and_declared_sampler(self):
        import json
        module=load_library(self.library,self.sha)
        engine=module.DualTraceActors(json.dumps({'size':3,'komi':0.5,'history':2,
            'scoring':'pass_alive_area','max_game_moves':36,'root_legal_mask':True}),1)
        noise=np.zeros((1,1,10),np.float32)
        actions=np.zeros((1,2,1,1),np.int32);logits=np.zeros((1,2,1,1,10),np.float32);logits[...,0]=2.
        ticket,_,_,_=engine.start([1,1],[0,0],1)
        engine.resolve(ticket,actions,logits,noise)
        ticket,_,_,legal=engine.start([1,1],[0,0],1)
        self.assertFalse(legal[0,0]);self.assertTrue(legal[0,1]);self.assertTrue(legal[0,9])
        logits[...,1]=1.
        with self.assertRaisesRegex(ValueError,'declared sampler'):engine.resolve(ticket,actions,logits,noise)
        self.assertEqual(engine.tapes(),[[0]])
        actions.fill(1);result=json.loads(engine.resolve(ticket,actions,logits,noise))[0]
        self.assertEqual(result['actions'],[1]);self.assertEqual(result['legality_corrections'],0)

    def test_state_trace_capture_cutoff_root_binding_and_batch_atomicity(self):
        import json
        module=load_library(self.library,self.sha)
        self.assertEqual(module.STATE_TRACE_ABI_VERSION,1)
        engine=module.DualTraceActors(json.dumps({'size':3,'komi':0.5,'history':2,
            'scoring':'pass_alive_area','max_game_moves':36,'root_legal_mask':True}),2)
        tickets,tokens,lengths,legal,stones=engine.start_with_states([1,1],[0,0],4)
        self.assertEqual(stones.shape,(2,9));self.assertEqual(stones.dtype,np.uint8)
        np.testing.assert_array_equal(stones,np.zeros((2,9),np.uint8))
        # Returned arrays are owned; tampering cannot change the native board.
        stones[1,0]=2
        actions=np.broadcast_to(np.array([1,0,3,8],np.int32),(2,2,1,4)).copy()
        logits=np.eye(10,dtype=np.float32)[actions];noise=np.zeros((2,4,10),np.float32)
        states=np.zeros((2,2,1,4,9),np.uint8)
        for depth in range(1,4):
            for previous in range(depth):states[:,:,:,depth,int(actions[0,0,0,previous])]=1+previous%2
        broken=states.copy();broken[1,1,0,0,0]=2
        with self.assertRaisesRegex(ValueError,'root state'):
            engine.resolve_with_states(tickets,actions,logits,noise,broken)
        self.assertEqual(engine.tapes(),[[],[]])
        with self.assertRaises(ValueError):
            engine.resolve_with_states(tickets,actions,logits,noise,states[...,:8].copy())
        self.assertEqual(engine.tapes(),[[],[]])
        results=json.loads(engine.resolve_with_states(tickets,actions,logits,noise,states))
        self.assertTrue(all(r['actions']==[1,0,3] and r['stop']=='state_mismatch' for r in results))
        self.assertEqual(engine.tapes(),[[1,0,3],[1,0,3]])
        with self.assertRaisesRegex(ValueError,'pending'):
            engine.resolve_with_states(tickets,actions,logits,noise,states)
        _,_,lengths,_,stones=engine.start_with_states([1,1],[0,0],4)
        np.testing.assert_array_equal(lengths,[3,3]);self.assertTrue((stones[:,0]==0).all())
        self.assertTrue((stones[:,1]==1).all());self.assertTrue((stones[:,3]==1).all())
        engine.cancel()

    def test_bounded_trace_exact_work_inactive_terminal_state_and_reactivation(self):
        import json
        module=load_library(self.library,self.sha)
        self.assertEqual(module.BOUNDED_STATE_TRACE_ABI_VERSION,1)
        engine=module.BoundedTraceActors(json.dumps({'size':3,'komi':0.5,'history':2,
            'scoring':'pass_alive_area','max_game_moves':3,'root_legal_mask':True}),2)
        actions=np.full((2,2,1,4),9,np.int32);logits=np.eye(10,dtype=np.float32)[actions]
        noise=np.zeros((2,4,10),np.float32);states=np.zeros((2,2,1,4,9),np.uint8)
        ticket,_,_,_,_=engine.start([1,1],[0,0],4,[True,True])
        bad=states.copy();bad[1,1,0,0,0]=1
        with self.assertRaisesRegex(ValueError,'root state'):
            engine.resolve(ticket,actions,logits,noise,bad,[1,2])
        with self.assertRaisesRegex(ValueError,'allowance'):
            engine.resolve(ticket,actions,logits,noise,states,[1,5])
        result=json.loads(engine.resolve(ticket,actions,logits,noise,states,[1,2]))
        self.assertEqual(result[0]['actions'],[9]);self.assertEqual(result[0]['stop'],'work_limit')
        self.assertEqual(result[1]['actions'],[9,9]);self.assertEqual(result[1]['stop'],'terminal')
        before=json.loads(engine.inspect());self.assertEqual([g['episode']for g in before],[0,0])
        ticket,tokens,lengths,legal,stones=engine.start([1,1],[0,0],4,[True,False])
        self.assertIsNone(json.loads(ticket)[1]);np.testing.assert_array_equal(lengths,[1,0])
        np.testing.assert_array_equal(np.flatnonzero(legal[1]),[9])
        forged=json.loads(ticket);forged[0]=None
        with self.assertRaisesRegex(ValueError,'stale'):
            engine.resolve(json.dumps(forged),actions,logits,noise,states,[0,0])
        result=json.loads(engine.resolve(ticket,actions,logits,noise,states,[1,0]))
        self.assertEqual(result[0]['stop'],'terminal');self.assertEqual(result[1]['actions'],[])
        both=json.loads(engine.inspect());self.assertEqual(both[1],before[1])
        ticket,_,_,_,_=engine.start([1,1],[0,0],4,[False,False])
        engine.resolve(ticket,actions,logits,noise,states,[0,0])
        self.assertEqual(json.loads(engine.inspect()),both)
        ticket,_,lengths,_,_=engine.start([1,1],[0,0],4,[False,True])
        np.testing.assert_array_equal(lengths,[0,0])
        engine.resolve(ticket,actions,logits,noise,states,[0,1])
        final=json.loads(engine.inspect());self.assertEqual(final[0],both[0])
        self.assertEqual(final[1]['episode'],1);self.assertEqual(final[1]['moves'],[9])

    def test_bounded_trace_cap_after_capture_has_no_outcome_and_does_not_overshoot(self):
        import json
        module=load_library(self.library,self.sha)
        engine=module.BoundedTraceActors(json.dumps({'size':3,'komi':0.5,'history':2,
            'scoring':'pass_alive_area','max_game_moves':3,'root_legal_mask':True}),1)
        actions=np.broadcast_to(np.array([1,0,3,8],np.int32),(1,2,1,4)).copy()
        logits=np.eye(10,dtype=np.float32)[actions];noise=np.zeros((1,4,10),np.float32)
        states=np.zeros((1,2,1,4,9),np.uint8)
        for depth in range(1,4):
            for previous in range(depth):states[:,:,:,depth,int(actions[0,0,0,previous])]=1+previous%2
        ticket,_,_,_,_=engine.start([1,1],[0,0],4,[True])
        result=json.loads(engine.resolve(ticket,actions,logits,noise,states,[3]))[0]
        self.assertEqual(result['actions'],[1,0,3]);self.assertEqual(result['stop'],'move_limit')
        self.assertIsNone(result['terminal_white_score'])
        before=json.loads(engine.inspect());self.assertEqual(before[0]['stones'][0],0)
        ticket,_,_,_,_=engine.start([1,1],[0,0],4,[False])
        engine.resolve(ticket,actions,logits,noise,states,[0])
        self.assertEqual(json.loads(engine.inspect()),before)
        ticket,_,lengths,_,_=engine.start([1,1],[0,0],4,[True])
        self.assertEqual(lengths.tolist(),[0]);engine.cancel()
        self.assertEqual(json.loads(engine.inspect())[0]['episode'],1)

    def test_terminal_outcomes_metadata_and_array_ownership(self):
        with Actors(config(),self.library,self.sha) as actors:
            self.assertEqual(len(self.finish_search(actors,1).outcomes),0)
            rows=self.finish_search(actors,2)
            np.testing.assert_array_equal(rows.outcomes,[-1,1,-1,1])
            np.testing.assert_array_equal(rows.metadata[:,1],[1,2,1,2])
            np.testing.assert_array_equal(rows.policies,[[0,1]]*4)
            self.assertEqual(rows.features.shape,(4,1,1,8))
            np.testing.assert_array_equal(rows.ownership,np.zeros((4,1,1),np.float32))
            self.assertFalse(rows.ownership.flags.writeable)
            self.assertFalse(rows.features.flags.writeable)
            self.assertEqual(len(rows.games),2)
            saved=rows.features.copy()
        del actors;gc.collect()
        np.testing.assert_array_equal(rows.features,saved)

    def test_root_prefetch_binding_preserves_targets_and_rejects_bad_batches(self):
        import json
        c=config(3)
        c.update(simulations=16,cpuct=0.,fpu_reduction=None,dirichlet_fraction=0.,
                 temperature_early=0.,temperature_late=0.,temperature_moves=0,
                 gumbel={'max_considered_actions':16,'value_scale':.1,'maxvisit_init':50.,
                         'rescale_values':True,'gumbel_scale':1.})

        def predict(batch):
            rows=batch.features.reshape((len(batch.active),-1))
            signature=np.sum(rows*np.arange(rows.shape[1],dtype=np.float32),axis=1)
            logits=np.sin(signature[:,None]*np.arange(1,11,dtype=np.float32)[None,:]*.01)
            return logits,np.sin(signature*.001)

        with Actors(c,self.library,self.sha) as plain, Actors(c,self.library,self.sha) as cached:
            for turn in range(24):
                for engine,prefetch in ((plain,False),(cached,True)):
                    batch=engine.start(turn//3);first=True
                    while batch.active_count:
                        batch=engine.evaluate(batch,*predict(batch))
                        if first and prefetch and batch.active_count:
                            proposal=engine.prefetch(batch,16)
                            self.assertEqual(proposal.features.shape,(32,3,3,8))
                            self.assertFalse(proposal.features.flags.writeable)
                            logits,values=predict(proposal)
                            with self.assertRaises(ValueError):engine.evaluate_prefetch(proposal,logits.reshape(16,20),values)
                            bad=values.copy();bad[-1]=np.nan
                            with self.assertRaises(ValueError):engine.evaluate_prefetch(proposal,logits,bad)
                            with self.assertRaises(ValueError):engine.evaluate(batch,*predict(batch))
                            batch=engine.evaluate_prefetch(proposal,logits,values)
                            with self.assertRaises(ValueError):engine.evaluate_prefetch(proposal,logits,values)
                        first=False
                left,right=plain.commit(),cached.commit()
                self.assertEqual(left.games,right.games)
                for field in ('features','policies','outcomes','root_values','metadata','ownership'):
                    np.testing.assert_array_equal(getattr(left,field),getattr(right,field))
                left,right=json.loads(plain.checkpoint()),json.loads(cached.checkpoint())
                left.pop('round');right.pop('round')
                self.assertEqual(left,right)
            evaluated,used=cached.prefetch_counters()
            self.assertGreater(used,0);self.assertGreaterEqual(evaluated,used)

    def test_native_shape_checks_reject_equal_length_wrong_dimensions(self):
        with Actors(config(),self.library,self.sha) as actors:
            batch=actors.start(0)
            with self.assertRaises(ValueError):
                actors.evaluate(batch,np.zeros((1,4),np.float32),np.zeros(2,np.float32))
            with self.assertRaises(ValueError):
                actors.evaluate(batch,np.zeros((2,2),np.float32),np.full(2,np.nan,np.float32))
            actors.evaluate(batch,np.zeros((2,2),np.float32),np.zeros(2,np.float32))
            with self.assertRaises(ValueError):
                actors.evaluate(batch,np.zeros((2,2),np.float32),np.zeros(2,np.float32))

    def test_score_utility_changes_search_backups_and_preserves_win_labels(self):
        settings=config();settings['komi']=2.;settings['score_utility']={'factor':1.,'scale':2.}
        with Actors(settings,self.library,self.sha) as actors:
            self.finish_search(actors,1)
            rows=self.finish_search(actors,2)
            np.testing.assert_array_equal(rows.outcomes,[-1,1,-1,1])
            np.testing.assert_array_equal(rows.root_values,[-0.5625,0.75,-0.5625,0.75])
            self.assertTrue(all(game['white_score']==2. for game in rows.games))

    def test_truncated_games_cannot_enter_training_rows(self):
        settings=config(3);settings['max_game_moves']=1
        with Actors(settings,self.library,self.sha) as actors:
            rows=self.finish_search(actors,0)
            self.assertEqual(rows.features.shape,(0,3,3,8))
            self.assertEqual(rows.metadata.shape,(0,6))
            self.assertEqual(rows.ownership.shape,(0,3,3))
            self.assertTrue(all(game['truncated'] and game['white_score'] is None for game in rows.games))

    def test_binary_hash_and_configuration_are_checked(self):
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            load_library(self.library,'0'*64)
        settings=config();settings['misspelled_visits']=4
        with self.assertRaises(ValueError):
            Actors(settings,self.library,self.sha)

    def test_serialized_checkpoint_restores_rng_and_training_rows(self):
        settings=config(3)
        with Actors(settings,self.library,self.sha) as original:
            for network in range(11): self.finish_search(original,network)
            state=original.checkpoint()
            settings['workers']=1
            with Actors(settings,self.library,self.sha,checkpoint=state) as restored:
                for network in range(11,45):
                    a=self.finish_search(original,network); b=self.finish_search(restored,network)
                    self.assertEqual(a.games,b.games)
                    for field in ('features','policies','outcomes','root_values','metadata','ownership'):
                        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
            settings['seed']+=1
            with self.assertRaises(ValueError):
                Actors(settings,self.library,self.sha,checkpoint=state)

    def test_external_game_search_and_forced_passes(self):
        import json
        module=load_library(self.library,self.sha)
        game=module.Game(json.dumps({'size':1,'komi':0.5,'history':2,'simulations':8,'cpuct':1.5,'max_search_edges':1000}))
        for color in (1,2):
            request,features=game.start(27)
            while request is not None:
                self.assertEqual(request[1],27)
                request,features=game.evaluate(request,np.zeros(2,np.float32),0.0)
            best,policy,value,sims,nn,terminal=game.finish()
            self.assertEqual(best,1);self.assertEqual(sims,8)
            game.play(color,best)
        self.assertEqual(game.state()[:4],(1,1,True,0.5))

    def test_fpu_setting_reaches_native_search_and_changes_low_budget_allocation(self):
        import json
        module=load_library(self.library,self.sha)
        supports=[]
        for reduction in (None,0.2):
            game=module.Game(json.dumps({'size':3,'komi':0.5,'history':2,'simulations':4,'cpuct':0.,'fpu_reduction':reduction,'max_search_edges':1000}))
            request,features=game.start(1)
            while request is not None:
                # Black has value -0.8 at every state. With zero exploration,
                # zero FPU still explores fresh actions instead of revisiting;
                # network-value FPU reuses the evaluated action.
                value=-0.8 if features[4]>0.5 else 0.8
                request,features=game.evaluate(request,np.zeros(10,np.float32),value)
            _,policy,_,simulations,_,_=game.finish()
            self.assertEqual(simulations,4)
            supports.append(int(np.count_nonzero(policy)))
        self.assertEqual(supports,[4,1])

    def test_gumbel_binding_returns_completed_policy_not_visit_proportions(self):
        import json
        module=load_library(self.library,self.sha)
        settings={'size':3,'komi':0.5,'history':2,'simulations':4,'cpuct':0.,'max_search_edges':1000,
                  'gumbel':{'max_considered_actions':16,'value_scale':0.1,'maxvisit_init':50.,
                            'rescale_values':True,'gumbel_scale':0.}}
        game=module.Game(json.dumps(settings));request,features=game.start(1)
        while request is not None:
            request,features=game.evaluate(request,np.zeros(10,np.float32),0.)
        action,policy,_,simulations,neural,terminal=game.finish()
        self.assertEqual((action,simulations,neural,terminal),(0,4,5,0))
        np.testing.assert_allclose(policy,np.full(10,0.1,np.float32),atol=1e-7,rtol=0)
        settings['gumbel']['gumbel_scale']=1.
        with self.assertRaises(ValueError):module.Game(json.dumps(settings))

    def test_real_katago_game_retains_raw_tromp_taylor_scoring(self):
        import json
        fixture=json.loads((Path(__file__).resolve().parents[1]/'eval/fixtures/pass_alive_area.json').read_text())
        module=load_library(self.library,self.sha)
        game=module.Game(json.dumps({'size':9,'komi':7.5,'history':2,'simulations':0,'cpuct':1.5,'max_search_edges':1000}))
        columns='ABCDEFGHJKLMNOPQRSTUVWXYZ'
        for ply,vertex in enumerate(fixture['action_vertices']):
            action=81 if vertex.lower()=='pass' else (9-int(vertex[1:]))*9+columns.index(vertex[0].upper())
            game.play(1+ply%2,action)
        _,_,terminal,score,stones=game.state()
        self.assertTrue(terminal);self.assertEqual(score,fixture['raw_white_minus_black'])
        board='\n'.join(''.join('.XO'[int(x)] for x in row) for row in stones.reshape(9,9))
        self.assertEqual(board,fixture['board'])

    def test_pass_alive_profile_matches_actual_katago_fixture_without_changing_stones(self):
        import json
        fixture=json.loads((Path(__file__).resolve().parents[1]/'eval/fixtures/pass_alive_area.json').read_text())
        module=load_library(self.library,self.sha)
        game=module.Game(json.dumps({'size':9,'komi':7.5,'scoring':'pass_alive_area','history':2,'simulations':0,'cpuct':1.5,'max_search_edges':1000}))
        columns='ABCDEFGHJKLMNOPQRSTUVWXYZ'
        for ply,vertex in enumerate(fixture['action_vertices']):
            a=81 if vertex.lower()=='pass' else (9-int(vertex[1:]))*9+columns.index(vertex[0].upper())
            game.play(1+ply%2,a)
        _,_,terminal,score,stones=game.state()
        self.assertTrue(terminal);self.assertEqual(score,fixture['katago_adjudicated_white_minus_black'])
        np.testing.assert_array_equal(game.ownership(),np.full(81,2,np.uint8))
        self.assertEqual('\n'.join(''.join('.XO'[int(x)] for x in row) for row in stones.reshape(9,9)),fixture['board'])

    def test_ownership_rows_follow_player_perspective_and_terminal_score(self):
        complete=0
        for profile in ('raw_area','pass_alive_area'):
            settings=config(3);settings['scoring']=profile;settings['max_game_moves']=128
            with Actors(settings,self.library,self.sha) as actors:
                for network in range(100):
                    rows=self.finish_search(actors,network)
                    for game in rows.games:
                        if game['truncated']:
                            self.assertEqual(game['ownership'],[]);continue
                        complete+=1
                        self.assertEqual(sum(game['ownership'])+settings['komi'],game['white_score'])
                        take=rows.metadata[:,0]==game['game_id']
                        own=rows.ownership[take].reshape(-1,9)
                        signs=1-2*rows.features[take,0,0,-4]
                        np.testing.assert_array_equal(own,signs[:,None]*np.array(game['ownership']))
                        np.testing.assert_array_equal(rows.outcomes[take],signs*np.sign(game['white_score']))
        self.assertGreater(complete,2)


if __name__=='__main__':
    unittest.main()
