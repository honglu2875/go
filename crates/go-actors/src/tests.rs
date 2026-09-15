use super::*;
use go_search::Position;

/// Deliberately depends on the complete encoded history, not a batch slot.
/// Exactly the same inputs therefore produce identical oracle outputs even
/// when root children are evaluated before the sequential search requests them.
fn prefetch_oracle(batch: &Batch, c: &Config, extreme: bool) -> (Vec<f32>, Vec<f32>) {
    let mut policies = Vec::new();
    let mut values = Vec::new();
    for row in batch.features.chunks_exact(c.feature_width()) {
        let signature: f64 = row
            .iter()
            .enumerate()
            .map(|(i, &x)| f64::from(x) * ((i * 17 + 3) % 29) as f64)
            .sum();
        for action in 0..c.actions() {
            let logit = ((signature * (action + 1) as f64 * 0.017).sin() * 3.0) as f32;
            policies.push(if extreme && action % 2 == 0 {
                -1e10
            } else {
                logit
            });
        }
        values.push((signature * 0.011).sin() as f32);
    }
    (policies, values)
}

#[test]
fn root_prefetch_preserves_sequential_games_targets_and_recovery() {
    for (simulations, limit, extreme) in [
        (1, 1, false),
        (7, 16, false),
        (16, 16, false),
        (31, 1, false),
        (64, 16, true),
    ] {
        let mut c = config();
        c.simulations = simulations;
        c.cpuct = 0.;
        c.dirichlet_fraction = 0.;
        c.temperature_early = 0.;
        c.temperature_moves = 0;
        c.gumbel = Some(GumbelSettings {
            max_considered_actions: 16,
            value_scale: 0.1,
            maxvisit_init: 50.,
            rescale_values: true,
            gumbel_scale: 1.,
        });
        let mut sequential = Pool::new(c.clone()).unwrap();
        let mut cached = Pool::new(c.clone()).unwrap();
        let mut prior_hits = 0;
        for turn in 0..48 {
            for (pool, prefetch) in [(&mut sequential, false), (&mut cached, true)] {
                let mut batch = pool.start(turn / 4).unwrap();
                let mut first = true;
                while batch.active_count() > 0 {
                    let (policy, value) = prefetch_oracle(&batch, &c, extreme);
                    batch = pool
                        .evaluate(batch.round, batch.network, &policy, &value)
                        .unwrap();
                    if first && prefetch && batch.active_count() > 0 {
                        let old_round = batch.round;
                        assert!(pool.prefetch(batch.round, batch.network, 0).is_err());
                        assert!(pool.prefetch(batch.round, batch.network, 65).is_err());
                        let proposal = pool.prefetch(batch.round, batch.network, limit).unwrap();
                        assert_eq!(proposal.active.len(), c.games * limit);
                        assert!(pool.commit().is_err());
                        assert!(
                            pool.prefetch(proposal.round, proposal.network, limit)
                                .is_err()
                        );
                        assert!(
                            pool.evaluate(old_round, proposal.network, &policy, &value)
                                .is_err()
                        );
                        let (mut pp, pv) = prefetch_oracle(&proposal, &c, extreme);
                        let counters = pool.prefetch_counters().unwrap();
                        assert!(
                            pool.evaluate_prefetch(proposal.round, proposal.network + 1, &pp, &pv)
                                .is_err()
                        );
                        let last = pp.len() - 1;
                        let saved = pp[last];
                        pp[last] = f32::NAN;
                        assert!(
                            pool.evaluate_prefetch(proposal.round, proposal.network, &pp, &pv)
                                .is_err()
                        );
                        assert_eq!(pool.prefetch_counters().unwrap(), counters);
                        pp[last] = saved;
                        batch = pool
                            .evaluate_prefetch(proposal.round, proposal.network, &pp, &pv)
                            .unwrap();
                        assert!(
                            pool.evaluate_prefetch(proposal.round, proposal.network, &pp, &pv)
                                .is_err()
                        );
                    }
                    first = false;
                }
            }
            assert_eq!(sequential.commit().unwrap(), cached.commit().unwrap());
            let left = sequential.checkpoint().unwrap();
            let right = cached.checkpoint().unwrap();
            // Transport request numbering differs because prefetch is a new
            // request type. Every actor, RNG, row, network and config must match.
            assert_eq!(left.actors, right.actors);
            assert_eq!(left.network, right.network);
            assert_eq!(left.config, right.config);
            let (evaluated, used) = cached.prefetch_counters().unwrap();
            assert!(used <= evaluated && used >= prior_hits);
            prior_hits = used;
            if turn == 23 {
                cached.close();
                cached = Pool::restore(c.clone(), right).unwrap();
                assert_eq!(cached.prefetch_counters().unwrap(), (0, 0));
                prior_hits = 0;
            }
        }
        assert!(cached.prefetch_counters().unwrap().1 > 0);
    }
}

#[test]
fn root_prefetch_rejects_puct_and_invalid_limits_without_poisoning_pool() {
    let mut pool = Pool::new(config()).unwrap();
    let batch = pool.start(0).unwrap();
    assert!(pool.prefetch(batch.round, batch.network, 16).is_err());
    let (p, v) = prefetch_oracle(&batch, pool.config(), false);
    assert!(pool.evaluate(batch.round, batch.network, &p, &v).is_ok());
}

fn config() -> Config {
    Config {
        size: 3,
        komi: 0.5,
        scoring: Scoring::RawArea,
        history: 2,
        games: 4,
        workers: 2,
        worker_cpus: Vec::new(),
        simulations: 8,
        cpuct: 1.5,
        fpu_reduction: None,
        score_utility: None,
        gumbel: None,
        max_search_edges: 10000,
        max_game_moves: 30,
        dirichlet_alpha: 0.3,
        dirichlet_fraction: 0.25,
        temperature_early: 1.0,
        temperature_late: 0.0,
        temperature_moves: 10,
        seed: 27,
        actor_offset: 0,
    }
}
fn search(pool: &mut Pool, network: u64) {
    let mut batch = pool.start(network).unwrap();
    let games = pool.config().games;
    let actions = pool.config().actions();
    let simulations = pool.config().simulations;
    let mut rounds = 0;
    while batch.active_count() > 0 {
        batch = pool
            .evaluate(
                batch.round,
                network,
                &vec![0.0; games * actions],
                &vec![0.0; games],
            )
            .unwrap();
        rounds += 1;
        assert!(rounds <= simulations + 1);
    }
}

#[test]
fn history_features_and_undo_keep_current_player_perspective() {
    let mut p = position::HistoryBoard::new(3, 0.5, 2).unwrap();
    let before = p.features(&p.legal_actions());
    assert_eq!(before.len(), 9 * 8);
    assert_eq!(before[4], 1.0);
    assert!(before[5] < 0.0);
    assert!(p.apply(0));
    let after = p.features(&p.legal_actions());
    assert_eq!(after[0], 0.0);
    assert_eq!(after[1], 1.0); // Previous black stone is now opponent.
    assert_eq!(after[4], 0.0);
    assert!(after[5] > 0.0);
    assert_eq!(after[7], 0.0);
    assert!(p.apply(1));
    assert!(p.apply(9));
    assert!(p.undo());
    assert!(p.undo());
    assert!(p.undo());
    assert_eq!(p.features(&p.legal_actions()), before);
    assert!(!p.undo());
    p.board.validate().unwrap();
    assert!(p.commit(0));
    assert!(!p.undo());
    assert_eq!(p.board.history_len(), 2);
}

#[test]
fn terminal_search_value_uses_the_configured_scoring_profile() {
    let text = ".O.OO.O.O\nO.OO.O.OO\nOOO.OOOO.\nO.OO.OO.O\n.O.OO.OOO\nOOOOOO.OO\n.OOO.OO.O\nOO.OO.OOO\nO.OX.O.O.";
    let stones: Vec<_> = text
        .bytes()
        .filter(|&c| c != b'\n')
        .enumerate()
        .filter_map(|(p, c)| match c {
            b'X' => Some((p, go_core::Color::Black)),
            b'O' => Some((p, go_core::Color::White)),
            _ => None,
        })
        .collect();
    for (profile, black_value) in [(Scoring::RawArea, 1.0), (Scoring::PassAliveArea, -1.0)] {
        let mut p = position::HistoryBoard::with_scoring(9, -79.5, 2, profile).unwrap();
        p.board = go_core::Board::from_setup(9, -79.5, &stones, go_core::Color::Black).unwrap();
        assert_eq!(p.terminal_value(), None);
        assert!(p.apply(81));
        assert!(p.apply(81));
        assert_eq!(p.terminal_value(), Some(black_value));
        assert!(p.undo());
        assert_eq!(p.terminal_value(), None);
    }
}

#[test]
fn only_complete_games_expose_terminal_training_rows() {
    let mut c = config();
    c.size = 1;
    c.games = 2;
    c.workers = 2;
    c.max_game_moves = 4;
    let mut pool = Pool::new(c).unwrap();
    search(&mut pool, 1);
    assert!(pool.commit().unwrap().is_empty());
    search(&mut pool, 2);
    let games = pool.commit().unwrap();
    assert_eq!(games.len(), 2);
    for game in games {
        assert_eq!(game.actions, vec![1, 1]);
        assert_eq!(game.networks, vec![1, 2]);
        assert_eq!(game.white_score, Some(0.5));
        assert_eq!(game.ownership, vec![0]);
        assert!(!game.truncated);
        assert_eq!(game.rows.len(), 2);
        assert!(game.rows[0].black_to_play);
        assert!(!game.rows[1].black_to_play);
        for row in &game.rows {
            assert_eq!(row.policy, vec![0.0, 1.0]);
            assert_eq!(row.simulations, 8);
        }
        assert_eq!(game.simulations, 16);
        assert_eq!(game.neural_evaluations, 3);
    }
    pool.close();
    pool.close();
    assert!(pool.start(3).is_err());
}

#[test]
fn truncations_are_recorded_without_invented_outcomes_or_rows() {
    let mut c = config();
    c.max_game_moves = 1;
    let mut pool = Pool::new(c).unwrap();
    search(&mut pool, 0);
    let games = pool.commit().unwrap();
    assert_eq!(games.len(), 4);
    for game in games {
        assert!(game.truncated);
        assert_eq!(game.white_score, None);
        assert!(game.ownership.is_empty());
        assert!(game.rows.is_empty());
        assert_eq!(game.actions.len(), 1);
    }
}

#[test]
fn worker_count_does_not_change_seeded_games_or_targets() {
    let mut a = config();
    a.workers = 1;
    let mut b = a.clone();
    b.workers = 2;
    let mut one = Pool::new(a).unwrap();
    let mut two = Pool::new(b).unwrap();
    let mut completed = 0;
    for network in 0..45 {
        search(&mut one, network);
        search(&mut two, network);
        let left = one.commit().unwrap();
        let right = two.commit().unwrap();
        assert_eq!(left.len(), right.len());
        completed += left.len();
        for (a, b) in left.iter().zip(&right) {
            assert_eq!(a.game_id, b.game_id);
            assert_eq!(a.actions, b.actions);
            assert_eq!(a.networks, b.networks);
            assert_eq!(a.white_score, b.white_score);
            assert_eq!(a.rows, b.rows);
            assert_eq!(a.truncated, b.truncated);
        }
    }
    assert!(completed >= 4);
}

#[test]
fn stale_or_bad_neural_batches_are_rejected_before_workers_change_state() {
    let mut pool = Pool::new(config()).unwrap();
    let batch = pool.start(2).unwrap();
    assert!(pool.commit().is_err());
    assert!(pool.start(2).is_err());
    let policies = vec![0.0; 40];
    let values = vec![0.0; 4];
    assert!(
        pool.evaluate(batch.round + 1, 2, &policies, &values)
            .is_err()
    );
    assert!(pool.evaluate(batch.round, 3, &policies, &values).is_err());
    assert!(
        pool.evaluate(batch.round, 2, &policies[..39], &values)
            .is_err()
    );
    assert!(
        pool.evaluate(batch.round, 2, &policies, &[f32::NAN; 4])
            .is_err()
    );
    let next = pool.evaluate(batch.round, 2, &policies, &values).unwrap();
    assert_eq!(next.round, batch.round + 1);
    assert!(pool.evaluate(batch.round, 2, &policies, &values).is_err());
}

#[test]
fn invalid_configuration_is_rejected_without_launching_workers() {
    for utility in [
        ScoreUtility {
            factor: -0.1,
            scale: 2.0,
        },
        ScoreUtility {
            factor: 0.3,
            scale: 0.0,
        },
        ScoreUtility {
            factor: f32::NAN,
            scale: 2.0,
        },
    ] {
        let mut c = config();
        c.score_utility = Some(utility);
        assert!(Pool::new(c).is_err());
    }
    for reduction in [f32::NAN, -0.01, 2.01] {
        let mut c = config();
        c.fpu_reduction = Some(reduction);
        assert!(Pool::new(c).is_err());
    }
    let mut c = config();
    c.workers = 0;
    assert!(Pool::new(c).is_err());
    let mut c = config();
    c.worker_cpus = vec![0, 0];
    assert!(Pool::new(c).is_err());
    let mut c = config();
    c.actor_offset = u64::MAX;
    assert!(Pool::new(c).is_err());
    let mut c = config();
    c.size = usize::MAX;
    assert!(Pool::new(c).is_err());
}

#[test]
fn checkpoint_restores_unfinished_games_rng_and_all_search_requests() {
    let mut c = config();
    c.score_utility = Some(ScoreUtility {
        factor: 0.3,
        scale: 2.0,
    });
    check_restore(c);
}

#[test]
fn gumbel_checkpoint_restores_draws_targets_and_worker_independence() {
    let mut c = config();
    c.cpuct = 0.0;
    c.fpu_reduction = None;
    c.dirichlet_fraction = 0.0;
    c.temperature_early = 0.0;
    c.temperature_late = 0.0;
    c.temperature_moves = 0;
    c.gumbel = Some(GumbelSettings {
        max_considered_actions: 16,
        value_scale: 0.1,
        maxvisit_init: 50.0,
        rescale_values: true,
        gumbel_scale: 1.0,
    });
    check_restore(c);
}

fn check_restore(mut c: Config) {
    c.workers = 1;
    let mut original = Pool::new(c.clone()).unwrap();
    for step in 0..11 {
        search(&mut original, step);
        original.commit().unwrap();
    }
    let state = original.checkpoint().unwrap();
    assert!(state.actors.iter().any(|a| !a.rows.is_empty()));
    c.workers = 2;
    let mut restored = Pool::restore(c, state).unwrap();
    let mut completed = 0;
    for step in 11..61 {
        let mut a = original.start(step).unwrap();
        let mut b = restored.start(step).unwrap();
        loop {
            assert_eq!(a.round, b.round);
            assert_eq!(a.active, b.active);
            assert_eq!(a.features, b.features);
            if a.active_count() == 0 {
                break;
            }
            let logits: Vec<f32> = (0..40).map(|i| (i as f32 * 0.17).sin()).collect();
            let values = [0.1, -0.2, 0.3, -0.4];
            a = original.evaluate(a.round, step, &logits, &values).unwrap();
            b = restored.evaluate(b.round, step, &logits, &values).unwrap();
        }
        let a = original.commit().unwrap();
        let b = restored.commit().unwrap();
        assert_eq!(a.len(), b.len());
        completed += a.len();
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.game_id, b.game_id);
            assert_eq!(a.actions, b.actions);
            assert_eq!(a.rows, b.rows);
            assert_eq!(a.white_score, b.white_score);
        }
    }
    assert!(completed > 4);
}

#[test]
fn checkpoint_rejects_partial_searches_corruption_and_changed_science() {
    let mut pool = Pool::new(config()).unwrap();
    let state = pool.checkpoint().unwrap();
    let mut changed = config();
    changed.seed += 1;
    assert!(Pool::restore(changed, state.clone()).is_err());
    let mut bad = state.clone();
    bad.actors[0].id += 1;
    assert!(Pool::restore(config(), bad).is_err());
    let mut bad = state;
    bad.actors[0].random_word_position = (1u128 << 68).to_string();
    assert!(Pool::restore(config(), bad).is_err());
    search(&mut pool, 0);
    assert!(pool.checkpoint().is_err());
    pool.commit().unwrap();
    let mut bad = pool.checkpoint().unwrap();
    bad.actors[0].rows[0].policy[0] = f32::NAN;
    assert!(Pool::restore(config(), bad).is_err());
}
