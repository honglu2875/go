use super::*;

fn identity() -> Identity {
    Identity {
        search: 17,
        network: 23,
    }
}

fn gumbel_config() -> GumbelConfig {
    GumbelConfig {
        max_considered_actions: 16,
        value_scale: 0.1,
        maxvisit_init: 50.0,
        rescale_values: true,
    }
}

#[test]
fn gumbel_policy_target_differs_from_visits_and_preserves_adversarial_backup() {
    let search = solve(
        Search::with_gumbel(
            Toy { path: vec![] },
            Config {
                simulations: 128,
                ..Config::default()
            },
            identity(),
            gumbel_config(),
            vec![0.0; 2],
        )
        .unwrap(),
        |_| (vec![0.0; 2], 0.0),
    );
    let stats = search.stats().unwrap();
    assert_eq!(stats.best_action(), Some(1));
    assert!(stats.action_values[0].unwrap() < -0.7);
    assert!(stats.action_values[1].unwrap() > 0.24);
    assert!(stats.policy()[1] > 0.98);
    assert_eq!(stats.visits.iter().sum::<u32>(), 128);
    assert_ne!(stats.policy()[1], stats.visits[1] as f32 / 128.0);
    assert_eq!(search.into_position().unwrap(), Toy { path: vec![] });
}

#[test]
fn gumbel_rejects_invalid_draws_and_zero_simulations() {
    for (budget, draws) in [
        (0, vec![0.0; 2]),
        (3, vec![0.0]),
        (3, vec![f32::INFINITY, 0.0]),
    ] {
        assert!(
            Search::with_gumbel(
                Toy { path: vec![] },
                Config {
                    simulations: budget,
                    ..Config::default()
                },
                identity(),
                gumbel_config(),
                draws
            )
            .is_err()
        );
    }
}

#[test]
fn gumbel_sampling_is_separate_from_the_improved_policy() {
    let mut policies = Vec::new();
    for (draws, expected) in [(vec![10., 0.], 0), (vec![0., 10.], 1)] {
        let search = solve(
            Search::with_gumbel(
                Toy { path: vec![] },
                Config {
                    simulations: 2,
                    ..Config::default()
                },
                identity(),
                gumbel_config(),
                draws,
            )
            .unwrap(),
            |_| (vec![0.0; 2], 0.0),
        );
        let stats = search.stats().unwrap();
        assert_eq!(stats.visits, vec![1, 1]);
        assert_eq!(stats.best_action(), Some(expected));
        policies.push(stats.policy());
    }
    assert_eq!(policies[0], policies[1]);
    assert_eq!(policies[0], vec![0.5; 2]);
}
fn request<P: Position>(search: &mut Search<P>) -> Request {
    let Progress::NeedsEvaluation(request) = search.advance().unwrap() else {
        panic!("expected request")
    };
    request
}

#[derive(Clone, Debug, PartialEq)]
struct Toy {
    path: Vec<usize>,
}
impl Position for Toy {
    fn action_count(&self) -> usize {
        2
    }
    fn legal_actions(&self) -> Vec<usize> {
        if self.path.len() == 2 {
            vec![]
        } else {
            vec![0, 1]
        }
    }
    fn apply(&mut self, action: usize) -> bool {
        if action > 1 || self.path.len() >= 2 {
            return false;
        }
        self.path.push(action);
        true
    }
    fn undo(&mut self) -> bool {
        self.path.pop().is_some()
    }
    fn terminal_value(&self) -> Option<f32> {
        // At even depth the original player moves. Move 0 is a trap: the opponent
        // can choose a loss for us; move 1 guarantees at least a quarter-point win.
        match self.path.as_slice() {
            [0, 0] => Some(1.0),
            [0, 1] => Some(-1.0),
            [1, 0] => Some(0.25),
            [1, 1] => Some(0.5),
            _ => None,
        }
    }
}

fn solve<P: Position>(
    mut search: Search<P>,
    mut evaluator: impl FnMut(&P) -> (Vec<f32>, f32),
) -> Search<P> {
    loop {
        match search.advance().unwrap() {
            Progress::Complete => break,
            Progress::NeedsEvaluation(ticket) => {
                let (logits, value) = evaluator(search.pending_position().unwrap());
                search.complete(ticket, &logits, value).unwrap();
                search.validate().unwrap();
            }
        }
    }
    search.validate().unwrap();
    search
}

#[test]
fn leaf_values_change_sign_and_root_inference_is_not_a_simulation() {
    let mut search = Search::new(
        Toy { path: vec![] },
        Config {
            simulations: 1,
            ..Config::default()
        },
        identity(),
        None,
    )
    .unwrap();
    let root = request(&mut search);
    assert!(root.is_root);
    assert_eq!(search.completed_simulations(), 0);
    search.complete(root, &[100.0, -100.0], 0.6).unwrap();
    let leaf = request(&mut search);
    assert!(!leaf.is_root);
    assert_eq!(search.pending_position().unwrap().path, vec![0]);
    assert_eq!(search.completed_simulations(), 0);
    search.complete(leaf, &[0.0, 0.0], 0.75).unwrap();
    assert_eq!(search.advance().unwrap(), Progress::Complete);
    let stats = search.stats().unwrap();
    assert_eq!(stats.visits, vec![1, 0]);
    assert_eq!(stats.value, -0.75);
    assert_eq!(stats.action_values, vec![Some(-0.75), None]);
    assert_eq!(stats.neural_evaluations, 2);
    assert_eq!(search.into_position().unwrap(), Toy { path: vec![] });
}

#[test]
fn adversary_chooses_its_own_best_response() {
    let search = solve(
        Search::new(
            Toy { path: vec![] },
            Config {
                simulations: 2048,
                ..Config::default()
            },
            identity(),
            None,
        )
        .unwrap(),
        |_| (vec![0.0; 2], 0.0),
    );
    let stats = search.stats().unwrap();
    assert_eq!(stats.best_action(), Some(1));
    assert!(stats.policy()[1] > 0.95);
    assert!(stats.action_values[0].unwrap() < -0.8);
    assert!((stats.action_values[1].unwrap() - 0.25).abs() < 0.05);
    assert_eq!(stats.completed_simulations, 2048);
    assert_eq!(stats.visits.iter().sum::<u32>(), 2048);
    assert_eq!(stats.neural_evaluations, 3);
    assert_eq!(stats.terminal_evaluations, 2046);
}

#[test]
fn pending_duplicate_stale_and_malformed_responses_never_count_as_visits() {
    let mut search =
        Search::new(Toy { path: vec![] }, Config::default(), identity(), None).unwrap();
    let ticket = request(&mut search);
    assert_eq!(request(&mut search), ticket);
    for wrong in [
        Request {
            sequence: 999,
            ..ticket
        },
        Request {
            identity: Identity {
                search: 99,
                ..identity()
            },
            ..ticket
        },
        Request {
            identity: Identity {
                network: 24,
                ..identity()
            },
            ..ticket
        },
    ] {
        assert_eq!(
            search.complete(wrong, &[0.0, 0.0], 0.0),
            Err(Error::WrongRequest)
        );
    }
    assert_eq!(
        search.complete(ticket, &[0.0], 0.0),
        Err(Error::InvalidPolicy)
    );
    assert_eq!(
        search.complete(ticket, &[f32::NAN, 0.0], 0.0),
        Err(Error::InvalidPolicy)
    );
    assert_eq!(
        search.complete(ticket, &[0.0, 0.0], f32::INFINITY),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        search.complete(ticket, &[0.0, 0.0], 1.01),
        Err(Error::InvalidValue)
    );
    assert_eq!(request(&mut search), ticket);
    assert_eq!(search.completed_simulations(), 0);
    search.complete(ticket, &[0.0, 0.0], 0.0).unwrap();
    assert_eq!(
        search.complete(ticket, &[0.0, 0.0], 0.0),
        Err(Error::NoRequest)
    );
    let next = request(&mut search);
    assert_eq!(next.sequence, ticket.sequence + 1);
    assert_eq!(
        search.complete(ticket, &[0.0, 0.0], 0.0),
        Err(Error::WrongRequest)
    );
    assert_eq!(search.abort().path, Vec::<usize>::new());
}

#[test]
fn masked_softmax_and_root_noise_are_normalized_once() {
    let mut board = Board::new(3, 0.5).unwrap();
    board.play(Move::Play(0)).unwrap();
    let mut noise = vec![0.0; 10];
    noise[0] = 1000.0;
    noise[1] = 2.0;
    noise[9] = 2.0;
    let mut logits = vec![1000.0; 10];
    logits[0] = f32::MAX;
    let mut search = Search::new(
        board,
        Config {
            simulations: 0,
            ..Config::default()
        },
        identity(),
        Some(RootNoise {
            fraction: 0.5,
            weights: noise,
        }),
    )
    .unwrap();
    let ticket = request(&mut search);
    search.complete(ticket, &logits, 0.2).unwrap();
    assert_eq!(search.advance().unwrap(), Progress::Complete);
    let stats = search.stats().unwrap();
    assert_eq!(stats.priors[0], 0.0);
    assert!((stats.priors.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!((stats.priors[1] - (0.5 / 9.0 + 0.25)).abs() < 1e-6);
    assert!((stats.priors[2] - 0.5 / 9.0).abs() < 1e-6);
    assert_eq!(stats.neural_evaluations, 1);
    assert_eq!(stats.value, 0.2);
    assert_eq!(stats.policy(), stats.priors);
}

#[test]
fn go_pass_terminal_values_and_exact_root_restoration() {
    let board = Board::new(1, 0.5).unwrap();
    let search = solve(
        Search::new(
            board,
            Config {
                simulations: 16,
                ..Config::default()
            },
            identity(),
            None,
        )
        .unwrap(),
        |p| {
            (
                vec![0.0; 2],
                if p.to_play() == Color::White {
                    1.0
                } else {
                    -1.0
                },
            )
        },
    );
    let stats = search.stats().unwrap();
    assert_eq!(stats.visits, vec![0, 16]);
    assert_eq!(stats.value, -1.0);
    assert_eq!(stats.neural_evaluations, 2);
    assert_eq!(stats.terminal_evaluations, 15);
    let mut restored = search.into_position().unwrap();
    assert_eq!(restored.move_number(), 0);
    assert_eq!(restored.history_len(), 1);
    restored.validate().unwrap();
    restored.play(Move::Pass).unwrap();
    restored.play(Move::Pass).unwrap();
    let mut terminal = Search::new(restored, Config::default(), identity(), None).unwrap();
    assert_eq!(terminal.advance().unwrap(), Progress::Complete);
    let stats = terminal.stats().unwrap();
    assert_eq!(stats.neural_evaluations, 0);
    assert_eq!(stats.value, -1.0);
    assert_eq!(stats.best_action(), None);
}

#[test]
fn search_respects_full_game_history_and_does_not_mutate_the_root() {
    let mut board = Board::new(5, 7.5).unwrap();
    for action in [0, 1, 5, 6, 10, 11, 24, 23] {
        board.play(Move::Play(action)).unwrap();
    }
    let cells = board.stones();
    let hash = board.position_hash();
    let history = board.history_len();
    let depth = board.undo_depth();
    let search = solve(
        Search::new(
            board,
            Config {
                simulations: 256,
                ..Config::default()
            },
            identity(),
            None,
        )
        .unwrap(),
        |p| (vec![0.0; p.action_count()], 0.0),
    );
    let stats = search.stats().unwrap();
    assert_eq!(stats.visits.iter().sum::<u32>(), 256);
    let mut restored = search.into_position().unwrap();
    assert_eq!(restored.stones(), cells);
    assert_eq!(restored.position_hash(), hash);
    assert_eq!(restored.history_len(), history);
    assert_eq!(restored.undo_depth(), depth);
    restored.validate().unwrap();
}

#[test]
fn resource_failure_is_explicit_and_restores_the_position() {
    let mut search = Search::new(
        Toy { path: vec![] },
        Config {
            max_edges: 2,
            ..Config::default()
        },
        identity(),
        None,
    )
    .unwrap();
    let ticket = request(&mut search);
    search.complete(ticket, &[0.0, 0.0], 0.0).unwrap();
    assert_eq!(search.advance(), Err(Error::Capacity));
    assert_eq!(search.advance(), Err(Error::Capacity));
    assert_eq!(search.stats(), Err(Error::Capacity));
    assert!(search.abort().path.is_empty());
}

#[test]
fn fixed_inputs_have_identical_search_results() {
    let config = Config {
        simulations: 99,
        ..Config::default()
    };
    let one = solve(
        Search::new(Toy { path: vec![] }, config, identity(), None).unwrap(),
        |_| (vec![0.25, -0.1], 0.2),
    );
    let two = solve(
        Search::new(Toy { path: vec![] }, config, identity(), None).unwrap(),
        |_| (vec![0.25, -0.1], 0.2),
    );
    assert_eq!(one.stats().unwrap(), two.stats().unwrap());
}
