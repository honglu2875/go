use crate::position::HistoryBoard;
use crate::{Config, GameRecord, Row};
use go_core::Color;
use go_search::{Identity, Position, Progress, Request, RootNoise, RootStats, Search};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use serde::{Deserialize, Serialize};

/// Only emitted between real moves, with no pending evaluations or search tree.
/// The pinned RNG algorithm restarts at exactly this word position.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActorState {
    pub id: u64,
    pub episode: u32,
    pub search_generation: u32,
    pub random_word_position: String,
    pub network: u64,
    pub rows: Vec<Row>,
}

pub(crate) struct Actor {
    pub position: HistoryBoard,
    config: Config,
    id: u64,
    episode: u32,
    search_generation: u32,
    random: ChaCha8Rng,
    search: Option<Search<HistoryBoard>>,
    pending: Option<Request>,
    ready: Option<RootStats>,
    root_features: Vec<f32>,
    rows: Vec<Row>,
    moves: Vec<usize>,
    versions: Vec<u64>,
    episode_nn: u64,
    episode_simulations: u64,
    network: u64,
    prefetch_actions: Vec<Option<usize>>,
    prefetched: Vec<(usize, Vec<f32>, f32)>,
    pub prefetch_evaluations: u64,
    pub prefetch_hits: u64,
}

impl Actor {
    pub fn checkpoint(&self) -> Result<ActorState, String> {
        if self.search.is_some() || self.pending.is_some() || self.ready.is_some() {
            return Err("actor checkpoint requires a committed move boundary".into());
        }
        Ok(ActorState {
            id: self.id,
            episode: self.episode,
            search_generation: self.search_generation,
            random_word_position: self.random.get_word_pos().to_string(),
            network: self.network,
            rows: self.rows.clone(),
        })
    }
    pub fn restore(id: u64, config: Config, state: ActorState) -> Result<Self, String> {
        if state.id != id
            || state.rows.len() >= config.max_game_moves
            || state.search_generation < state.rows.len() as u32
        {
            return Err("invalid actor checkpoint identity or move count".into());
        }
        let word_position = state
            .random_word_position
            .parse::<u128>()
            .map_err(|_| "invalid RNG word position")?;
        if word_position >= 1u128 << 68 {
            return Err("RNG word position exceeds the ChaCha stream".into());
        }
        let mut actor = Self::new(id, config)?;
        actor.episode = state.episode;
        actor.search_generation = state.search_generation;
        actor.network = state.network;
        actor.random.set_word_pos(word_position);
        for mut row in state.rows {
            let legal = actor.position.legal_actions();
            if row.policy.len() != actor.config.actions()
                || row.black_to_play != (actor.position.board.to_play() == Color::Black)
                || !row.root_value.is_finite()
                || !(-1.0..=1.0).contains(&row.root_value)
                || row.policy.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (row.policy.iter().map(|v| f64::from(*v)).sum::<f64>() - 1.0).abs() > 1e-5
                || row
                    .policy
                    .iter()
                    .enumerate()
                    .any(|(a, &p)| p > 0.0 && legal.binary_search(&a).is_err())
                || row.simulations != actor.config.simulations
                || row.neural_evaluations == 0
                || row.neural_evaluations > row.simulations + 1
                || row.terminal_evaluations > row.simulations
            {
                return Err("invalid unfinished-game training row".into());
            }
            row.features = actor.position.features(&legal);
            if !actor.position.commit(row.action) || actor.position.board.is_terminal() {
                return Err(
                    "checkpoint contains illegal moves or an unflushed terminal game".into(),
                );
            }
            actor.episode_nn += u64::from(row.neural_evaluations);
            actor.episode_simulations += u64::from(row.simulations);
            actor.moves.push(row.action);
            actor.versions.push(row.network);
            actor.rows.push(row);
        }
        actor.position.board.validate().map_err(|e| e.to_string())?;
        Ok(actor)
    }
    pub fn new(id: u64, config: Config) -> Result<Self, String> {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&config.seed.to_le_bytes());
        seed[8..16].copy_from_slice(&id.to_le_bytes());
        seed[16..24].copy_from_slice(&0x676f7a65726fu64.to_le_bytes());
        Ok(Self {
            position: HistoryBoard::with_utility(
                config.size,
                config.komi,
                config.history,
                config.scoring,
                config.score_utility,
            )
            .map_err(|e| e.to_string())?,
            config,
            id,
            episode: 0,
            search_generation: 0,
            random: ChaCha8Rng::from_seed(seed),
            search: None,
            pending: None,
            ready: None,
            root_features: Vec::new(),
            rows: Vec::new(),
            moves: Vec::new(),
            versions: Vec::new(),
            episode_nn: 0,
            episode_simulations: 0,
            network: 0,
            prefetch_actions: Vec::new(),
            prefetched: Vec::new(),
            prefetch_evaluations: 0,
            prefetch_hits: 0,
        })
    }
    pub fn start(&mut self, network: u64) -> Result<(), String> {
        if self.search.is_some() || self.ready.is_some() {
            return Err("search already active".into());
        }
        self.search_generation = self
            .search_generation
            .checked_add(1)
            .ok_or("search identity exhausted")?;
        let legal = self.position.legal_actions();
        self.root_features = self.position.features(&legal);
        let noise = if self.config.dirichlet_fraction > 0.0 {
            let gamma = Gamma::new(self.config.dirichlet_alpha, 1.0).map_err(|e| e.to_string())?;
            let mut weights = vec![0.0; self.position.action_count()];
            // Normalize in f64 before converting, including small-alpha draws.
            let draws: Vec<f64> = legal
                .iter()
                .map(|_| gamma.sample(&mut self.random))
                .collect();
            let sum: f64 = draws.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err("invalid Dirichlet draw".into());
            }
            for (&a, v) in legal.iter().zip(draws) {
                weights[a] = (v / sum) as f32;
            }
            Some(RootNoise {
                fraction: self.config.dirichlet_fraction,
                weights,
            })
        } else {
            None
        };
        self.network = network;
        let search_config =
            go_search::Config {
                simulations: self.config.simulations,
                cpuct: self.config.cpuct / self.config.score_utility.map_or(1.0, |u| u.divisor()),
                fpu: self.config.fpu_reduction.map_or(
                    go_search::FirstPlayUrgency::Zero,
                    |reduction| go_search::FirstPlayUrgency::NetworkValue {
                        reduction: reduction
                            / self.config.score_utility.map_or(1.0, |u| u.divisor()),
                    },
                ),
                max_edges: self.config.max_search_edges,
            };
        let identity = Identity {
            search: (self.id << 32) | u64::from(self.search_generation),
            network,
        };
        let search = if let Some(settings) = self.config.gumbel {
            let draws = (0..self.config.actions())
                .map(|_| {
                    if settings.gumbel_scale == 0.0 {
                        return 0.0;
                    }
                    let uniform: f64 = self.random.sample(rand::distr::Open01);
                    (-(-uniform.ln()).ln() * f64::from(settings.gumbel_scale)) as f32
                })
                .collect();
            Search::with_gumbel(
                self.position.clone(),
                search_config,
                identity,
                settings.search(),
                draws,
            )
        } else {
            Search::new(self.position.clone(), search_config, identity, noise)
        };
        self.search = Some(search.map_err(|e| e.to_string())?);
        self.advance()
    }
    fn advance(&mut self) -> Result<(), String> {
        let search = self.search.as_mut().ok_or("search absent")?;
        loop {
            match search.advance().map_err(|e| e.to_string())? {
                Progress::NeedsEvaluation(request) => {
                    if let Some(action) = search.pending_root_child_action()
                        && let Some(index) = self.prefetched.iter().position(|row| row.0 == action)
                    {
                        let (_, policy, value) = self.prefetched.swap_remove(index);
                        search
                            .complete(request, &policy, value)
                            .map_err(|e| e.to_string())?;
                        self.prefetch_hits += 1;
                        continue;
                    }
                    self.pending = Some(request);
                }
                Progress::Complete => {
                    self.pending = None;
                    self.ready = Some(search.stats().map_err(|e| e.to_string())?);
                }
            }
            break;
        }
        Ok(())
    }

    pub fn prefetch_features(&mut self, limit: usize) -> Result<(Vec<f32>, Vec<u8>), String> {
        if !self.prefetch_actions.is_empty() || !self.prefetched.is_empty() {
            return Err("root predictions already pending or cached".into());
        }
        let search = self.search.as_ref().ok_or("search absent")?;
        let actions = search.root_sweep_candidates(limit);
        let mut position = self.position.clone();
        let width = self.config.feature_width();
        let mut features = Vec::with_capacity(limit * width);
        let mut active = Vec::with_capacity(limit);
        for action in actions {
            if !position.apply(action) {
                return Err("root prefetch proposed an illegal move".into());
            }
            // Terminal values remain authoritative native evaluations.
            if position.terminal_value().is_none() {
                features.extend(position.features(&position.legal_actions()));
                active.push(1);
                self.prefetch_actions.push(Some(action));
            }
            if !position.undo() {
                return Err("root prefetch could not undo move".into());
            }
        }
        self.prefetch_actions.resize(limit, None);
        features.resize(limit * width, 0.0);
        active.resize(limit, 0);
        Ok((features, active))
    }

    pub fn evaluate_prefetch(&mut self, policies: &[f32], values: &[f32]) -> Result<(), String> {
        let actions = self.config.actions();
        if values.len() != self.prefetch_actions.len() || policies.len() != values.len() * actions {
            return Err("root prediction shape differs".into());
        }
        for (i, action) in self.prefetch_actions.drain(..).enumerate() {
            if let Some(action) = action {
                self.prefetched.push((
                    action,
                    policies[i * actions..(i + 1) * actions].to_vec(),
                    values[i],
                ));
                self.prefetch_evaluations += 1;
            }
        }
        self.advance()
    }
    pub fn evaluate(&mut self, policy: &[f32], value: f32) -> Result<(), String> {
        if let Some(request) = self.pending {
            self.search
                .as_mut()
                .unwrap()
                .complete(request, policy, value)
                .map_err(|e| e.to_string())?;
            self.advance()?;
        }
        Ok(())
    }
    pub fn features(&self) -> Option<Vec<f32>> {
        let request = self.pending?;
        if request.is_root {
            return Some(self.root_features.clone());
        }
        let search = self.search.as_ref().unwrap();
        Some(
            search
                .pending_position()
                .unwrap()
                .features(search.pending_legal_actions().unwrap()),
        )
    }
    pub fn commit(&mut self) -> Result<Option<GameRecord>, String> {
        let stats = self.ready.take().ok_or("search incomplete")?;
        self.prefetched.clear();
        let policy = stats.policy();
        let temperature = if self.position.board.move_number() < self.config.temperature_moves {
            self.config.temperature_early
        } else {
            self.config.temperature_late
        };
        let action = if self.config.gumbel.is_some() || temperature == 0.0 {
            stats.best_action().ok_or("no root action")?
        } else {
            // Log-space normalization avoids overflow/underflow at low temperatures.
            let log_weights: Vec<f64> = policy
                .iter()
                .map(|&p| {
                    if p > 0.0 {
                        f64::from(p).ln() / f64::from(temperature)
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect();
            let max = log_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = log_weights.iter().map(|v| (v - max).exp()).collect();
            let mut sample = self.random.random::<f64>() * weights.iter().sum::<f64>();
            let mut selected = weights
                .iter()
                .rposition(|p| *p > 0.0)
                .ok_or("no policy mass")?;
            for (i, p) in weights.iter().enumerate() {
                sample -= p;
                if sample < 0.0 {
                    selected = i;
                    break;
                }
            }
            selected
        };
        self.episode_nn += u64::from(stats.neural_evaluations);
        self.episode_simulations += u64::from(stats.completed_simulations);
        self.rows.push(Row {
            features: std::mem::take(&mut self.root_features),
            policy,
            black_to_play: self.position.board.to_play() == Color::Black,
            action,
            network: self.network,
            root_value: stats.value,
            simulations: stats.completed_simulations,
            neural_evaluations: stats.neural_evaluations,
            terminal_evaluations: stats.terminal_evaluations,
        });
        if !self.position.commit(action) {
            return Err("search selected an illegal move".into());
        }
        self.search = None;
        self.moves.push(action);
        self.versions.push(self.network);
        let terminal = self.position.board.is_terminal();
        let truncated = !terminal && self.moves.len() >= self.config.max_game_moves;
        if !terminal && !truncated {
            return Ok(None);
        }
        let ownership: Vec<i8> = if terminal {
            self.position
                .board
                .ownership(self.config.scoring.core())
                .iter()
                .map(|&c| match c {
                    1 => -1,
                    2 => 1,
                    _ => 0,
                })
                .collect()
        } else {
            Vec::new()
        };
        let white_score = terminal
            .then(|| ownership.iter().map(|&c| f64::from(c)).sum::<f64>() + self.config.komi);
        let record = GameRecord {
            game_id: (self.id << 32) | u64::from(self.episode),
            actor_id: self.id,
            size: self.config.size,
            komi: self.config.komi,
            scoring: self.config.scoring,
            actions: std::mem::take(&mut self.moves),
            networks: std::mem::take(&mut self.versions),
            white_score,
            ownership,
            truncated,
            neural_evaluations: self.episode_nn,
            simulations: self.episode_simulations,
            rows: if terminal {
                std::mem::take(&mut self.rows)
            } else {
                self.rows.clear();
                Vec::new()
            },
        };
        self.episode = self
            .episode
            .checked_add(1)
            .ok_or("game identity exhausted")?;
        self.position = HistoryBoard::with_utility(
            self.config.size,
            self.config.komi,
            self.config.history,
            self.config.scoring,
            self.config.score_utility,
        )
        .map_err(|e| e.to_string())?;
        self.episode_nn = 0;
        self.episode_simulations = 0;
        Ok(Some(record))
    }
}
