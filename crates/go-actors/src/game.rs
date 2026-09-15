//! A externally driven game with one frozen-network search per requested move.
//! Opponent moves use the same authoritative rules; every leaf in a search uses
//! the root engine's evaluator, regardless of whose turn the leaf represents.
use crate::position::HistoryBoard;
use go_search::{Identity, Position, Progress, Request, RootStats, Search};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GameConfig {
    pub size: usize,
    pub komi: f64,
    #[serde(default)]
    pub scoring: crate::Scoring,
    pub history: usize,
    pub simulations: u32,
    pub cpuct: f32,
    #[serde(default)]
    pub fpu_reduction: Option<f32>,
    #[serde(default)]
    pub score_utility: Option<crate::ScoreUtility>,
    #[serde(default)]
    pub gumbel: Option<crate::GumbelSettings>,
    pub max_search_edges: usize,
}

pub struct Game {
    pub position: HistoryBoard,
    config: GameConfig,
    search: Option<Search<HistoryBoard>>,
    pending: Option<Request>,
    generation: u64,
    moves: Vec<usize>,
}

pub struct GameBatch {
    pub request: Option<Request>,
    pub features: Vec<f32>,
}

impl Game {
    pub fn new(config: GameConfig) -> Result<Self, String> {
        if config.size == 0
            || config.size > 1024
            || config.simulations > 1_000_000
            || !config.cpuct.is_finite()
            || config.cpuct < 0.0
            || config
                .fpu_reduction
                .is_some_and(|r| !r.is_finite() || !(0.0..=2.0).contains(&r))
            || config.score_utility.is_some_and(|u| !u.is_valid())
            || config.gumbel.is_some_and(|g| {
                !g.is_valid()
                    || g.gumbel_scale != 0.0
                    || config.simulations == 0
                    || config.cpuct != 0.0
                    || config.fpu_reduction.is_some()
            })
            || config.max_search_edges < config.size * config.size + 1
        {
            return Err("invalid game configuration".into());
        }
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
            search: None,
            pending: None,
            generation: 0,
            moves: Vec::new(),
        })
    }
    pub fn play(&mut self, color: u8, action: usize) -> Result<(), String> {
        if self.search.is_some() || color != self.position.board.to_play() as u8 {
            return Err("move during a search or wrong player".into());
        }
        if !self.position.commit(action) {
            return Err("illegal move".into());
        }
        self.moves.push(action);
        Ok(())
    }
    /// Full causal tape before the requested leaf's next move. The request
    /// binds the hypothetical suffix to the exact root and frozen network.
    pub fn request_history(&self, request: Request) -> Result<Vec<usize>, String> {
        if self.pending != Some(request) {
            return Err("stale or mismatched history request".into());
        }
        let suffix = self
            .search
            .as_ref()
            .and_then(Search::pending_actions)
            .ok_or("pending search absent")?;
        let mut history = self.moves.clone();
        history.extend(suffix);
        Ok(history)
    }
    pub fn start(&mut self, network: u64) -> Result<GameBatch, String> {
        if self.search.is_some() || self.position.board.is_terminal() {
            return Err("search already active or game over".into());
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or("search identity exhausted")?;
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
            search: self.generation,
            network,
        };
        let search = if let Some(settings) = self.config.gumbel {
            Search::with_gumbel(
                self.position.clone(),
                search_config,
                identity,
                settings.search(),
                vec![0.0; self.position.action_count()],
            )
        } else {
            Search::new(self.position.clone(), search_config, identity, None)
        };
        self.search = Some(search.map_err(|e| e.to_string())?);
        self.advance()
    }
    fn advance(&mut self) -> Result<GameBatch, String> {
        let search = self.search.as_mut().ok_or("search absent")?;
        match search.advance().map_err(|e| e.to_string())? {
            Progress::NeedsEvaluation(request) => {
                self.pending = Some(request);
                Ok(GameBatch {
                    request: Some(request),
                    features: search
                        .pending_position()
                        .unwrap()
                        .features(search.pending_legal_actions().unwrap()),
                })
            }
            Progress::Complete => {
                self.pending = None;
                Ok(GameBatch {
                    request: None,
                    features: Vec::new(),
                })
            }
        }
    }
    pub fn evaluate(
        &mut self,
        request: Request,
        logits: &[f32],
        value: f32,
    ) -> Result<GameBatch, String> {
        if self.pending != Some(request) {
            return Err("stale or mismatched evaluation".into());
        }
        self.search
            .as_mut()
            .ok_or("search absent")?
            .complete(request, logits, value)
            .map_err(|e| e.to_string())?;
        self.advance()
    }
    pub fn finish(&mut self) -> Result<RootStats, String> {
        let stats = self
            .search
            .as_ref()
            .ok_or("search absent")?
            .stats()
            .map_err(|e| e.to_string())?;
        self.search = None;
        self.pending = None;
        Ok(stats)
    }
    pub fn inspect_search(&self) -> Result<go_search::RootInspection, String> {
        self.search
            .as_ref()
            .ok_or("search absent")?
            .inspect_root()
            .map_err(|e| e.to_string())
    }
    pub fn legal(&self) -> Vec<usize> {
        self.position.legal_actions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_score_utility_is_used_in_backups_without_changing_game_score() {
        let mut game = Game::new(GameConfig {
            size: 1,
            komi: 2.0,
            scoring: crate::Scoring::PassAliveArea,
            history: 2,
            simulations: 2,
            cpuct: 1.5,
            fpu_reduction: Some(0.2),
            score_utility: Some(crate::ScoreUtility {
                factor: 1.0,
                scale: 2.0,
            }),
            gumbel: None,
            max_search_edges: 1000,
        })
        .unwrap();
        let mut batch = game.start(1).unwrap();
        while let Some(request) = batch.request {
            batch = game.evaluate(request, &[0.0; 2], 0.0).unwrap();
        }
        let stats = game.finish().unwrap();
        // One neural traversal contributes zero, one solved traversal loses
        // with utility -(1 + 0.5)/2. The mean is -0.375 from Black's view.
        assert_eq!(stats.value, -0.375);
        assert_eq!(stats.terminal_evaluations, 1);
        game.play(1, 1).unwrap();
        game.play(2, 1).unwrap();
        assert_eq!(game.position.score().white_minus_black, 2.0);
        assert_eq!(game.position.terminal_value(), Some(-0.75));
    }
    #[test]
    fn external_moves_preserve_history_and_search_uses_one_network_identity() {
        let mut game = Game::new(GameConfig {
            size: 3,
            komi: 0.5,
            scoring: crate::Scoring::RawArea,
            history: 2,
            simulations: 64,
            cpuct: 1.5,
            fpu_reduction: None,
            score_utility: None,
            gumbel: None,
            max_search_edges: 10000,
        })
        .unwrap();
        assert!(game.play(2, 0).is_err());
        game.play(1, 0).unwrap();
        game.play(2, 1).unwrap();
        let before = game.position.features(&game.legal());
        let mut batch = game.start(27).unwrap();
        assert!(game.play(1, 2).is_err());
        assert!(game.finish().is_err());
        let old = batch.request.unwrap();
        let mut requests = 0;
        let mut deepest_suffix = 0;
        while let Some(request) = batch.request {
            assert_eq!(request.identity.network, 27);
            let history = game.request_history(request).unwrap();
            assert_eq!(&history[..2], &[0, 1]);
            deepest_suffix = deepest_suffix.max(history.len() - 2);
            // Reconstruct the pending leaf independently from an empty board.
            // This checks root tape plus search suffix, including undo behavior
            // between leaves, against the actual feature request.
            let mut replay =
                HistoryBoard::with_scoring(3, 0.5, 2, crate::Scoring::RawArea).unwrap();
            for action in history {
                assert!(replay.commit(action));
            }
            assert_eq!(replay.features(&replay.legal_actions()), batch.features);
            if requests > 0 {
                assert!(game.evaluate(old, &[0.0; 10], 0.0).is_err());
                assert!(game.request_history(old).is_err());
            }
            batch = game.evaluate(request, &[0.0; 10], 0.0).unwrap();
            requests += 1;
        }
        let stats = game.finish().unwrap();
        assert!(game.request_history(old).is_err());
        assert_eq!(stats.completed_simulations, 64);
        assert!(deepest_suffix >= 2);
        assert_eq!(game.position.features(&game.legal()), before);
        assert_eq!(game.position.board.history_len(), 3);
        game.play(1, stats.best_action().unwrap()).unwrap();
        let root_request = game.start(28).unwrap().request.unwrap();
        assert_eq!(game.request_history(root_request).unwrap().len(), 3);
    }
}
