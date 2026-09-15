//! Policy-only paired causal traces. This does not represent completed MCTS.
//! Only the active player's own logits under an already matched prefix can act.
use crate::{Scoring, position::HistoryBoard};
use go_search::Position;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub size: usize,
    pub komi: f64,
    pub history: usize,
    pub scoring: Scoring,
    pub max_game_moves: usize,
    #[serde(default)]
    pub root_legal_mask: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Ticket {
    pub generation: u64,
    pub move_number: usize,
    pub models: [u64; 2],
    pub contexts: [u64; 2],
    pub episode: u64,
}

/// Logical layouts: actions[2,k,H], own_logits[2,k,H,A], noise[H,A].
/// Views are Black then White. Own-policy Gumbel noise is shared across samples
/// and views at each real ply, so selecting a branch cannot select a new draw.
pub struct Packet<'a> {
    pub samples: usize,
    pub horizon: usize,
    pub actions: &'a [i32],
    pub own_logits: &'a [f32],
    pub noise: &'a [f32],
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Horizon,
    MissingContinuation,
    StateMismatch,
    WorkLimit,
    Terminal,
    MoveLimit,
}

#[derive(Debug, Serialize)]
pub struct Resolution {
    pub actions: Vec<usize>,
    pub selected_samples: Vec<usize>,
    pub legality_corrections: usize,
    pub stop: StopReason,
    pub terminal_white_score: Option<f64>,
}

pub struct TraceGame {
    config: Config,
    position: HistoryBoard,
    moves: Vec<usize>,
    generation: u64,
    episode: u64,
    pending: Option<Ticket>,
}

fn argmax(logits: &[f32], noise: &[f32], legal: impl Iterator<Item = usize>) -> usize {
    let mut best = None;
    let mut score = f32::NEG_INFINITY;
    for action in legal {
        let value = logits[action] + noise[action];
        if value > score {
            best = Some(action);
            score = value;
        }
    }
    best.expect("validated logits and at least one legal action")
}

impl TraceGame {
    pub fn new(config: Config) -> Result<Self, String> {
        if config.size == 0
            || config.size > 1024
            || config.history == 0
            || config.history > 64
            || config.max_game_moves == 0
            || config.max_game_moves > 1_000_000
        {
            return Err("invalid trace game configuration".into());
        }
        let position =
            HistoryBoard::with_scoring(config.size, config.komi, config.history, config.scoring)
                .map_err(|e| e.to_string())?;
        Ok(Self {
            config,
            position,
            moves: Vec::new(),
            generation: 0,
            episode: 0,
            pending: None,
        })
    }
    pub fn position(&self) -> &HistoryBoard {
        &self.position
    }
    pub fn moves(&self) -> &[usize] {
        &self.moves
    }
    pub fn episode(&self) -> u64 {
        self.episode
    }
    pub fn legal_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; self.position.action_count()];
        for action in self.position.legal_actions() {
            mask[action] = true;
        }
        mask
    }
    pub fn recycle_finished(&mut self) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("trace request active".into());
        }
        if self.position.board.is_terminal() || self.moves.len() >= self.config.max_game_moves {
            self.episode = self
                .episode
                .checked_add(1)
                .ok_or("trace episode identity exhausted")?;
            self.position = HistoryBoard::with_scoring(
                self.config.size,
                self.config.komi,
                self.config.history,
                self.config.scoring,
            )
            .map_err(|e| e.to_string())?;
            self.moves.clear();
        }
        Ok(())
    }
    pub fn start(&mut self, models: [u64; 2], contexts: [u64; 2]) -> Result<Ticket, String> {
        if self.pending.is_some()
            || self.position.board.is_terminal()
            || self.moves.len() >= self.config.max_game_moves
        {
            return Err("trace request active or game finished".into());
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or("trace identity exhausted")?;
        let ticket = Ticket {
            generation: self.generation,
            move_number: self.moves.len(),
            models,
            contexts,
            episode: self.episode,
        };
        self.pending = Some(ticket.clone());
        Ok(ticket)
    }
    pub fn cancel(&mut self) {
        self.pending = None;
    }
    pub fn validate_packet(&self, ticket: &Ticket, packet: &Packet<'_>) -> Result<(), String> {
        if self.pending.as_ref() != Some(ticket) {
            return Err("stale trace ticket, model, context or root".into());
        }
        let actions = self.position.action_count();
        if !(1..=64).contains(&packet.samples) || !(1..=64).contains(&packet.horizon) {
            return Err("invalid trace sample count or horizon".into());
        }
        let count = 2 * packet.samples * packet.horizon;
        if count.checked_mul(actions).is_none_or(|n| n > 1 << 26)
            || packet.actions.len() != count
            || packet.own_logits.len() != count * actions
            || packet.noise.len() != packet.horizon * actions
            || packet
                .actions
                .iter()
                .any(|&a| a < 0 || a as usize >= actions)
            || packet.own_logits.iter().any(|x| !x.is_finite())
            || packet.noise.iter().any(|x| !x.is_finite())
        {
            return Err("invalid trace packet dimensions or numbers".into());
        }
        let first_player = (self.position.board.to_play() as usize) - 1;
        let root_legal = if self.config.root_legal_mask {
            self.legal_mask()
        } else {
            Vec::new()
        };
        // Validate every own proposal before mutating real state. Opponent
        // proposals are predictions and have no authority to select a move.
        for view in 0..2 {
            for sample in 0..packet.samples {
                for ply in 0..packet.horizon {
                    if (first_player + ply) % 2 != view {
                        continue;
                    }
                    let row = (view * packet.samples + sample) * packet.horizon + ply;
                    let logits = &packet.own_logits[row * actions..(row + 1) * actions];
                    let noise = &packet.noise[ply * actions..(ply + 1) * actions];
                    if logits
                        .iter()
                        .zip(noise)
                        .any(|(&a, &b)| !(a + b).is_finite())
                        || argmax(
                            logits,
                            noise,
                            (0..actions).filter(|&a| {
                                !self.config.root_legal_mask || ply != 0 || root_legal[a]
                            }),
                        ) != packet.actions[row] as usize
                    {
                        return Err("own proposal differs from the declared sampler".into());
                    }
                }
            }
        }
        Ok(())
    }
    pub fn resolve(&mut self, ticket: &Ticket, packet: Packet<'_>) -> Result<Resolution, String> {
        self.validate_packet(ticket, &packet)?;
        let allowance = packet.horizon;
        self.resolve_validated(packet, None, allowance)
    }
    /// State inputs have layout [2,k,H,size²], in absolute colors 0,1,2.
    /// They describe the inputs used to compute each row's policy, not an
    /// assertion that the device implements Go dynamics or legality.
    pub fn validate_state_packet(
        &self,
        ticket: &Ticket,
        packet: &Packet<'_>,
        states: &[u8],
    ) -> Result<(), String> {
        self.validate_packet(ticket, packet)?;
        let points = self.position.action_count() - 1;
        if states.len() != 2 * packet.samples * packet.horizon * points
            || states.iter().any(|&s| s > 2)
        {
            return Err("invalid trace state dimensions or colors".into());
        }
        let actual = self.position.board.stones();
        for branch in 0..2 * packet.samples {
            let start = branch * packet.horizon * points;
            if states[start..start + points] != actual {
                return Err("trace root state differs from the pending position".into());
            }
        }
        Ok(())
    }
    pub fn resolve_with_states(
        &mut self,
        ticket: &Ticket,
        packet: Packet<'_>,
        states: &[u8],
    ) -> Result<Resolution, String> {
        self.validate_state_packet(ticket, &packet, states)?;
        let allowance = packet.horizon;
        self.resolve_validated(packet, Some(states), allowance)
    }
    pub fn resolve_with_states_limited(
        &mut self,
        ticket: &Ticket,
        packet: Packet<'_>,
        states: &[u8],
        allowance: usize,
    ) -> Result<Resolution, String> {
        if allowance > packet.horizon {
            return Err("work allowance exceeds packet horizon".into());
        }
        self.validate_state_packet(ticket, &packet, states)?;
        self.resolve_validated(packet, Some(states), allowance)
    }
    fn resolve_validated(
        &mut self,
        packet: Packet<'_>,
        states: Option<&[u8]>,
        allowance: usize,
    ) -> Result<Resolution, String> {
        let actions = self.position.action_count();
        self.pending = None;
        let mut viable = vec![true; 2 * packet.samples];
        let mut result = Resolution {
            actions: Vec::new(),
            selected_samples: Vec::new(),
            legality_corrections: 0,
            stop: if allowance < packet.horizon { StopReason::WorkLimit } else { StopReason::Horizon },
            terminal_white_score: None,
        };
        for ply in 0..allowance {
            let player = (self.position.board.to_play() as usize) - 1;
            if !(0..packet.samples).any(|s| viable[player * packet.samples + s]) {
                result.stop = StopReason::MissingContinuation;
                break;
            }
            // A matching action prefix is insufficient for a state-conditioned
            // model. Validate the input to this own-policy prediction before
            // reading its logits. Opponent forecasts themselves have no move
            // authority, and their earlier input errors do not poison a later
            // own prediction if its canonical history and board now agree.
            let actual = states.map(|_| self.position.board.stones());
            let Some(sample) = (0..packet.samples).find(|&sample| {
                if !viable[player * packet.samples + sample] {
                    return false;
                }
                states.is_none_or(|input| {
                    let row = (player * packet.samples + sample) * packet.horizon + ply;
                    input[row * (actions - 1)..(row + 1) * (actions - 1)]
                        == actual.as_ref().unwrap()[..]
                })
            }) else {
                result.stop = StopReason::StateMismatch;
                break;
            };
            let row = (player * packet.samples + sample) * packet.horizon + ply;
            let logits = &packet.own_logits[row * actions..(row + 1) * actions];
            let noise = &packet.noise[ply * actions..(ply + 1) * actions];
            let selected = argmax(logits, noise, self.position.legal_actions().into_iter());
            result.legality_corrections += usize::from(selected != packet.actions[row] as usize);
            if !self.position.commit(selected) {
                return Err("authoritative legal move was rejected".into());
            }
            self.moves.push(selected);
            result.actions.push(selected);
            result.selected_samples.push(sample);
            for (branch, valid) in viable.iter_mut().enumerate() {
                *valid &= packet.actions[branch * packet.horizon + ply] as usize == selected;
            }
            if self.position.board.is_terminal() {
                result.stop = StopReason::Terminal;
                result.terminal_white_score = Some(self.position.score().white_minus_black);
                break;
            }
            if self.moves.len() >= self.config.max_game_moves {
                result.stop = StopReason::MoveLimit;
                break;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn game() -> TraceGame {
        TraceGame::new(Config {
            size: 3,
            komi: 0.5,
            history: 2,
            scoring: Scoring::PassAliveArea,
            max_game_moves: 36,
            root_legal_mask: false,
        })
        .unwrap()
    }
    fn logits(actions: &[i32]) -> Vec<f32> {
        actions
            .iter()
            .flat_map(|&a| (0..10).map(move |i| if i == a { 1. } else { 0. }))
            .collect()
    }
    #[test]
    fn last_opponent_prediction_is_not_required_and_stale_replies_do_nothing() {
        let mut game = game();
        let ticket = game.start([7, 9], [11, 13]).unwrap();
        let paths = [0, 1, 2, 8, 0, 1, 2, 3];
        let p = logits(&paths);
        let noise = [0.; 40];
        let packet = || Packet {
            samples: 1,
            horizon: 4,
            actions: &paths,
            own_logits: &p,
            noise: &noise,
        };
        let mut stale = ticket.clone();
        stale.models[0] += 1;
        assert!(game.resolve(&stale, packet()).is_err());
        assert!(game.moves().is_empty());
        let resolved = game.resolve(&ticket, packet()).unwrap();
        assert_eq!(resolved.actions, vec![0, 1, 2, 3]);
        assert_eq!(resolved.stop, StopReason::Horizon);
        assert!(game.resolve(&ticket, packet()).is_err());
        assert_eq!(game.moves(), &[0, 1, 2, 3]);
    }
    #[test]
    fn priority_ignores_future_agreement_and_legal_correction_stops_stale_paths() {
        let mut game = game();
        let ticket = game.start([1, 1], [0, 0]).unwrap();
        // The first Black trace disagrees with all White forecasts. A resolver
        // choosing the longest future match would wrongly choose Black sample1.
        let paths = [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7];
        let p = logits(&paths);
        let r = game
            .resolve(
                &ticket,
                Packet {
                    samples: 2,
                    horizon: 4,
                    actions: &paths,
                    own_logits: &p,
                    noise: &[0.; 40],
                },
            )
            .unwrap();
        assert_eq!(r.actions, vec![0]);
        assert_eq!(r.stop, StopReason::MissingContinuation);
        let ticket = game.start([1, 1], [0, 0]).unwrap();
        let paths = [0, 1, 0, 1];
        let p = logits(&paths);
        let r = game
            .resolve(
                &ticket,
                Packet {
                    samples: 1,
                    horizon: 2,
                    actions: &paths,
                    own_logits: &p,
                    noise: &[0.; 20],
                },
            )
            .unwrap();
        assert_eq!(r.actions, vec![1]);
        assert_eq!(r.legality_corrections, 1);
        assert_eq!(r.stop, StopReason::MissingContinuation);
    }
    #[test]
    fn malformed_own_policy_and_terminal_cutoff_are_explicit() {
        let mut game = game();
        let ticket = game.start([1, 1], [0, 0]).unwrap();
        let paths = [9; 8];
        let mut p = logits(&paths);
        p[9] = f32::NAN;
        assert!(
            game.resolve(
                &ticket,
                Packet {
                    samples: 1,
                    horizon: 4,
                    actions: &paths,
                    own_logits: &p,
                    noise: &[0.; 40]
                }
            )
            .is_err()
        );
        assert!(game.moves().is_empty());
        p[9] = 1.;
        let r = game
            .resolve(
                &ticket,
                Packet {
                    samples: 1,
                    horizon: 4,
                    actions: &paths,
                    own_logits: &p,
                    noise: &[0.; 40],
                },
            )
            .unwrap();
        assert_eq!(r.actions, vec![9, 9]);
        assert_eq!(r.stop, StopReason::Terminal);
        assert_eq!(r.terminal_white_score, Some(0.5));
        assert!(game.start([1, 1], [0, 0]).is_err());
    }

    #[test]
    fn a_capture_invalidates_the_next_input_before_its_policy_can_act() {
        // B1, W0, B3 captures W0. A place-only device proposal leaves W0 on
        // its predicted board; W8 is nevertheless legal, so legality alone
        // would silently consume a policy computed from the wrong input.
        let paths = [1, 0, 3, 8, 1, 0, 3, 8];
        let p = logits(&paths);
        let packet = || Packet {
            samples: 1,
            horizon: 4,
            actions: &paths,
            own_logits: &p,
            noise: &[0.; 40],
        };
        let mut state = vec![0; 2 * 4 * 9];
        for view in 0..2 {
            for depth in 1..4 {
                let row = (view * 4 + depth) * 9;
                for prior in 0..depth {
                    state[row + paths[prior] as usize] = (prior % 2 + 1) as u8;
                }
            }
        }
        let mut g = game();
        let t = g.start([7, 7], [0, 0]).unwrap();
        let r = g.resolve_with_states(&t, packet(), &state).unwrap();
        assert_eq!(r.actions, vec![1, 0, 3]);
        assert_eq!(r.stop, StopReason::StateMismatch);
        assert!(g.position.legal_actions().contains(&8));
        assert_eq!(g.position.board.stones()[0], 0);

        let mut g = game();
        let t = g.start([7, 7], [0, 0]).unwrap();
        for view in 0..2 {
            state[(view * 4 + 3) * 9] = 0;
        }
        // A wrong input used only for Black's earlier opponent forecast has
        // no authority. The actual prefix and all consumed own inputs agree.
        state[9 + 2] = 2;
        let r = g.resolve_with_states(&t, packet(), &state).unwrap();
        assert_eq!(r.actions, vec![1, 0, 3, 8]);
        assert_eq!(r.stop, StopReason::Horizon);
    }

    #[test]
    fn root_state_and_color_errors_are_rejected_without_consuming_the_ticket() {
        let paths = [9; 8];
        let p = logits(&paths);
        let packet = || Packet {
            samples: 1,
            horizon: 4,
            actions: &paths,
            own_logits: &p,
            noise: &[0.; 40],
        };
        let mut g = game();
        let t = g.start([1, 1], [0, 0]).unwrap();
        let mut state = vec![0; 2 * 4 * 9];
        state[4 * 9] = 1;
        assert!(g.resolve_with_states(&t, packet(), &state).is_err());
        assert!(g.moves().is_empty());
        state[4 * 9] = 0;
        state[3 * 9] = 3;
        assert!(g.resolve_with_states(&t, packet(), &state).is_err());
        assert!(g.moves().is_empty());
        state[3 * 9] = 1; // Unconsumed after the terminal second pass.
        let r = g.resolve_with_states(&t, packet(), &state).unwrap();
        assert_eq!(r.actions, vec![9, 9]);
        assert_eq!(r.stop, StopReason::Terminal);
        assert_eq!(r.terminal_white_score, Some(0.5));
        assert!(g.resolve_with_states(&t, packet(), &state).is_err());
    }

    #[test]
    fn exact_work_limit_stops_before_unused_predictions_and_preserves_terminal_priority() {
        let mut g = game();
        let paths = [9; 8];
        let p = logits(&paths);
        let states = vec![0; 2 * 4 * 9];
        let packet = || Packet { samples: 1, horizon: 4, actions: &paths, own_logits: &p, noise: &[0.; 40] };
        let t = g.start([1, 1], [0, 0]).unwrap();
        assert!(g.resolve_with_states_limited(&t, packet(), &states, 5).is_err());
        assert!(g.moves().is_empty());
        let r = g.resolve_with_states_limited(&t, packet(), &states, 0).unwrap();
        assert!(r.actions.is_empty());
        assert_eq!(r.stop, StopReason::WorkLimit);
        assert!(r.terminal_white_score.is_none());
        assert_eq!(g.episode(), 0);

        let t = g.start([1, 1], [0, 0]).unwrap();
        let r = g.resolve_with_states_limited(&t, packet(), &states, 1).unwrap();
        assert_eq!(r.actions, vec![9]);
        assert_eq!(r.stop, StopReason::WorkLimit);
        assert!(!g.position.board.is_terminal());
        let t = g.start([1, 1], [0, 0]).unwrap();
        let r = g.resolve_with_states_limited(&t, packet(), &states, 1).unwrap();
        assert_eq!(r.actions, vec![9]);
        assert_eq!(r.stop, StopReason::Terminal);
        assert_eq!(r.terminal_white_score, Some(0.5));
        g.recycle_finished().unwrap();
        assert_eq!(g.episode(), 1);
        assert!(g.moves().is_empty());
    }
}
