//! Replay packed complete histories into exact pre-action board observations.
use crate::Scoring;
use go_core::Board;
use go_search::Position;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub size: usize,
    pub komi: f64,
    pub scoring: Scoring,
}

#[derive(Serialize)]
pub struct Outcome {
    pub terminal: bool,
    pub white_score: Option<f64>,
}

pub struct Observations {
    /// Absolute colors: 0 empty, 1 Black, 2 White, before each action.
    pub stones: Vec<u8>,
    pub legal: Vec<bool>,
    pub outcomes: Vec<Outcome>,
}

pub fn replay(config: Config, actions: &[i32], offsets: &[i64]) -> Result<Observations, String> {
    replay_suffix(config, actions, offsets, &vec![0; offsets.len().saturating_sub(1)])
}

/// Replay every action for exact superko, but encode only the requested suffix.
/// Skipped rows still use Board::apply's full legality check. No prefix state,
/// legal masks, hashes or terminal assumptions are supplied by the caller.
pub fn replay_suffix(config: Config, actions: &[i32], offsets: &[i64], starts: &[i64]) -> Result<Observations, String> {
    if !(1..=1024).contains(&config.size)
        || actions.is_empty()
        || actions.len() > 1 << 24
        || offsets.len() < 2
        || starts.len() + 1 != offsets.len()
        || offsets[0] != 0
        || offsets.last().copied() != Some(actions.len() as i64)
        || offsets.windows(2).any(|x| x[0] < 0 || x[0] >= x[1])
        || offsets.windows(2).zip(starts).any(|(x, &s)| s < 0 || s >= x[1] - x[0])
    {
        return Err("invalid packed history dimensions".into());
    }
    let area = config.size * config.size;
    // Bound both result arrays before allocating. Nothing escapes on failure.
    let rows = actions.len() - starts.iter().map(|&x| x as usize).sum::<usize>();
    if rows
        .checked_mul(2 * area + 1)
        .and_then(|n| n.checked_add(offsets.len() * 32))
        .is_none_or(|n| n > 1 << 30)
        || actions.iter().any(|&a| a < 0 || a as usize > area)
    {
        return Err("observation batch exceeds action or resource bound".into());
    }
    let mut result = Observations {
        stones: Vec::with_capacity(rows * area),
        legal: Vec::with_capacity(rows * (area + 1)),
        outcomes: Vec::with_capacity(offsets.len() - 1),
    };
    for (game, pair) in offsets.windows(2).enumerate() {
        let mut board = Board::new(config.size, config.komi).map_err(|e| e.to_string())?;
        for row in pair[0] as usize..pair[1] as usize {
            if board.is_terminal() {
                return Err(format!("game {game} continues after terminal at row {row}"));
            }
            if row >= (pair[0] + starts[game]) as usize {
                result.stones.extend(board.stones());
                let begin = result.legal.len();
                result.legal.resize(begin + area + 1, false);
                for a in board.legal_actions() {
                    result.legal[begin + a] = true;
                }
            }
            let action = actions[row] as usize;
            if !board.apply(action) {
                return Err(format!("game {game} has an illegal action at row {row}"));
            }
            board.discard_undo();
        }
        let terminal = board.is_terminal();
        result.outcomes.push(Outcome {
            terminal,
            white_score: terminal.then(|| board.score_with(config.scoring.core()).white_minus_black),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config { size: 3, komi: 0.5, scoring: Scoring::RawArea }
    }

    #[test]
    fn observations_precede_targets_and_episode_resets_do_not_leak() {
        let r = replay(config(), &[0, 1, 9, 9, 8], &[0, 4, 5]).unwrap();
        assert_eq!(&r.stones[..9], &[0; 9]);
        assert_eq!(r.stones[9], 1);
        assert_eq!(r.stones[9 + 1], 0); // Current target 1 has not been played.
        assert!(!r.legal[10]);
        assert!(r.legal[10 + 1]);
        assert_eq!(&r.stones[4 * 9..5 * 9], &[0; 9]);
        assert!(r.outcomes[0].terminal);
        assert_eq!(r.outcomes[0].white_score, Some(0.5));
        assert!(!r.outcomes[1].terminal);
        assert!(r.outcomes[1].white_score.is_none());
    }

    #[test]
    fn bad_offsets_illegal_moves_and_post_terminal_actions_are_rejected() {
        for (actions, offsets) in [
            (vec![0, 0], vec![0, 2]),
            (vec![9, 9, 0], vec![0, 3]),
            (vec![0], vec![0, 0, 1]),
            (vec![0], vec![0, 2]),
            (vec![0], vec![-1, 1]),
            (vec![-1], vec![0, 1]),
            (vec![10], vec![0, 1]),
        ] {
            assert!(replay(config(), &actions, &offsets).is_err());
        }
    }

    #[test]
    fn suffix_rows_match_complete_replay_including_capture_and_pass() {
        let actions = [0, 1, 3, 4, 8, 6, 0, 9, 9];
        let full = replay(config(), &actions, &[0, 9]).unwrap();
        for start in 0..9 {
            let part = replay_suffix(config(), &actions, &[0, 9], &[start]).unwrap();
            assert_eq!(part.stones, full.stones[start as usize * 9..]);
            assert_eq!(part.legal, full.legal[start as usize * 10..]);
            assert_eq!(part.outcomes[0].terminal, full.outcomes[0].terminal);
            assert_eq!(part.outcomes[0].white_score, full.outcomes[0].white_score);
        }
        let joined = replay_suffix(config(), &[0, 1, 9, 8, 9], &[0, 3, 5], &[2, 0]).unwrap();
        assert_eq!(joined.stones.len(), 3 * 9);
        assert_eq!(joined.legal.len(), 3 * 10);
        assert_eq!(&joined.stones[9..18], &[0; 9]);
    }

    #[test]
    fn omitted_prefix_still_rejects_illegal_and_post_terminal_actions() {
        assert!(replay_suffix(config(), &[0, 0, 1], &[0, 3], &[2]).is_err());
        assert!(replay_suffix(config(), &[9, 9, 1], &[0, 3], &[2]).is_err());
        for starts in [vec![], vec![-1], vec![3], vec![0, 1]] {
            assert!(replay_suffix(config(), &[0, 1, 2], &[0, 3], &starts).is_err());
        }
    }
}
