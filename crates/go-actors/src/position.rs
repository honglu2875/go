use go_core::{Board, BoardError, Color};
use go_search::Position;
use std::collections::VecDeque;

/// Authoritative rules plus finite neural history. The board keeps the complete
/// superko history independently of this observation window.
#[derive(Clone, Debug)]
pub struct HistoryBoard {
    pub board: Board,
    pub scoring: crate::Scoring,
    score_utility: Option<crate::ScoreUtility>,
    depth: usize,
    history: VecDeque<Vec<u8>>,
    removed: Vec<Option<Vec<u8>>>,
}

impl HistoryBoard {
    pub fn new(size: usize, komi: f64, depth: usize) -> Result<Self, BoardError> {
        Self::with_scoring(size, komi, depth, crate::Scoring::RawArea)
    }
    pub fn with_scoring(
        size: usize,
        komi: f64,
        depth: usize,
        scoring: crate::Scoring,
    ) -> Result<Self, BoardError> {
        Self::with_utility(size, komi, depth, scoring, None)
    }
    pub fn with_utility(
        size: usize,
        komi: f64,
        depth: usize,
        scoring: crate::Scoring,
        score_utility: Option<crate::ScoreUtility>,
    ) -> Result<Self, BoardError> {
        if depth == 0 || depth > 64 {
            return Err(BoardError::InvalidSize);
        }
        if score_utility.is_some_and(|u| !u.is_valid()) {
            return Err(BoardError::InvalidSize);
        }
        let board = Board::new(size, komi)?;
        let history = VecDeque::from([board.stones()]);
        Ok(Self {
            board,
            scoring,
            score_utility,
            depth,
            history,
            removed: Vec::new(),
        })
    }
    pub fn channels(&self) -> usize {
        2 * self.depth + 4
    }
    pub fn score(&self) -> go_core::Score {
        self.board.score_with(self.scoring.core())
    }
    pub fn commit(&mut self, action: usize) -> bool {
        if !self.apply(action) {
            return false;
        }
        self.board.discard_undo();
        self.removed.clear();
        true
    }
    /// NHWC planes: own/opponent stones for current then previous positions,
    /// black-to-play, signed komi / area, consecutive passes / 2, point legality.
    pub fn features(&self, legal: &[usize]) -> Vec<f32> {
        let area = self.board.area();
        let channels = self.channels();
        let own = self.board.to_play() as u8;
        let other = self.board.to_play().other() as u8;
        let mut output = vec![0.0; area * channels];
        for (t, stones) in self.history.iter().enumerate() {
            for (p, &stone) in stones.iter().enumerate() {
                output[p * channels + 2 * t] = f32::from(stone == own);
                output[p * channels + 2 * t + 1] = f32::from(stone == other);
            }
        }
        let komi = self.board.komi() as f32 / area as f32
            * if self.board.to_play() == Color::White {
                1.0
            } else {
                -1.0
            };
        for p in 0..area {
            output[p * channels + channels - 4] = f32::from(self.board.to_play() == Color::Black);
            output[p * channels + channels - 3] = komi;
            output[p * channels + channels - 2] = self.board.consecutive_passes() as f32 / 2.0;
        }
        for &p in legal {
            if p < area {
                output[p * channels + channels - 1] = 1.0;
            }
        }
        output
    }
}

impl Position for HistoryBoard {
    fn action_count(&self) -> usize {
        self.board.action_count()
    }
    fn legal_actions(&self) -> Vec<usize> {
        self.board.legal_actions()
    }
    fn terminal_value(&self) -> Option<f32> {
        if !self.board.is_terminal() {
            return None;
        }
        let score = self.score().white_minus_black;
        let white_value = if let Some(utility) = self.score_utility {
            utility.value(score, self.board.size())
        } else if score > 0.0 {
            1.0
        } else if score < 0.0 {
            -1.0
        } else {
            0.0
        };
        Some(if self.board.to_play() == Color::White {
            white_value
        } else {
            -white_value
        })
    }
    fn apply(&mut self, action: usize) -> bool {
        if !self.board.apply(action) {
            return false;
        }
        self.history.push_front(self.board.stones());
        self.removed.push(if self.history.len() > self.depth {
            self.history.pop_back()
        } else {
            None
        });
        true
    }
    fn undo(&mut self) -> bool {
        let Some(old) = self.removed.pop() else {
            return false;
        };
        assert!(self.board.undo());
        self.history.pop_front();
        if let Some(old) = old {
            self.history.push_back(old);
        }
        true
    }
}
