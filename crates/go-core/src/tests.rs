use super::*;
use std::collections::HashSet;

// Deliberately independent oracle: unpadded byte grid, flood fills, complete board
// copies, and exact HashSet keys. It shares only the public action/color types.
#[derive(Clone)]
struct Oracle {
    n: usize,
    cells: Vec<u8>,
    side: Color,
    passes: u8,
    komi: f64,
    history: HashSet<Vec<u8>>,
    undo: Vec<(Vec<u8>, Color, u8, bool)>,
}
impl Oracle {
    fn new(n: usize, komi: f64) -> Self {
        let cells = vec![0; n * n];
        Self {
            n,
            cells: cells.clone(),
            side: Color::Black,
            passes: 0,
            komi,
            history: HashSet::from([cells]),
            undo: Vec::new(),
        }
    }
    fn neighbors(&self, p: usize) -> Vec<usize> {
        let mut neighbors = Vec::with_capacity(4);
        if p >= self.n {
            neighbors.push(p - self.n);
        }
        if !p.is_multiple_of(self.n) {
            neighbors.push(p - 1);
        }
        if p % self.n + 1 < self.n {
            neighbors.push(p + 1);
        }
        if p + self.n < self.cells.len() {
            neighbors.push(p + self.n);
        }
        neighbors
    }
    fn group(&self, cells: &[u8], start: usize) -> (Vec<usize>, bool) {
        let mut visited = vec![false; cells.len()];
        let mut stack = vec![start];
        visited[start] = true;
        let mut stones = Vec::new();
        let mut liberty = false;
        while let Some(p) = stack.pop() {
            stones.push(p);
            for q in self.neighbors(p) {
                if cells[q] == 0 {
                    liberty = true;
                }
                if cells[q] == cells[start] && !visited[q] {
                    visited[q] = true;
                    stack.push(q);
                }
            }
        }
        (stones, liberty)
    }
    fn play(&mut self, action: Move) -> Result<(), IllegalMove> {
        if self.passes == 2 {
            return Err(IllegalMove::GameOver);
        }
        let mut cells = self.cells.clone();
        let added = if let Move::Play(p) = action {
            if p >= cells.len() {
                return Err(IllegalMove::OutOfBounds);
            }
            if cells[p] != 0 {
                return Err(IllegalMove::Occupied);
            }
            cells[p] = self.side as u8;
            for q in self.neighbors(p) {
                if cells[q] == self.side.other() as u8 {
                    let (group, has_liberty) = self.group(&cells, q);
                    if !has_liberty {
                        for stone in group {
                            cells[stone] = 0;
                        }
                    }
                }
            }
            let (own, has_liberty) = self.group(&cells, p);
            if !has_liberty {
                for stone in own {
                    cells[stone] = 0;
                }
            }
            if self.history.contains(&cells) {
                return Err(IllegalMove::Superko);
            }
            self.history.insert(cells.clone());
            true
        } else {
            false
        };
        self.undo.push((
            std::mem::replace(&mut self.cells, cells),
            self.side,
            self.passes,
            added,
        ));
        self.passes = if added { 0 } else { self.passes + 1 };
        self.side = self.side.other();
        Ok(())
    }
    fn undo(&mut self) {
        let (cells, side, passes, added) = self.undo.pop().unwrap();
        if added {
            assert!(self.history.remove(&self.cells));
        }
        self.cells = cells;
        self.side = side;
        self.passes = passes;
    }
    fn score(&self) -> Score {
        let mut covered = HashSet::new();
        let mut totals = [0usize; 3];
        for p in 0..self.cells.len() {
            if self.cells[p] != 0 {
                totals[self.cells[p] as usize] += 1;
                continue;
            }
            if covered.contains(&p) {
                continue;
            }
            let (region, _) = self.group(&self.cells, p);
            let mut border = HashSet::new();
            for &q in &region {
                for r in self.neighbors(q) {
                    if self.cells[r] != 0 {
                        border.insert(self.cells[r]);
                    }
                }
                covered.insert(q);
            }
            let owner = if border.len() == 1 {
                *border.iter().next().unwrap() as usize
            } else {
                0
            };
            totals[owner] += region.len();
        }
        Score {
            black_area: totals[1],
            white_area: totals[2],
            neutral: totals[0],
            white_minus_black: totals[2] as f64 - totals[1] as f64 + self.komi,
        }
    }
}

fn setup(rows: &[&str], side: Color) -> Board {
    let n = rows.len();
    let mut stones = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), n);
        for (c, stone) in row.bytes().enumerate() {
            if let Some(color) = Color::from_byte(match stone {
                b'X' => 1,
                b'O' => 2,
                _ => 0,
            }) {
                stones.push((r * n + c, color));
            }
        }
    }
    Board::from_setup(n, 7.5, &stones, side).unwrap()
}
fn equal(board: &mut Board, oracle: &Oracle) {
    assert_eq!(board.stones(), oracle.cells);
    assert_eq!(board.to_play(), oracle.side);
    assert_eq!(board.consecutive_passes(), oracle.passes);
    assert_eq!(board.history_len(), oracle.history.len());
    board.validate().unwrap();
}
fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn capture_ko_pass_and_undo() {
    let mut board = setup(&[".XO..", "XO.O.", ".XO..", ".....", "....."], Color::Black);
    let initial = board.stones();
    let hash = board.position_hash();
    board.play(Move::Play(7)).unwrap();
    assert_eq!(board.stone_at(6), None);
    let captured = board.stones();
    assert_eq!(board.play(Move::Play(6)), Err(IllegalMove::Superko));
    assert_eq!(board.stones(), captured);
    board.validate().unwrap();
    board.play(Move::Pass).unwrap();
    board.play(Move::Pass).unwrap();
    assert!(board.is_terminal());
    assert_eq!(board.play(Move::Play(24)), Err(IllegalMove::GameOver));
    assert!(board.undo());
    assert!(!board.is_terminal());
    assert!(board.undo());
    assert_eq!(board.play(Move::Play(6)), Err(IllegalMove::Superko));
    assert!(board.undo());
    assert_eq!(board.stones(), initial);
    assert_eq!(board.position_hash(), hash);
    assert_eq!(board.history_len(), 1);
    assert!(!board.undo());
    board.validate().unwrap();
}

#[test]
fn suicide_removes_multiple_stones_but_repetition_is_forbidden() {
    let mut board = setup(&[".OOO.", "OX.XO", ".OOO.", ".....", "....."], Color::Black);
    board.play(Move::Play(7)).unwrap();
    for p in [6, 7, 8] {
        assert_eq!(board.stone_at(p), None);
    }
    board.validate().unwrap();
    assert!(board.undo());
    board.validate().unwrap();
    let mut one = Board::new(1, 0.5).unwrap();
    assert_eq!(one.play(Move::Play(0)), Err(IllegalMove::Superko));
    assert_eq!(one.legal_moves(), vec![Move::Pass]);
    one.play(Move::Pass).unwrap();
    one.play(Move::Pass).unwrap();
    assert_eq!(
        one.score(),
        Score {
            black_area: 0,
            white_area: 0,
            neutral: 1,
            white_minus_black: 0.5
        }
    );
}

#[test]
fn scoring_keeps_actual_stones_and_separates_neutral_regions() {
    let board = setup(&["XXXOO", "X.X.O", "XXXOO", ".....", "....."], Color::Black);
    assert_eq!(
        board.score(),
        Score {
            black_area: 9,
            white_area: 5,
            neutral: 11,
            white_minus_black: 3.5
        }
    );
}

#[test]
fn dimensions_and_commit_contract() {
    assert!(Board::new(0, 0.0).is_err());
    assert!(Board::new(usize::MAX, 0.0).is_err());
    assert!(Board::new(9, f64::NAN).is_err());
    assert!(Board::with_point_limit(37, 0.0, 100).is_err());
    for size in [1, 2, 5, 9, 19, 25, 37, 101] {
        let mut board = Board::new(size, 0.0).unwrap();
        assert_eq!(board.score().neutral, size * size);
        assert_eq!(
            board.play(Move::Play(size * size)),
            Err(IllegalMove::OutOfBounds)
        );
        if size > 1 {
            board.play(Move::Play(size * size - 1)).unwrap();
            board.discard_undo();
            assert!(!board.undo());
            assert_eq!(board.history_len(), 2);
            board.play(Move::Pass).unwrap();
            assert!(board.undo());
            board.validate().unwrap();
        }
    }
}

#[test]
fn exact_comparison_rejects_false_hash_matches() {
    let mut board = Board::new(5, 0.0).unwrap();
    board.play(Move::Play(0)).unwrap();
    let hash = board.hash;
    let packed = board.packed.clone();
    board.undo();
    // Force a bucket collision with a different exact position.
    board.history_index.entry(hash).or_default().push(0);
    assert!(board.is_legal(Move::Play(0)));
    board.play(Move::Play(0)).unwrap();
    assert_eq!(board.packed, packed);
    board.undo();
    assert_eq!(board.history_index.get(&hash), Some(&vec![0]));
}

fn differential(iterations: usize) -> usize {
    let sizes = [1, 2, 3, 5, 7, 9, 13, 19, 25, 37];
    let mut checked = 0usize;
    let mut successful = 0usize;
    for (seed, n) in sizes.into_iter().enumerate() {
        let mut random = 0x9e3779b97f4a7c15 ^ (seed as u64 + 1);
        let mut board = Board::new(n, 7.5).unwrap();
        let mut oracle = Oracle::new(n, 7.5);
        for t in 0..iterations {
            if board.is_terminal() || t % (n * n * 3 + 10) == 0 {
                board = Board::new(n, 7.5).unwrap();
                oracle = Oracle::new(n, 7.5);
            }
            if !oracle.undo.is_empty() && rng(&mut random).is_multiple_of(13) {
                let depth = (rng(&mut random) as usize % oracle.undo.len().min(8)) + 1;
                for _ in 0..depth {
                    assert!(board.undo());
                    oracle.undo();
                }
            }
            let action = match rng(&mut random) as usize % (n * n + 12) {
                v if v >= n * n + 2 => Move::Pass,
                v => Move::Play(v),
            };
            let actual = board.play(action);
            let expected = oracle.play(action);
            assert_eq!(actual, expected, "size={n}, trial={t}, action={action:?}");
            successful += usize::from(actual.is_ok());
            checked += 1;
            equal(&mut board, &oracle);
            if t % 251 == 0 {
                assert_eq!(board.score(), oracle.score());
                for p in 0..=n * n {
                    let action = if p == n * n {
                        Move::Pass
                    } else {
                        Move::Play(p)
                    };
                    let result = oracle.play(action);
                    assert_eq!(
                        board.is_legal(action),
                        result.is_ok(),
                        "legal mask n={n} t={t} p={p}"
                    );
                    if result.is_ok() {
                        oracle.undo();
                    }
                    checked += 1;
                }
                equal(&mut board, &oracle);
            }
        }
    }
    eprintln!(
        "differential: {checked} checked proposals; {successful} committed legal moves; sizes={sizes:?}"
    );
    successful
}

#[test]
fn randomized_differential_and_deep_undo() {
    differential(1500);
}

#[test]
#[ignore = "qualification: run explicitly in release mode; at least one million committed transitions"]
fn million_transition_qualification() {
    assert!(differential(400_000) >= 1_000_000);
}
