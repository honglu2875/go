//! Authoritative square-board Tromp–Taylor rules.
//!
//! Chains are circular lists, with incremental liberties and reversible mutations.
//! Positional superko uses a 128-bit filter followed by exact packed-board comparison.
//! Passes are repetition-exempt; two consecutive passes end the game. Raw area
//! scoring is the default; pass-alive adjudication is explicit and leaves board
//! stones unchanged. There is no built-in 19x19 size restriction.

use std::collections::HashMap;
use std::fmt;
mod scoring;
pub use scoring::Scoring;

const EMPTY: u8 = 0;
const BORDER: u8 = 3;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    Black = 1,
    White = 2,
}

impl Color {
    pub fn other(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Black),
            2 => Some(Self::White),
            _ => None,
        }
    }
}

/// A play uses a row-major index, with row zero at the top of the diagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Play(usize),
    Pass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IllegalMove {
    OutOfBounds,
    Occupied,
    Superko,
    GameOver,
}

impl fmt::Display for IllegalMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for IllegalMove {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoardError {
    InvalidSize,
    Allocation,
    InvalidKomi,
    InvalidSetup,
}
impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for BoardError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub black_area: usize,
    pub white_area: usize,
    pub neutral: usize,
    /// Positive means White leads; includes komi.
    pub white_minus_black: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Point {
    color: u8,
    root: u32,
    next: u32,
}
impl Point {
    const EMPTY: Self = Self {
        color: EMPTY,
        root: NONE,
        next: NONE,
    };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Chain {
    stones: u32,
    liberties: u32,
}

#[derive(Clone, Copy, Debug)]
enum Change {
    Point(u32, Point),
    Chain(u32, Chain),
}

#[derive(Clone, Copy, Debug)]
struct Frame {
    journal: usize,
    hash: u128,
    to_play: Color,
    passes: u8,
    move_number: usize,
    history_added: bool,
}

#[derive(Clone, Debug)]
struct Position {
    hash: u128,
    stones: Box<[u64]>,
}

#[derive(Clone, Debug)]
pub struct Board {
    size: usize,
    stride: u32,
    points: Vec<Point>,
    chains: Vec<Chain>,
    packed: Vec<u64>,
    hash: u128,
    history: Vec<Position>,
    history_index: HashMap<u128, Vec<usize>>,
    journal: Vec<Change>,
    frames: Vec<Frame>,
    marks: Vec<u32>,
    epoch: u32,
    to_play: Color,
    passes: u8,
    move_number: usize,
    komi: f64,
}

fn allocated<T: Clone>(n: usize, value: T) -> Result<Vec<T>, BoardError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|_| BoardError::Allocation)?;
    result.resize(n, value);
    Ok(result)
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn stone_hash(point: u32, color: u8) -> u128 {
    if color == EMPTY {
        return 0;
    }
    let key = u64::from(point) * 2 + u64::from(color - 1);
    u128::from(mix(key.wrapping_add(0x9e3779b97f4a7c15)))
        | (u128::from(mix(key.wrapping_add(0xd1b54a32d192ed03))) << 64)
}

impl Board {
    /// The default allocation guard permits up to 1,048,576 playable points.
    /// Applications can choose a different guard with `with_point_limit`.
    pub fn new(size: usize, komi: f64) -> Result<Self, BoardError> {
        Self::with_point_limit(size, komi, 1 << 20)
    }

    pub fn with_point_limit(size: usize, komi: f64, max_points: usize) -> Result<Self, BoardError> {
        if !komi.is_finite() {
            return Err(BoardError::InvalidKomi);
        }
        let area = size.checked_mul(size).ok_or(BoardError::InvalidSize)?;
        let stride = size.checked_add(2).ok_or(BoardError::InvalidSize)?;
        let padded = stride.checked_mul(stride).ok_or(BoardError::InvalidSize)?;
        if size == 0 || area > max_points || padded >= NONE as usize {
            return Err(BoardError::InvalidSize);
        }
        let mut points = allocated(
            padded,
            Point {
                color: BORDER,
                ..Point::EMPTY
            },
        )?;
        for row in 0..size {
            for col in 0..size {
                points[(row + 1) * stride + col + 1] = Point::EMPTY;
            }
        }
        let mut result = Self {
            size,
            stride: stride as u32,
            points,
            chains: allocated(padded, Chain::default())?,
            packed: allocated(padded.div_ceil(32), 0)?,
            hash: 0,
            history: Vec::new(),
            history_index: HashMap::new(),
            journal: Vec::new(),
            frames: Vec::new(),
            marks: allocated(padded, 0)?,
            epoch: 0,
            to_play: Color::Black,
            passes: 0,
            move_number: 0,
            komi,
        };
        result.push_history();
        Ok(result)
    }

    /// Sets an initial position with a fresh superko history. Existing stones must
    /// have liberties; setup is not a way to resume a game without its history.
    pub fn from_setup(
        size: usize,
        komi: f64,
        stones: &[(usize, Color)],
        to_play: Color,
    ) -> Result<Self, BoardError> {
        let mut board = Self::new(size, komi)?;
        for &(index, color) in stones {
            let p = board.padded(index).ok_or(BoardError::InvalidSetup)?;
            if board.points[p as usize].color != EMPTY {
                return Err(BoardError::InvalidSetup);
            }
            board.replace_point(
                p,
                Point {
                    color: color as u8,
                    root: p,
                    next: p,
                },
            );
            board.replace_chain(
                p,
                Chain {
                    stones: 1,
                    liberties: 0,
                },
            );
        }
        for &(index, _) in stones {
            let p = board.padded(index).unwrap();
            for q in board.neighbors(p) {
                if board.points[q as usize].color == board.points[p as usize].color {
                    let a = board.points[p as usize].root;
                    let b = board.points[q as usize].root;
                    if a != b {
                        board.merge(a, b);
                    }
                }
            }
        }
        for &(index, _) in stones {
            let root = board.points[board.padded(index).unwrap() as usize].root;
            let liberties = board.count_liberties(root);
            board.chains[root as usize].liberties = liberties;
            if liberties == 0 {
                return Err(BoardError::InvalidSetup);
            }
        }
        board.to_play = to_play;
        board.discard_undo();
        board.history.clear();
        board.history_index.clear();
        board.push_history();
        Ok(board)
    }

    pub fn size(&self) -> usize {
        self.size
    }
    pub fn area(&self) -> usize {
        self.size * self.size
    }
    pub fn to_play(&self) -> Color {
        self.to_play
    }
    pub fn komi(&self) -> f64 {
        self.komi
    }
    pub fn move_number(&self) -> usize {
        self.move_number
    }
    pub fn consecutive_passes(&self) -> u8 {
        self.passes
    }
    pub fn is_terminal(&self) -> bool {
        self.passes >= 2
    }
    pub fn position_hash(&self) -> u128 {
        self.hash
    }
    pub fn undo_depth(&self) -> usize {
        self.frames.len()
    }
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    pub fn set_komi(&mut self, komi: f64) -> Result<(), BoardError> {
        if !komi.is_finite() {
            return Err(BoardError::InvalidKomi);
        }
        self.komi = komi;
        Ok(())
    }
    pub fn stone_at(&self, index: usize) -> Option<Color> {
        self.padded(index)
            .and_then(|p| Color::from_byte(self.points[p as usize].color))
    }
    pub fn stones(&self) -> Vec<u8> {
        (0..self.area())
            .map(|i| self.points[self.padded(i).unwrap() as usize].color)
            .collect()
    }

    fn padded(&self, index: usize) -> Option<u32> {
        (index < self.area())
            .then(|| (index / self.size + 1) as u32 * self.stride + (index % self.size + 1) as u32)
    }
    fn neighbors(&self, p: u32) -> [u32; 4] {
        [p - self.stride, p - 1, p + 1, p + self.stride]
    }
    fn next_epoch(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }
    fn write_point(&mut self, p: u32, value: Point) {
        let old = self.points[p as usize];
        if old.color != value.color {
            self.hash ^= stone_hash(p, old.color) ^ stone_hash(p, value.color);
            let shift = (p % 32) * 2;
            let word = &mut self.packed[p as usize / 32];
            *word = (*word & !(3 << shift)) | (u64::from(value.color) << shift);
        }
        self.points[p as usize] = value;
    }
    fn replace_point(&mut self, p: u32, value: Point) {
        self.journal.push(Change::Point(p, self.points[p as usize]));
        self.write_point(p, value);
    }
    fn replace_chain(&mut self, root: u32, value: Chain) {
        self.journal
            .push(Change::Chain(root, self.chains[root as usize]));
        self.chains[root as usize] = value;
    }
    fn adjust_liberties(&mut self, root: u32, delta: i32) {
        let mut chain = self.chains[root as usize];
        chain.liberties = chain
            .liberties
            .checked_add_signed(delta)
            .expect("liberty accounting invariant");
        self.replace_chain(root, chain);
    }
    fn count_liberties(&mut self, root: u32) -> u32 {
        let epoch = self.next_epoch();
        let mut count = 0;
        let mut p = root;
        loop {
            for q in self.neighbors(p) {
                if self.points[q as usize].color == EMPTY && self.marks[q as usize] != epoch {
                    self.marks[q as usize] = epoch;
                    count += 1;
                }
            }
            p = self.points[p as usize].next;
            if p == root {
                break;
            }
        }
        count
    }
    fn merge(&mut self, mut a: u32, mut b: u32) {
        if self.chains[a as usize].stones < self.chains[b as usize].stones {
            std::mem::swap(&mut a, &mut b);
        }
        let mut p = b;
        loop {
            let point = self.points[p as usize];
            self.replace_point(p, Point { root: a, ..point });
            p = point.next;
            if p == b {
                break;
            }
        }
        let pa = self.points[a as usize];
        let pb = self.points[b as usize];
        self.replace_point(
            a,
            Point {
                next: pb.next,
                ..pa
            },
        );
        self.replace_point(
            b,
            Point {
                next: pa.next,
                ..pb
            },
        );
        let stones = self.chains[a as usize].stones + self.chains[b as usize].stones;
        let liberties = self.count_liberties(a);
        self.replace_chain(a, Chain { stones, liberties });
        self.replace_chain(b, Chain::default());
    }
    fn remove_chain(&mut self, root: u32) {
        let mut p = root;
        loop {
            let next = self.points[p as usize].next;
            self.replace_point(p, Point::EMPTY);
            let mut seen = [NONE; 4];
            let mut used = 0;
            for q in self.neighbors(p) {
                let adjacent = self.points[q as usize];
                if adjacent.color != EMPTY
                    && adjacent.color != BORDER
                    && adjacent.root != root
                    && !seen[..used].contains(&adjacent.root)
                {
                    seen[used] = adjacent.root;
                    used += 1;
                    self.adjust_liberties(adjacent.root, 1);
                }
            }
            p = next;
            if p == root {
                break;
            }
        }
        self.replace_chain(root, Chain::default());
    }
    fn push_history(&mut self) {
        self.history_index
            .entry(self.hash)
            .or_default()
            .push(self.history.len());
        self.history.push(Position {
            hash: self.hash,
            stones: self.packed.clone().into_boxed_slice(),
        });
    }
    fn repeats(&self) -> bool {
        self.history_index.get(&self.hash).is_some_and(|indices| {
            indices
                .iter()
                .any(|&i| self.history[i].stones.as_ref() == self.packed)
        })
    }
    fn restore(&mut self, frame: Frame) {
        if frame.history_added {
            let position = self.history.pop().unwrap();
            let indices = self.history_index.get_mut(&position.hash).unwrap();
            assert_eq!(indices.pop(), Some(self.history.len()));
            if indices.is_empty() {
                self.history_index.remove(&position.hash);
            }
        }
        while self.journal.len() > frame.journal {
            match self.journal.pop().unwrap() {
                Change::Point(p, value) => self.write_point(p, value),
                Change::Chain(root, value) => self.chains[root as usize] = value,
            }
        }
        debug_assert_eq!(self.hash, frame.hash);
        self.to_play = frame.to_play;
        self.passes = frame.passes;
        self.move_number = frame.move_number;
    }

    pub fn play(&mut self, action: Move) -> Result<(), IllegalMove> {
        if self.is_terminal() {
            return Err(IllegalMove::GameOver);
        }
        let mut frame = Frame {
            journal: self.journal.len(),
            hash: self.hash,
            to_play: self.to_play,
            passes: self.passes,
            move_number: self.move_number,
            history_added: false,
        };
        if let Move::Play(index) = action {
            let p = self.padded(index).ok_or(IllegalMove::OutOfBounds)?;
            if self.points[p as usize].color != EMPTY {
                return Err(IllegalMove::Occupied);
            }
            let neighbors = self.neighbors(p);
            let liberties = neighbors
                .iter()
                .filter(|&&q| self.points[q as usize].color == EMPTY)
                .count() as u32;
            self.replace_point(
                p,
                Point {
                    color: self.to_play as u8,
                    root: p,
                    next: p,
                },
            );
            self.replace_chain(
                p,
                Chain {
                    stones: 1,
                    liberties,
                },
            );
            // All previously adjacent chains lose the occupied liberty exactly once.
            let mut roots = [NONE; 4];
            let mut used = 0;
            for q in neighbors {
                let adjacent = self.points[q as usize];
                if adjacent.color != EMPTY
                    && adjacent.color != BORDER
                    && !roots[..used].contains(&adjacent.root)
                {
                    roots[used] = adjacent.root;
                    used += 1;
                    self.adjust_liberties(adjacent.root, -1);
                }
            }
            for &root in &roots[..used] {
                if self.points[root as usize].color == self.to_play.other() as u8
                    && self.chains[root as usize].liberties == 0
                {
                    self.remove_chain(root);
                }
            }
            for q in neighbors {
                let adjacent = self.points[q as usize];
                let root = self.points[p as usize].root;
                if adjacent.color == self.to_play as u8 && adjacent.root != root {
                    self.merge(root, adjacent.root);
                }
            }
            let root = self.points[p as usize].root;
            if self.chains[root as usize].liberties == 0 {
                self.remove_chain(root);
            }
            if self.repeats() {
                self.restore(frame);
                return Err(IllegalMove::Superko);
            }
            self.push_history();
            frame.history_added = true;
            self.passes = 0;
        } else {
            self.passes += 1;
        }
        self.to_play = self.to_play.other();
        self.move_number += 1;
        self.frames.push(frame);
        Ok(())
    }

    /// Restores a successful move, including turn, pass status, and superko history.
    pub fn undo(&mut self) -> bool {
        if let Some(frame) = self.frames.pop() {
            self.restore(frame);
            true
        } else {
            false
        }
    }
    /// Releases committed-game undo storage without forgetting superko history.
    pub fn discard_undo(&mut self) {
        self.frames.clear();
        self.journal.clear();
    }
    fn chain_points(&self, root: u32) -> impl Iterator<Item = u32> + '_ {
        let mut point = root;
        let mut first = true;
        std::iter::from_fn(move || {
            if !first && point == root {
                return None;
            }
            first = false;
            let current = point;
            point = self.points[point as usize].next;
            Some(current)
        })
    }

    /// Exact legality without temporary board mutation. Only a matching history
    /// hash requires constructing the candidate packed position. Chain liberties
    /// suffice to identify captures and whether the newly joined own group dies.
    pub fn is_legal(&self, action: Move) -> bool {
        if self.is_terminal() {
            return false;
        }
        let Move::Play(index) = action else {
            return true;
        };
        let Some(p) = self.padded(index) else {
            return false;
        };
        if self.points[p as usize].color != EMPTY {
            return false;
        }
        let mut friendly = [NONE; 4];
        let mut friends = 0;
        let mut captured = [NONE; 4];
        let mut captures = 0;
        let mut has_liberty = false;
        for q in self.neighbors(p) {
            let adjacent = self.points[q as usize];
            if adjacent.color == EMPTY {
                has_liberty = true;
            } else if adjacent.color == self.to_play as u8 {
                if !friendly[..friends].contains(&adjacent.root) {
                    friendly[friends] = adjacent.root;
                    friends += 1;
                    has_liberty |= self.chains[adjacent.root as usize].liberties > 1;
                }
            } else if adjacent.color == self.to_play.other() as u8
                && self.chains[adjacent.root as usize].liberties == 1
                && !captured[..captures].contains(&adjacent.root)
            {
                captured[captures] = adjacent.root;
                captures += 1;
            }
        }
        let suicide = !has_liberty && captures == 0;
        // A solitary suicide restores the current position and violates PSK.
        if suicide && friends == 0 {
            return false;
        }
        let mut hash = self.hash;
        if !suicide {
            hash ^= stone_hash(p, self.to_play as u8);
        }
        let removed = if suicide {
            &friendly[..friends]
        } else {
            &captured[..captures]
        };
        for &root in removed {
            for q in self.chain_points(root) {
                hash ^= stone_hash(q, self.points[q as usize].color);
            }
        }
        let Some(indices) = self.history_index.get(&hash) else {
            return true;
        };
        let mut packed = self.packed.clone();
        if !suicide {
            packed[p as usize / 32] |= u64::from(self.to_play as u8) << ((p % 32) * 2);
        }
        for &root in removed {
            for q in self.chain_points(root) {
                packed[q as usize / 32] &= !(3 << ((q % 32) * 2));
            }
        }
        !indices
            .iter()
            .any(|&i| self.history[i].stones.as_ref() == packed)
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut actions = Vec::with_capacity(self.area() + 1);
        for index in 0..self.area() {
            let action = Move::Play(index);
            if self.is_legal(action) {
                actions.push(action);
            }
        }
        if !self.is_terminal() {
            actions.push(Move::Pass);
        }
        actions
    }

    /// Defined on any position; only use terminal results as training outcomes.
    pub fn score(&self) -> Score {
        let mut seen = vec![false; self.points.len()];
        let mut stack = Vec::new();
        let mut black = 0;
        let mut white = 0;
        let mut neutral = 0;
        for index in 0..self.area() {
            let p = self.padded(index).unwrap();
            match self.points[p as usize].color {
                1 => black += 1,
                2 => white += 1,
                _ if !seen[p as usize] => {
                    stack.push(p);
                    seen[p as usize] = true;
                    let mut region = 0;
                    let mut borders = 0;
                    while let Some(q) = stack.pop() {
                        region += 1;
                        for n in self.neighbors(q) {
                            match self.points[n as usize].color {
                                EMPTY if !seen[n as usize] => {
                                    seen[n as usize] = true;
                                    stack.push(n);
                                }
                                1 => borders |= 1,
                                2 => borders |= 2,
                                _ => (),
                            }
                        }
                    }
                    match borders {
                        1 => black += region,
                        2 => white += region,
                        _ => neutral += region,
                    }
                }
                _ => (),
            }
        }
        Score {
            black_area: black,
            white_area: white,
            neutral,
            white_minus_black: white as f64 - black as f64 + self.komi,
        }
    }

    /// Expensive diagnostics for fixtures and differential testing, not actor hot paths.
    pub fn validate(&mut self) -> Result<(), String> {
        let mut hash = 0;
        let mut packed = vec![0u64; self.packed.len()];
        let mut visited = vec![false; self.points.len()];
        for index in 0..self.area() {
            let p = self.padded(index).unwrap();
            let point = self.points[p as usize];
            hash ^= stone_hash(p, point.color);
            packed[p as usize / 32] |= u64::from(point.color) << ((p % 32) * 2);
            if point.color == EMPTY {
                if point != Point::EMPTY {
                    return Err("empty point has chain metadata".into());
                }
                continue;
            }
            if point.root as usize >= self.points.len()
                || self.points[point.root as usize].root != point.root
            {
                return Err("invalid chain root".into());
            }
            for q in self.neighbors(p) {
                if self.points[q as usize].color == point.color
                    && self.points[q as usize].root != point.root
                {
                    return Err("adjacent friendly stones have different roots".into());
                }
            }
            if visited[p as usize] {
                continue;
            }
            let root = point.root;
            let mut q = root;
            let mut count = 0;
            loop {
                if q as usize >= self.points.len() || visited[q as usize] {
                    return Err("invalid chain ring".into());
                }
                let v = self.points[q as usize];
                if v.root != root || v.color != point.color {
                    return Err("inconsistent chain ring".into());
                }
                visited[q as usize] = true;
                count += 1;
                q = v.next;
                if q == root {
                    break;
                }
            }
            let liberties = self.count_liberties(root);
            if liberties == 0
                || self.chains[root as usize]
                    != (Chain {
                        stones: count,
                        liberties,
                    })
            {
                return Err("incorrect chain size/liberties".into());
            }
        }
        if hash != self.hash || packed != self.packed {
            return Err("hash/packed board mismatch".into());
        }
        if !self.repeats() {
            return Err("current position absent from history".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
