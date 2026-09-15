//! Explicit terminal ownership profiles, on runtime-sized boards.
//!
//! The pass-alive algorithm follows KataGo's calculateArea/calculateAreaForPla
//! with nonPassAliveStones, safeBigTerritories and unsafeBigTerritories enabled,
//! multi-stone suicide legal, and no territory tax. See KATAGO_LICENSE.md.
use crate::{Board, Color, EMPTY, NONE, Score};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scoring {
    #[default]
    RawArea,
    PassAliveArea,
}

struct Region {
    points: Vec<u32>,
    vital: Vec<u32>,
    internal: u8,
    opponent: bool,
    active: bool,
}

impl Board {
    /// Row-major colors: 0 neutral, 1 Black, 2 White. Does not change the board.
    pub fn ownership(&self, profile: Scoring) -> Vec<u8> {
        let mut owner = vec![EMPTY; self.points.len()];
        if profile == Scoring::PassAliveArea {
            self.mark_pass_alive(Color::Black as u8, &mut owner);
            self.mark_pass_alive(Color::White as u8, &mut owner);
        } else {
            let mut seen = vec![false; self.points.len()];
            for index in 0..self.area() {
                let p = self.padded(index).unwrap();
                if self.points[p as usize].color != EMPTY || seen[p as usize] {
                    continue;
                }
                let mut queue = vec![p];
                seen[p as usize] = true;
                let mut cursor = 0;
                let mut borders = 0;
                while cursor < queue.len() {
                    let q = queue[cursor];
                    cursor += 1;
                    for n in self.neighbors(q) {
                        match self.points[n as usize].color {
                            EMPTY if !seen[n as usize] => {
                                seen[n as usize] = true;
                                queue.push(n);
                            }
                            1 => borders |= 1,
                            2 => borders |= 2,
                            _ => (),
                        }
                    }
                }
                let color = if borders == 1 || borders == 2 {
                    borders
                } else {
                    EMPTY
                };
                for p in queue {
                    owner[p as usize] = color;
                }
            }
        }
        (0..self.area())
            .map(|index| {
                let p = self.padded(index).unwrap() as usize;
                if owner[p] == EMPTY {
                    self.points[p].color
                } else {
                    owner[p]
                }
            })
            .collect()
    }

    pub fn score_with(&self, profile: Scoring) -> Score {
        if profile == Scoring::RawArea {
            return self.score();
        }
        let owner = self.ownership(profile);
        let black = owner.iter().filter(|&&c| c == 1).count();
        let white = owner.iter().filter(|&&c| c == 2).count();
        Score {
            black_area: black,
            white_area: white,
            neutral: self.area() - black - white,
            white_minus_black: white as f64 - black as f64 + self.komi,
        }
    }

    fn adjacent_roots(&self, p: u32, player: u8) -> Vec<u32> {
        let mut roots = Vec::with_capacity(4);
        for n in self.neighbors(p) {
            let point = self.points[n as usize];
            if point.color == player && !roots.contains(&point.root) {
                roots.push(point.root);
            }
        }
        roots
    }

    fn mark_pass_alive(&self, player: u8, owner: &mut [u8]) {
        let opponent = 3 - player;
        let mut region_at = vec![NONE; self.points.len()];
        let mut regions = Vec::<Region>::new();
        let mut roots = Vec::new();
        for index in 0..self.area() {
            let p = self.padded(index).unwrap();
            let point = self.points[p as usize];
            if point.color == player && point.root == p {
                roots.push(p);
            }
            if point.color != EMPTY || region_at[p as usize] != NONE {
                continue;
            }
            let id = regions.len() as u32;
            let mut region = Region {
                points: vec![p],
                vital: self.adjacent_roots(p, player),
                internal: 0,
                opponent: false,
                active: true,
            };
            region_at[p as usize] = id;
            let mut cursor = 0;
            while cursor < region.points.len() {
                let q = region.points[cursor];
                cursor += 1;
                let adjacent = self.adjacent_roots(q, player);
                // Suicide is legal in these profiles: opponent points also constrain vitality.
                region.vital.retain(|root| adjacent.contains(root));
                if adjacent.is_empty() {
                    region.internal = (region.internal + 1).min(2);
                }
                region.opponent |= self.points[q as usize].color == opponent;
                for n in self.neighbors(q) {
                    let color = self.points[n as usize].color;
                    if (color == EMPTY || color == opponent) && region_at[n as usize] == NONE {
                        region_at[n as usize] = id;
                        region.points.push(n);
                    }
                }
            }
            regions.push(region);
        }
        let mut vital_count = vec![0usize; self.points.len()];
        for region in &regions {
            for &root in &region.vital {
                vital_count[root as usize] += 1;
            }
        }
        let mut dead = vec![false; self.points.len()];
        let mut queue: Vec<u32> = roots
            .iter()
            .copied()
            .filter(|&root| vital_count[root as usize] < 2)
            .collect();
        while let Some(root) = queue.pop() {
            if dead[root as usize] {
                continue;
            }
            dead[root as usize] = true;
            let mut p = root;
            loop {
                for n in self.neighbors(p) {
                    let id = region_at[n as usize];
                    if id == NONE || !regions[id as usize].active {
                        continue;
                    }
                    regions[id as usize].active = false;
                    for &vital_root in &regions[id as usize].vital {
                        vital_count[vital_root as usize] -= 1;
                        if vital_count[vital_root as usize] < 2 && !dead[vital_root as usize] {
                            queue.push(vital_root);
                        }
                    }
                }
                p = self.points[p as usize].next;
                if p == root {
                    break;
                }
            }
        }
        for &root in &roots {
            if dead[root as usize] {
                continue;
            }
            let mut p = root;
            loop {
                owner[p as usize] = player;
                p = self.points[p as usize].next;
                if p == root {
                    break;
                }
            }
        }
        if roots.is_empty() {
            return;
        }
        for region in regions {
            if region.active && (region.internal <= 1 || !region.opponent) {
                // Proven pass-alive territory supersedes earlier unsafe territory.
                for p in region.points {
                    owner[p as usize] = player;
                }
            } else if !region.opponent {
                for p in region.points {
                    if owner[p as usize] == EMPTY {
                        owner[p as usize] = player;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Move;

    const ACTUAL_BOARD: &str = ".O.OO.O.O\nO.OO.O.OO\nOOO.OOOO.\nO.OO.OO.O\n.O.OO.OOO\nOOOOOO.OO\n.OOO.OO.O\nOO.OO.OOO\nO.OX.O.O.";

    #[test]
    fn pinned_katago_regression_and_all_symmetries_both_colors() {
        // Retained from a real game against the pinned official KataGo binary.
        // Raw W+85.5 and KataGo W+88.5 differ because one black stone is pass-dead.
        let stones: Vec<_> = ACTUAL_BOARD
            .bytes()
            .filter(|&c| c != b'\n')
            .enumerate()
            .filter_map(|(p, c)| match c {
                b'X' => Some((p, Color::Black)),
                b'O' => Some((p, Color::White)),
                _ => None,
            })
            .collect();
        for swap in [false, true] {
            for symmetry in 0..8 {
                let setup: Vec<_> = stones
                    .iter()
                    .map(|&(p, c)| {
                        let (mut y, mut x) = (p / 9, p % 9);
                        if symmetry >= 4 {
                            x = 8 - x;
                        }
                        for _ in 0..symmetry % 4 {
                            (y, x) = (x, 8 - y);
                        }
                        (y * 9 + x, if swap { c.other() } else { c })
                    })
                    .collect();
                let mut board = Board::from_setup(9, 7.5, &setup, Color::Black).unwrap();
                let before = board.stones();
                let hash = board.position_hash();
                assert_eq!(
                    board.score().white_minus_black,
                    if swap { -70.5 } else { 85.5 }
                );
                assert_eq!(
                    board.score_with(Scoring::PassAliveArea).white_minus_black,
                    if swap { -73.5 } else { 88.5 }
                );
                assert!(
                    board
                        .ownership(Scoring::PassAliveArea)
                        .iter()
                        .all(|&c| c == if swap { 1 } else { 2 })
                );
                assert_eq!(board.stones(), before);
                assert_eq!(board.position_hash(), hash);
                board.validate().unwrap();
            }
        }
    }

    #[test]
    fn raw_ownership_agrees_with_independent_existing_score_across_sizes() {
        let mut rng = 27u64;
        for size in [1, 3, 5, 9, 19, 37] {
            let mut board = Board::new(size, 0.5).unwrap();
            for _ in 0..256 {
                let owner = board.ownership(Scoring::RawArea);
                let score = board.score();
                assert_eq!(owner.iter().filter(|&&c| c == 1).count(), score.black_area);
                assert_eq!(owner.iter().filter(|&&c| c == 2).count(), score.white_area);
                let moves = board.legal_moves();
                if moves.is_empty() {
                    board = Board::new(size, 0.5).unwrap();
                    continue;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                board.play(moves[rng as usize % moves.len()]).unwrap();
                board.discard_undo();
            }
            let mut empty = Board::new(size, 0.5).unwrap();
            empty.play(Move::Pass).unwrap();
            empty.play(Move::Pass).unwrap();
            assert_eq!(
                empty.ownership(Scoring::PassAliveArea),
                vec![0; size * size]
            );
        }
    }
}
