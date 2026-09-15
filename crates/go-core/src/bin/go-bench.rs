//! Deterministic native rules workload. No neural inference or Go-strength claim.
use go_core::{Board, Move};
use std::time::Instant;

fn next(state: &mut u64) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as usize
}

#[derive(Default)]
struct Counts {
    proposals: u64,
    moves: u64,
    passes: u64,
    terminal_games: u64,
    truncations: u64,
    masks: u64,
}

fn worker(size: usize, iterations: usize, mask_every: usize, seed: u64) -> Counts {
    let mut random = seed;
    let mut counts = Counts::default();
    let mut board = Board::new(size, 7.5).unwrap();
    for i in 0..iterations {
        if board.is_terminal() || board.move_number() >= 3 * size * size + 2 {
            if board.is_terminal() {
                counts.terminal_games += 1;
            } else {
                counts.truncations += 1;
            }
            board = Board::new(size, 7.5).unwrap();
        }
        if mask_every != 0 && i % mask_every == 0 {
            std::hint::black_box(board.legal_moves());
            counts.masks += 1;
        }
        let v = next(&mut random) % (size * size + 1);
        let action = if v == size * size {
            Move::Pass
        } else {
            Move::Play(v)
        };
        counts.proposals += 1;
        if board.play(action).is_ok() {
            counts.moves += 1;
            counts.passes += u64::from(action == Move::Pass);
            board.discard_undo();
        }
    }
    counts
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: go-bench SIZE PROPOSALS_PER_WORKER WORKERS MASK_EVERY SEED");
        std::process::exit(2);
    }
    let size: usize = args[1].parse().unwrap();
    let iterations: usize = args[2].parse().unwrap();
    let workers: usize = args[3].parse().unwrap();
    let mask_every: usize = args[4].parse().unwrap();
    let seed: u64 = args[5].parse().unwrap();
    assert!(workers > 0 && workers <= 4096 && iterations > 0 && seed > 0);
    Board::new(size, 7.5).unwrap();
    // Warm the code and allocator before measuring. Worker boards use first-touch allocation.
    std::hint::black_box(worker(size, 1000, mask_every, seed));
    let start = Instant::now();
    let threads: Vec<_> = (0..workers)
        .map(|i| {
            std::thread::spawn(move || {
                worker(
                    size,
                    iterations,
                    mask_every,
                    seed.wrapping_add(i as u64).max(1),
                )
            })
        })
        .collect();
    let mut counts = Counts::default();
    for thread in threads {
        let value = thread.join().unwrap();
        counts.proposals += value.proposals;
        counts.moves += value.moves;
        counts.passes += value.passes;
        counts.terminal_games += value.terminal_games;
        counts.truncations += value.truncations;
        counts.masks += value.masks;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{{\"kind\":\"native_rules_microbenchmark\",\"size\":{size},\"workers\":{workers},\"seed\":{seed},\"mask_every\":{mask_every},\"proposals\":{},\"legal_moves\":{},\"passes\":{},\"terminal_games\":{},\"truncations\":{},\"legal_masks\":{},\"seconds\":{seconds},\"proposals_per_second\":{},\"legal_moves_per_second\":{}}}",
        counts.proposals,
        counts.moves,
        counts.passes,
        counts.terminal_games,
        counts.truncations,
        counts.masks,
        counts.proposals as f64 / seconds,
        counts.moves as f64 / seconds
    );
}
