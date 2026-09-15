//! Rules/GTP bring-up engine. Random genmove is explicitly not a learned opponent.
use go_core::{Board, Color, Move};
use std::io::{self, BufRead, Write};

const COMMANDS: &[&str] = &[
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "undo",
    "showboard",
    "final_score",
    "legal_moves",
    "quit",
];

fn color(text: &str) -> Result<Color, String> {
    match text.to_ascii_lowercase().as_str() {
        "b" | "black" => Ok(Color::Black),
        "w" | "white" => Ok(Color::White),
        _ => Err("invalid color".into()),
    }
}
fn vertex(text: &str, n: usize) -> Result<Move, String> {
    if text.eq_ignore_ascii_case("pass") {
        return Ok(Move::Pass);
    }
    let bytes = text.as_bytes();
    if bytes.len() < 2 || !bytes[0].is_ascii_alphabetic() {
        return Err("invalid vertex".into());
    }
    let letter = bytes[0].to_ascii_uppercase();
    if letter == b'I' {
        return Err("invalid vertex".into());
    }
    let col = usize::from(letter - b'A') - usize::from(letter > b'I');
    let row: usize = text[1..].parse().map_err(|_| "invalid vertex")?;
    if row == 0 || row > n || col >= n {
        return Err("vertex outside board".into());
    }
    Ok(Move::Play((n - row) * n + col))
}
fn format_vertex(action: Move, n: usize) -> String {
    match action {
        Move::Pass => "pass".into(),
        Move::Play(p) => {
            let col = p % n;
            format!(
                "{}{}",
                (b'A' + col as u8 + u8::from(col >= 8)) as char,
                n - p / n
            )
        }
    }
}
fn execute(parts: &[&str], board: &mut Board, random: &mut u64) -> Result<String, String> {
    let arg = |i: usize| {
        parts
            .get(i)
            .copied()
            .ok_or_else(|| "missing argument".to_string())
    };
    match parts[0] {
        "protocol_version" => Ok("2".into()),
        "name" => Ok("gozero-rules-random".into()),
        "version" => Ok(env!("CARGO_PKG_VERSION").into()),
        "list_commands" => Ok(COMMANDS.join("\n")),
        "known_command" => Ok(COMMANDS.contains(&arg(1)?).to_string()),
        "boardsize" => {
            let n: usize = arg(1)?.parse().map_err(|_| "invalid size")?;
            // GTP's single-letter coordinates have only 25 columns. The core has no such limit.
            if n > 25 {
                return Err(
                    "standard GTP supports at most 25 columns; core supports larger boards".into(),
                );
            }
            *board = Board::new(n, board.komi()).map_err(|e| e.to_string())?;
            Ok(String::new())
        }
        "clear_board" => {
            *board = Board::new(board.size(), board.komi()).unwrap();
            Ok(String::new())
        }
        "komi" => {
            board
                .set_komi(arg(1)?.parse().map_err(|_| "invalid komi")?)
                .map_err(|e| e.to_string())?;
            Ok(String::new())
        }
        "play" | "genmove" => {
            if color(arg(1)?)? != board.to_play() {
                return Err(
                    "out-of-turn move; this strict rules endpoint requires alternating colors"
                        .into(),
                );
            }
            let action = if parts[0] == "play" {
                vertex(arg(2)?, board.size())?
            } else {
                let actions = board.legal_moves();
                if actions.is_empty() {
                    return Err("game is over".into());
                }
                *random ^= *random << 13;
                *random ^= *random >> 7;
                *random ^= *random << 17;
                actions[*random as usize % actions.len()]
            };
            board.play(action).map_err(|e| e.to_string())?;
            Ok(if parts[0] == "genmove" {
                format_vertex(action, board.size())
            } else {
                String::new()
            })
        }
        "undo" => {
            if board.undo() {
                Ok(String::new())
            } else {
                Err("cannot undo".into())
            }
        }
        "legal_moves" => Ok(board
            .legal_moves()
            .iter()
            .map(|&m| format_vertex(m, board.size()))
            .collect::<Vec<_>>()
            .join(" ")),
        "showboard" => Ok(board
            .stones()
            .chunks(board.size())
            .map(|row| {
                row.iter()
                    .map(|c| match c {
                        1 => 'X',
                        2 => 'O',
                        _ => '.',
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")),
        "final_score" => {
            if !board.is_terminal() {
                return Err(
                    "final_score requires two passes under strict Tromp-Taylor rules".into(),
                );
            }
            let value = board.score().white_minus_black;
            Ok(if value == 0.0 {
                "0".into()
            } else {
                format!("{}+{}", if value > 0.0 { 'W' } else { 'B' }, value.abs())
            })
        }
        "quit" => Ok(String::new()),
        _ => Err("unknown command".into()),
    }
}
fn main() {
    let mut board = Board::new(19, 7.5).unwrap();
    let mut random = 1u64;
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let text = line.split('#').next().unwrap_or("");
        let mut parts: Vec<&str> = text.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let id = if parts[0].bytes().all(|c| c.is_ascii_digit()) {
            parts.remove(0)
        } else {
            ""
        };
        if parts.is_empty() {
            continue;
        }
        let (marker, body) = match execute(&parts, &mut board, &mut random) {
            Ok(body) => ('=', body),
            Err(error) => ('?', error),
        };
        if writeln!(stdout, "{marker}{id} {body}\n")
            .and_then(|_| stdout.flush())
            .is_err()
        {
            break;
        }
        if parts[0] == "quit" {
            break;
        }
    }
}
