//! Native persistent workers own games, searches, features, and outcome targets.
mod actor;
pub mod game;
pub mod observations;
mod pool;
pub mod position;
pub mod traces;
pub use pool::Pool;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Scoring {
    #[default]
    RawArea,
    PassAliveArea,
}
impl Scoring {
    pub fn core(self) -> go_core::Scoring {
        match self {
            Self::RawArea => go_core::Scoring::RawArea,
            Self::PassAliveArea => go_core::Scoring::PassAliveArea,
        }
    }
}

/// Static bounded score preference, in addition to the win/loss objective.
/// The arctangent uses score / (scale * sqrt(board area)). Search utilities,
/// exploration and FPU reductions share the same 1 + factor normalization.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScoreUtility {
    pub factor: f32,
    pub scale: f32,
}
impl ScoreUtility {
    pub fn is_valid(self) -> bool {
        self.factor.is_finite()
            && (0.0..=2.0).contains(&self.factor)
            && self.scale.is_finite()
            && (0.001..=100.0).contains(&self.scale)
    }
    pub fn divisor(self) -> f32 {
        1.0 + self.factor
    }
    pub fn value(self, score: f64, size: usize) -> f32 {
        let win = if score > 0.0 {
            1.0
        } else if score < 0.0 {
            -1.0
        } else {
            0.0
        };
        let score_value =
            (score / (f64::from(self.scale) * size as f64)).atan() * std::f64::consts::FRAC_2_PI;
        ((win + f64::from(self.factor) * score_value) / (1.0 + f64::from(self.factor))) as f32
    }
}

/// Full Gumbel AlphaZero parameters; root perturbations use the saved actor RNG.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GumbelSettings {
    pub max_considered_actions: usize,
    pub value_scale: f32,
    pub maxvisit_init: f32,
    pub rescale_values: bool,
    pub gumbel_scale: f32,
}
impl GumbelSettings {
    pub fn search(self) -> go_search::GumbelConfig {
        go_search::GumbelConfig {
            max_considered_actions: self.max_considered_actions,
            value_scale: self.value_scale,
            maxvisit_init: self.maxvisit_init,
            rescale_values: self.rescale_values,
        }
    }
    pub fn is_valid(self) -> bool {
        self.search().is_valid()
            && self.gumbel_scale.is_finite()
            && (0.0..=10.0).contains(&self.gumbel_scale)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub size: usize,
    pub komi: f64,
    #[serde(default)]
    pub scoring: Scoring,
    pub history: usize,
    pub games: usize,
    pub workers: usize,
    /// Empty disables pinning; otherwise one explicit allowed CPU per worker.
    pub worker_cpus: Vec<usize>,
    pub simulations: u32,
    pub cpuct: f32,
    /// None uses zero FPU; Some(r) uses the node's network value minus r.
    #[serde(default)]
    pub fpu_reduction: Option<f32>,
    #[serde(default)]
    pub score_utility: Option<ScoreUtility>,
    #[serde(default)]
    pub gumbel: Option<GumbelSettings>,
    pub max_search_edges: usize,
    pub max_game_moves: usize,
    pub dirichlet_alpha: f64,
    pub dirichlet_fraction: f32,
    pub temperature_early: f32,
    pub temperature_late: f32,
    pub temperature_moves: usize,
    pub seed: u64,
    /// Distinguishes actor/game identities on different hosts or independent pools.
    pub actor_offset: u64,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.size == 0
            || self.size > 1024
            || !self.komi.is_finite()
            || self.history == 0
            || self.history > 64
            || self.games == 0
            || self.workers == 0
            || self.workers > self.games
            || self.workers > 4096
            || (!self.worker_cpus.is_empty() && self.worker_cpus.len() != self.workers)
            || self
                .actor_offset
                .checked_add(self.games as u64)
                .is_none_or(|n| n > u64::from(u32::MAX))
            || self.max_game_moves == 0
            || self.simulations > 1_000_000
            || !self.cpuct.is_finite()
            || self.cpuct < 0.0
            || self
                .fpu_reduction
                .is_some_and(|r| !r.is_finite() || !(0.0..=2.0).contains(&r))
            || self.score_utility.is_some_and(|u| !u.is_valid())
            || self.gumbel.is_some_and(|g| {
                !g.is_valid()
                    || self.simulations == 0
                    || self.cpuct != 0.0
                    || self.fpu_reduction.is_some()
                    || self.dirichlet_fraction != 0.0
                    || self.temperature_early != 0.0
                    || self.temperature_late != 0.0
                    || self.temperature_moves != 0
            })
            || self.max_search_edges < self.actions()
            || !self.dirichlet_alpha.is_finite()
            || self.dirichlet_alpha <= 0.0
            || !self.dirichlet_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.dirichlet_fraction)
            || [self.temperature_early, self.temperature_late]
                .iter()
                .any(|t| !t.is_finite() || *t < 0.0)
        {
            return Err("invalid actor configuration".into());
        }
        let mut unique = self.worker_cpus.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != self.worker_cpus.len() {
            return Err("worker CPUs must be distinct".into());
        }
        let count = self
            .games
            .checked_mul(self.feature_width())
            .ok_or("feature allocation overflow")?;
        if count > (1usize << 28) {
            return Err("feature batch exceeds 1 GiB allocation guard".into());
        }
        Ok(())
    }
    pub fn channels(&self) -> usize {
        self.history * 2 + 4
    }
    pub fn actions(&self) -> usize {
        self.size * self.size + 1
    }
    pub fn feature_width(&self) -> usize {
        self.size * self.size * self.channels()
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Rebuilt from the complete move sequence on restore; never trusted from disk.
    #[serde(skip)]
    pub features: Vec<f32>,
    pub policy: Vec<f32>,
    pub black_to_play: bool,
    pub action: usize,
    pub network: u64,
    pub root_value: f32,
    pub simulations: u32,
    pub neural_evaluations: u32,
    pub terminal_evaluations: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PoolState {
    pub schema_version: u32,
    pub config: Config,
    pub round: u64,
    pub network: u64,
    pub actors: Vec<actor::ActorState>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct GameRecord {
    pub game_id: u64,
    pub actor_id: u64,
    pub size: usize,
    pub komi: f64,
    pub scoring: Scoring,
    pub actions: Vec<usize>,
    pub networks: Vec<u64>,
    pub white_score: Option<f64>,
    /// Terminal area ownership from White's perspective: Black -1, neutral 0, White 1.
    pub ownership: Vec<i8>,
    pub truncated: bool,
    pub neural_evaluations: u64,
    pub simulations: u64,
    #[serde(skip)]
    pub rows: Vec<Row>,
}

#[derive(Debug)]
pub struct Batch {
    pub round: u64,
    pub network: u64,
    pub features: Vec<f32>,
    pub active: Vec<u8>,
}
impl Batch {
    pub fn active_count(&self) -> usize {
        self.active.iter().filter(|&&v| v != 0).count()
    }
}

#[cfg(test)]
mod tests;
