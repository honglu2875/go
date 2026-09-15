//! Worker-owned, resumable PUCT search with one outstanding neural leaf per tree.
//!
//! Neural values use the leaf's side-to-move perspective. Edges store their
//! parent's perspective. Root inference is counted separately from simulations;
//! a simulation is exactly one completed traversal from the root. No virtual
//! visits, transpositions, cross-version cache, or implicit tree reuse are used.

use go_core::{Board, Color, Move};
use std::fmt;
mod gumbel;
pub use gumbel::GumbelConfig;

const UNSET: u32 = u32::MAX;

/// Deterministic alternating-player, zero-sum position with reversible moves.
/// `apply` must leave the position unchanged when it returns false.
pub trait Position {
    fn action_count(&self) -> usize;
    /// Unique actions in strictly increasing canonical order.
    fn legal_actions(&self) -> Vec<usize>;
    fn apply(&mut self, action: usize) -> bool;
    fn undo(&mut self) -> bool;
    /// Terminal result in the current side-to-move perspective, in [-1, 1].
    fn terminal_value(&self) -> Option<f32>;
}

impl Position for Board {
    fn action_count(&self) -> usize {
        self.area() + 1
    }
    fn legal_actions(&self) -> Vec<usize> {
        self.legal_moves()
            .into_iter()
            .map(|m| match m {
                Move::Play(p) => p,
                Move::Pass => self.area(),
            })
            .collect()
    }
    fn apply(&mut self, action: usize) -> bool {
        self.play(if action == self.area() {
            Move::Pass
        } else {
            Move::Play(action)
        })
        .is_ok()
    }
    fn undo(&mut self) -> bool {
        Board::undo(self)
    }
    fn terminal_value(&self) -> Option<f32> {
        if !self.is_terminal() {
            return None;
        }
        let score = self.score().white_minus_black;
        let white_value = if score > 0.0 {
            1.0
        } else if score < 0.0 {
            -1.0
        } else {
            0.0
        };
        Some(if self.to_play() == Color::White {
            white_value
        } else {
            -white_value
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FirstPlayUrgency {
    Zero,
    NetworkValue { reduction: f32 },
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Completed simulations, excluding the root neural evaluation.
    pub simulations: u32,
    /// U = cpuct * P * sqrt(1 + completed outgoing visits) / (1 + edge visits).
    pub cpuct: f32,
    pub fpu: FirstPlayUrgency,
    /// Bounds total edge storage; exhaustion is an error, never a game result.
    pub max_edges: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            simulations: 64,
            cpuct: 1.5,
            fpu: FirstPlayUrgency::Zero,
            max_edges: 4_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RootNoise {
    pub fraction: f32,
    /// Nonnegative action weights, masked and normalized over legal root actions.
    pub weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub search: u64,
    pub network: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub identity: Identity,
    pub sequence: u64,
    pub is_root: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    NeedsEvaluation(Request),
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidConfig,
    InvalidPolicy,
    InvalidValue,
    InvalidNoise,
    InvalidPosition,
    Capacity,
    NoRequest,
    WrongRequest,
    NotComplete,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
struct Edge {
    action: u32,
    prior: f32,
    /// Stable logit relative to the largest legal logit, before any root noise.
    log_prior: f64,
    child: u32,
    visits: u32,
    value_sum: f64,
}
#[derive(Clone, Copy, Debug)]
struct Node {
    first_edge: usize,
    edge_count: u32,
    expanded: bool,
    terminal: bool,
    /// Completed outgoing traversals; always equals the sum of child-edge visits.
    visits: u32,
    value_sum: f64,
    network_value: f32,
}
impl Node {
    fn new(terminal: Option<f32>) -> Self {
        Self {
            first_edge: 0,
            edge_count: 0,
            expanded: terminal.is_some(),
            terminal: terminal.is_some(),
            visits: 0,
            value_sum: 0.0,
            network_value: terminal.unwrap_or(0.0),
        }
    }
}

struct Pending {
    request: Request,
    node: u32,
    legal: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootStats {
    pub completed_simulations: u32,
    pub neural_evaluations: u32,
    pub terminal_evaluations: u32,
    pub visits: Vec<u32>,
    pub priors: Vec<f32>,
    pub action_values: Vec<Option<f32>>,
    pub value: f32,
    pub nodes: usize,
    pub edges: usize,
    /// Gumbel policy improvement is distinct from completed visitation counts.
    pub improved_policy: Option<Vec<f32>>,
    /// Gumbel's recommended move includes its sampled root perturbation.
    pub recommended_action: Option<usize>,
}

/// Read-only completed-root evidence; unrounded sums allow independent checks
/// of value transformations without changing selection, backups or finish.
#[derive(Clone, Debug, PartialEq)]
pub struct RootInspection {
    pub stats: RootStats,
    pub network_value: f32,
    pub visits: u32,
    pub log_priors: Vec<Option<f64>>,
    pub value_sums: Vec<Option<f64>>,
    pub terminal_child_values: Vec<Option<f32>>,
}
impl RootStats {
    /// The configured policy-improvement target; PUCT uses completed visits.
    pub fn policy(&self) -> Vec<f32> {
        if let Some(policy) = &self.improved_policy {
            return policy.clone();
        }
        if self.completed_simulations == 0 {
            return self.priors.clone();
        }
        self.visits
            .iter()
            .map(|&n| n as f32 / self.completed_simulations as f32)
            .collect()
    }
    pub fn best_action(&self) -> Option<usize> {
        if self.recommended_action.is_some() {
            return self.recommended_action;
        }
        let policy = self.policy();
        policy
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .max_by(|(a, p), (b, q)| p.total_cmp(q).then_with(|| b.cmp(a)))
            .map(|(i, _)| i)
    }
}

pub struct Search<P: Position> {
    identity: Identity,
    config: Config,
    position: P,
    actions: usize,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    /// (parent node, selected edge) for the single in-flight simulation.
    path: Vec<(u32, usize)>,
    pending: Option<Pending>,
    noise: Option<RootNoise>,
    gumbel: Option<gumbel::GumbelState>,
    next_sequence: u64,
    completed: u32,
    neural_evaluations: u32,
    terminal_evaluations: u32,
    failure: Option<Error>,
}

impl<P: Position> Search<P> {
    pub fn new(
        position: P,
        config: Config,
        identity: Identity,
        noise: Option<RootNoise>,
    ) -> Result<Self, Error> {
        let actions = position.action_count();
        let fpu_valid = match config.fpu {
            FirstPlayUrgency::Zero => true,
            FirstPlayUrgency::NetworkValue { reduction } => {
                reduction.is_finite() && (0.0..=2.0).contains(&reduction)
            }
        };
        if !config.cpuct.is_finite()
            || config.cpuct < 0.0
            || config.simulations > 1_000_000
            || actions == 0
            || actions >= UNSET as usize
            || !fpu_valid
            || config.max_edges == 0
        {
            return Err(Error::InvalidConfig);
        }
        if let Some(noise) = &noise
            && (!noise.fraction.is_finite()
                || !(0.0..=1.0).contains(&noise.fraction)
                || noise.weights.len() != actions
                || noise.weights.iter().any(|v| !v.is_finite() || *v < 0.0))
        {
            return Err(Error::InvalidNoise);
        }
        let terminal = position.terminal_value();
        if terminal.is_some_and(|v| !v.is_finite() || !(-1.0..=1.0).contains(&v)) {
            return Err(Error::InvalidPosition);
        }
        Ok(Self {
            identity,
            config,
            position,
            actions,
            nodes: vec![Node::new(terminal)],
            edges: Vec::new(),
            path: Vec::new(),
            pending: None,
            noise,
            gumbel: None,
            next_sequence: 1,
            completed: 0,
            neural_evaluations: 0,
            terminal_evaluations: 0,
            failure: None,
        })
    }

    /// Full Gumbel AlphaZero selection. Draws are explicit scaled Gumbel values
    /// owned by the actor RNG; all-zero draws give deterministic evaluation.
    pub fn with_gumbel(
        position: P,
        config: Config,
        identity: Identity,
        gumbel: GumbelConfig,
        draws: Vec<f32>,
    ) -> Result<Self, Error> {
        if !gumbel.is_valid()
            || config.simulations == 0
            || draws.len() != position.action_count()
            || draws.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidConfig);
        }
        let mut search = Self::new(position, config, identity, None)?;
        search.gumbel = Some(gumbel::GumbelState::new(gumbel, draws));
        Ok(search)
    }

    /// Available only while a neural request is pending. Encoding copies features,
    /// not a mutable board: workers retain ownership while the accelerator runs.
    pub fn pending_position(&self) -> Option<&P> {
        self.pending.as_ref().map(|_| &self.position)
    }
    pub fn pending_legal_actions(&self) -> Option<&[usize]> {
        self.pending.as_ref().map(|p| p.legal.as_slice())
    }
    /// Root-relative action path of the currently pending neural leaf.
    /// This is an observation only: it never advances, backs up or rewinds search.
    pub fn pending_actions(&self) -> Option<Vec<usize>> {
        self.pending.as_ref().map(|_| {
            self.path
                .iter()
                .map(|&(_, edge)| self.edges[edge].action as usize)
                .collect()
        })
    }
    pub fn completed_simulations(&self) -> u32 {
        self.completed
    }

    /// Optional read-only proposals for batched inference of Gumbel root
    /// children. The caller owns the root board and prediction cache. Proposals
    /// neither advance a position nor allocate nodes or count simulations.
    pub fn root_sweep_candidates(&self, limit: usize) -> Vec<usize> {
        let node = &self.nodes[0];
        if self.failure.is_some()
            || !node.expanded
            || node.terminal
            || self.completed >= self.config.simulations
        {
            return Vec::new();
        }
        self.gumbel.as_ref().map_or_else(Vec::new, |gumbel| {
            let edges = &self.edges[node.first_edge..node.first_edge + node.edge_count as usize];
            gumbel.root_sweep_candidates(node, edges, self.actions, limit)
        })
    }

    /// Identifies a pending, unevaluated direct child of this exact search's
    /// root. A cache scoped to this Search/Identity may satisfy this request.
    /// Deeper leaves and root requests deliberately have no root-child key.
    pub fn pending_root_child_action(&self) -> Option<usize> {
        let pending = self.pending.as_ref()?;
        if pending.request.is_root || self.path.len() != 1 || self.path[0].0 != 0 {
            return None;
        }
        Some(self.edges[self.path[0].1].action as usize)
    }

    fn request(&mut self, node: u32) -> Result<Progress, Error> {
        let legal = self.position.legal_actions();
        if legal.is_empty()
            || legal.iter().any(|&a| a >= self.actions)
            || legal.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(Error::InvalidPosition);
        }
        if self
            .edges
            .len()
            .checked_add(legal.len())
            .is_none_or(|n| n > self.config.max_edges)
        {
            return Err(Error::Capacity);
        }
        let request = Request {
            identity: self.identity,
            sequence: self.next_sequence,
            is_root: node == 0,
        };
        self.next_sequence += 1;
        self.pending = Some(Pending {
            request,
            node,
            legal,
        });
        Ok(Progress::NeedsEvaluation(request))
    }

    fn select(&self, index: u32) -> usize {
        let node = &self.nodes[index as usize];
        if let Some(gumbel) = &self.gumbel {
            let edges = &self.edges[node.first_edge..node.first_edge + node.edge_count as usize];
            return node.first_edge + gumbel.select(index == 0, node, edges, self.actions);
        }
        let fpu = match self.config.fpu {
            FirstPlayUrgency::Zero => 0.0,
            FirstPlayUrgency::NetworkValue { reduction } => {
                (node.network_value - reduction).max(-1.0)
            }
        };
        let scale = f64::from(self.config.cpuct) * f64::from(node.visits + 1).sqrt();
        let mut best = node.first_edge;
        let mut best_score = f64::NEG_INFINITY;
        for i in node.first_edge..node.first_edge + node.edge_count as usize {
            let edge = &self.edges[i];
            let q = if edge.visits == 0 {
                f64::from(fpu)
            } else {
                edge.value_sum / f64::from(edge.visits)
            };
            let score = q + scale * f64::from(edge.prior) / f64::from(edge.visits + 1);
            // Ties use ascending canonical action order. Root noise/symmetries are explicit actor choices.
            if score > best_score {
                best = i;
                best_score = score;
            }
        }
        best
    }

    fn backup(&mut self, mut value: f32) {
        for &(parent, index) in self.path.iter().rev() {
            value = -value;
            let edge = &mut self.edges[index];
            edge.visits += 1;
            edge.value_sum += f64::from(value);
            let node = &mut self.nodes[parent as usize];
            node.visits += 1;
            node.value_sum += f64::from(value);
            assert!(
                self.position.undo(),
                "position must undo each successful simulated action"
            );
        }
        self.path.clear();
        self.completed += 1;
    }

    /// Performs native work until one leaf needs inference or the exact budget is
    /// complete. Repeated polling while pending returns the same request unchanged.
    pub fn advance(&mut self) -> Result<Progress, Error> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let progress = self.advance_inner();
        if let Err(error) = progress {
            while self.path.pop().is_some() {
                assert!(self.position.undo());
            }
            self.pending = None;
            self.failure = Some(error);
        }
        progress
    }

    fn advance_inner(&mut self) -> Result<Progress, Error> {
        if let Some(pending) = &self.pending {
            return Ok(Progress::NeedsEvaluation(pending.request));
        }
        if self.nodes[0].terminal {
            return Ok(Progress::Complete);
        }
        if !self.nodes[0].expanded {
            return self.request(0);
        }
        while self.completed < self.config.simulations {
            assert!(self.path.is_empty());
            let mut node = 0;
            loop {
                if self.nodes[node as usize].terminal {
                    let value = self.nodes[node as usize].network_value;
                    self.terminal_evaluations += 1;
                    self.backup(value);
                    break;
                }
                if !self.nodes[node as usize].expanded {
                    return self.request(node);
                }
                let edge_index = self.select(node);
                let action = self.edges[edge_index].action as usize;
                if !self.position.apply(action) {
                    return Err(Error::InvalidPosition);
                }
                self.path.push((node, edge_index));
                let child = self.edges[edge_index].child;
                if child == UNSET {
                    let terminal = self.position.terminal_value();
                    if terminal.is_some_and(|v| !v.is_finite() || !(-1.0..=1.0).contains(&v)) {
                        return Err(Error::InvalidPosition);
                    }
                    let index = self.nodes.len() as u32;
                    self.nodes.push(Node::new(terminal));
                    self.edges[edge_index].child = index;
                    node = index;
                } else {
                    node = child;
                }
            }
        }
        Ok(Progress::Complete)
    }

    /// A malformed, duplicate, stale-search, or stale-network response changes no
    /// tree statistics. A valid retry may complete the outstanding request.
    pub fn complete(&mut self, request: Request, logits: &[f32], value: f32) -> Result<(), Error> {
        let pending = self.pending.as_ref().ok_or(Error::NoRequest)?;
        if pending.request != request {
            return Err(Error::WrongRequest);
        }
        if logits.len() != self.actions || logits.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidPolicy);
        }
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(Error::InvalidValue);
        }
        let maximum = pending
            .legal
            .iter()
            .map(|&a| logits[a])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f64> = pending
            .legal
            .iter()
            .map(|&a| (f64::from(logits[a]) - f64::from(maximum)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        for p in &mut weights {
            *p /= total;
        }
        if pending.node == 0
            && let Some(noise) = &self.noise
            && noise.fraction > 0.0
        {
            let sum: f64 = pending
                .legal
                .iter()
                .map(|&a| f64::from(noise.weights[a]))
                .sum();
            if sum <= 0.0 {
                return Err(Error::InvalidNoise);
            }
            for (i, &a) in pending.legal.iter().enumerate() {
                weights[i] = weights[i] * f64::from(1.0 - noise.fraction)
                    + f64::from(noise.fraction) * f64::from(noise.weights[a]) / sum;
            }
        }
        let pending = self.pending.take().unwrap();
        let node = &mut self.nodes[pending.node as usize];
        node.first_edge = self.edges.len();
        node.edge_count = pending.legal.len() as u32;
        node.expanded = true;
        node.network_value = value;
        if pending.node == 0
            && let Some(gumbel) = &mut self.gumbel
        {
            gumbel.prepare(pending.legal.len(), self.config.simulations);
        }
        for (a, prior) in pending.legal.into_iter().zip(weights) {
            self.edges.push(Edge {
                action: a as u32,
                prior: prior as f32,
                log_prior: f64::from(logits[a]) - f64::from(maximum),
                child: UNSET,
                visits: 0,
                value_sum: 0.0,
            });
        }
        self.neural_evaluations += 1;
        if pending.node != 0 {
            self.backup(value);
        }
        Ok(())
    }

    pub fn stats(&self) -> Result<RootStats, Error> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let root = &self.nodes[0];
        if self.pending.is_some()
            || !root.expanded
            || (!root.terminal && self.completed != self.config.simulations)
        {
            return Err(Error::NotComplete);
        }
        let mut visits = vec![0; self.actions];
        let mut priors = vec![0.0; self.actions];
        let mut values = vec![None; self.actions];
        for edge in &self.edges[root.first_edge..root.first_edge + root.edge_count as usize] {
            let a = edge.action as usize;
            visits[a] = edge.visits;
            priors[a] = edge.prior;
            if edge.visits > 0 {
                values[a] = Some((edge.value_sum / f64::from(edge.visits)) as f32);
            }
        }
        let (improved_policy, recommended_action) = if let Some(gumbel) = &self.gumbel
            && root.edge_count > 0
        {
            let (policy, action) = gumbel.result(
                root,
                &self.edges[root.first_edge..root.first_edge + root.edge_count as usize],
                self.actions,
            );
            (Some(policy), Some(action))
        } else {
            (None, None)
        };
        Ok(RootStats {
            completed_simulations: self.completed,
            neural_evaluations: self.neural_evaluations,
            terminal_evaluations: self.terminal_evaluations,
            visits,
            priors,
            action_values: values,
            value: if root.visits > 0 {
                (root.value_sum / f64::from(root.visits)) as f32
            } else {
                root.network_value
            },
            nodes: self.nodes.len(),
            edges: self.edges.len(),
            improved_policy,
            recommended_action,
        })
    }

    pub fn inspect_root(&self) -> Result<RootInspection, Error> {
        let stats = self.stats()?;
        let root = &self.nodes[0];
        let mut log_priors = vec![None; self.actions];
        let mut value_sums = vec![None; self.actions];
        let mut terminal_child_values = vec![None; self.actions];
        for edge in &self.edges[root.first_edge..root.first_edge + root.edge_count as usize] {
            let action = edge.action as usize;
            log_priors[action] = Some(edge.log_prior);
            value_sums[action] = Some(edge.value_sum);
            if let Some(child) = self.nodes.get(edge.child as usize) {
                if child.terminal {
                    // Child values use the child player's perspective.
                    terminal_child_values[action] = Some(child.network_value);
                }
            }
        }
        Ok(RootInspection {
            stats,
            network_value: root.network_value,
            visits: root.visits,
            log_priors,
            value_sums,
            terminal_child_values,
        })
    }

    /// Consumes a complete search and returns its exact original root position.
    pub fn into_position(self) -> Result<P, Error> {
        self.stats()?;
        Ok(self.position)
    }

    /// Cancels pending work and restores the original root. Partial targets are
    /// deliberately unavailable; restart with a fresh search identity if needed.
    pub fn abort(mut self) -> P {
        while self.path.pop().is_some() {
            assert!(self.position.undo());
        }
        self.position
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        for node in &self.nodes {
            let edges = &self.edges[node.first_edge..node.first_edge + node.edge_count as usize];
            if edges.iter().map(|e| e.visits).sum::<u32>() != node.visits {
                return Err("visit accounting");
            }
            let sum: f64 = edges.iter().map(|e| e.value_sum).sum();
            if (sum - node.value_sum).abs() > 1e-6 {
                return Err("backup accounting");
            }
            if node.visits > self.completed {
                return Err("incomplete work counted as visits");
            }
        }
        if self.nodes[0].visits != self.completed {
            return Err("root simulation accounting");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
