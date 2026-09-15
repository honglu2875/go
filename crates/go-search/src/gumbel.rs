// Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
// Copyright 2026 GoZero contributors.
// Licensed under the Apache License, Version 2.0. See ../MCTX_LICENSE.md.
// Rust adaptation of the published Gumbel AlphaZero selection/completion rules
// and Mctx's sequential-halving schedule; source revision is in that notice.

use crate::{Edge, Node};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelConfig {
    pub max_considered_actions: usize,
    pub value_scale: f32,
    pub maxvisit_init: f32,
    pub rescale_values: bool,
}
impl GumbelConfig {
    pub fn is_valid(self) -> bool {
        (1..=4096).contains(&self.max_considered_actions)
            && self.value_scale.is_finite()
            && (0.0001..=100.0).contains(&self.value_scale)
            && self.maxvisit_init.is_finite()
            && (0.0..=1_000_000.0).contains(&self.maxvisit_init)
    }
}

/// Visit strata for sequential halving, including odd action counts and budgets.
/// The caller caps `considered` by both legal actions and the simulation budget.
fn schedule(considered: usize, budget: u32) -> Vec<u32> {
    if considered <= 1 {
        return (0..budget).collect();
    }
    let rounds = considered.next_power_of_two().trailing_zeros() as usize;
    let mut visits = vec![0; considered];
    let mut active = considered;
    let mut sequence = Vec::with_capacity(budget as usize);
    while sequence.len() < budget as usize {
        let extra = (budget as usize / (rounds * active)).max(1);
        for _ in 0..extra {
            for count in &mut visits[..active] {
                if sequence.len() == budget as usize {
                    return sequence;
                }
                sequence.push(*count);
                *count += 1;
            }
        }
        active = (active / 2).max(2);
    }
    sequence
}

pub(crate) struct GumbelState {
    pub config: GumbelConfig,
    draws: Vec<f32>,
    schedule: Vec<u32>,
}
impl GumbelState {
    pub fn new(config: GumbelConfig, draws: Vec<f32>) -> Self {
        Self {
            config,
            draws,
            schedule: Vec::new(),
        }
    }
    pub fn prepare(&mut self, legal: usize, budget: u32) {
        let considered = legal
            .min(self.config.max_considered_actions)
            .min(budget as usize);
        self.schedule = schedule(considered, budget);
    }
    fn completed_logits(&self, node: &Node, edges: &[Edge], actions: usize) -> Vec<f64> {
        let mut mass = 0.0;
        let mut weighted = 0.0;
        for edge in edges.iter().filter(|edge| edge.visits > 0) {
            // Match the reference's tiny-probability guard for extreme logits.
            let prior = f64::from(edge.prior.max(f32::MIN_POSITIVE));
            mass += prior;
            weighted += prior * edge.value_sum / f64::from(edge.visits);
        }
        let mixed = (f64::from(node.network_value)
            + f64::from(node.visits) * if mass > 0.0 { weighted / mass } else { 0.0 })
            / f64::from(node.visits + 1);
        let values: Vec<f64> = edges
            .iter()
            .map(|edge| {
                if edge.visits > 0 {
                    edge.value_sum / f64::from(edge.visits)
                } else {
                    mixed
                }
            })
            .collect();
        let mut low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let mut high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Mctx completes the full action tensor, including masked actions.
        // They affect this numerical rescaling but never receive policy mass.
        if edges.len() < actions {
            low = low.min(mixed);
            high = high.max(mixed);
        }
        let largest = edges.iter().map(|edge| edge.visits).max().unwrap_or(0);
        let scale = (f64::from(self.config.maxvisit_init) + f64::from(largest))
            * f64::from(self.config.value_scale);
        values
            .iter()
            .zip(edges)
            .map(|(&value, edge)| {
                let q = if self.config.rescale_values {
                    (value - low) / (high - low).max(1e-8)
                } else {
                    value
                };
                edge.log_prior + scale * q
            })
            .collect()
    }
    fn root_choice(&self, edges: &[Edge], logits: &[f64], visits: u32) -> usize {
        let mut best = None;
        let mut score = f64::NEG_INFINITY;
        for (i, (edge, &logit)) in edges.iter().zip(logits).enumerate() {
            if edge.visits == visits {
                let candidate = (logit + f64::from(self.draws[edge.action as usize])).max(-1e9);
                if candidate > score {
                    best = Some(i);
                    score = candidate;
                }
            }
        }
        best.expect("sequential halving always has a legal action in the scheduled visit stratum")
    }
    pub fn select(&self, root: bool, node: &Node, edges: &[Edge], actions: usize) -> usize {
        let logits = self.completed_logits(node, edges, actions);
        if root {
            return self.root_choice(edges, &logits, self.schedule[node.visits as usize]);
        }
        let policy = softmax(&logits);
        let mut best = 0;
        let mut score = f64::NEG_INFINITY;
        for (i, (edge, &prob)) in edges.iter().zip(&policy).enumerate() {
            let candidate = prob - f64::from(edge.visits) / f64::from(node.visits + 1);
            if candidate > score {
                best = i;
                score = candidate;
            }
        }
        best
    }
    /// Predict the remaining distinct children in the initial visit stratum.
    /// These are cache proposals only. Later value-dependent floating-point
    /// rounding and the score floor can change tie ordering; selection must
    /// still run through `select` and use a proposal only on an actual hit.
    pub fn root_sweep_candidates(
        &self,
        node: &Node,
        edges: &[Edge],
        actions: usize,
        limit: usize,
    ) -> Vec<usize> {
        let remaining = self.schedule[node.visits as usize..]
            .iter()
            .take_while(|&&visits| visits == 0)
            .count()
            .min(limit);
        let logits = self.completed_logits(node, edges, actions);
        let mut proposed = edges.to_vec();
        (0..remaining)
            .map(|_| {
                let index = self.root_choice(&proposed, &logits, 0);
                proposed[index].visits = 1;
                proposed[index].action as usize
            })
            .collect()
    }
    pub fn result(&self, node: &Node, edges: &[Edge], actions: usize) -> (Vec<f32>, usize) {
        let logits = self.completed_logits(node, edges, actions);
        let considered = edges.iter().map(|e| e.visits).max().unwrap();
        let chosen = edges[self.root_choice(edges, &logits, considered)].action as usize;
        let mut policy = vec![0.; actions];
        for (edge, weight) in edges.iter().zip(softmax(&logits)) {
            policy[edge.action as usize] = weight as f32;
        }
        (policy, chosen)
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    for value in &mut weights {
        *value /= total;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn odd_budgets_and_action_counts_keep_a_nonempty_stratum() {
        for budget in 1..130 {
            for actions in 1..40 {
                let mut visits = vec![0; actions];
                for count in schedule(actions.min(budget as usize), budget) {
                    // Arbitrary changing scores may choose any eligible action.
                    let selected = visits.iter().rposition(|&n| n == count).unwrap();
                    visits[selected] += 1;
                }
                assert_eq!(visits.iter().sum::<u32>(), budget);
            }
        }
    }
}
