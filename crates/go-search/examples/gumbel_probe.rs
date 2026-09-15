//! JSON-lines adapter for an independent Mctx audit of finite game trees.
use go_search::{Config, GumbelConfig, Identity, Position, Progress, Search};
use serde_json::{Value, json};
use std::io::{self, BufRead};

struct Fixture {
    children: Vec<Vec<Option<usize>>>,
    logits: Vec<Vec<f32>>,
    values: Vec<f32>,
    terminal: Vec<bool>,
}
struct Tree<'a> {
    fixture: &'a Fixture,
    path: Vec<usize>,
}
impl Position for Tree<'_> {
    fn action_count(&self) -> usize {
        self.fixture.children[0].len()
    }
    fn legal_actions(&self) -> Vec<usize> {
        self.fixture.children[*self.path.last().unwrap()]
            .iter()
            .enumerate()
            .filter_map(|(a, c)| c.map(|_| a))
            .collect()
    }
    fn apply(&mut self, action: usize) -> bool {
        match self.fixture.children[*self.path.last().unwrap()]
            .get(action)
            .copied()
            .flatten()
        {
            Some(child) => {
                self.path.push(child);
                true
            }
            None => false,
        }
    }
    fn undo(&mut self) -> bool {
        if self.path.len() == 1 {
            false
        } else {
            self.path.pop();
            true
        }
    }
    fn terminal_value(&self) -> Option<f32> {
        let node = *self.path.last().unwrap();
        self.fixture.terminal[node].then_some(self.fixture.values[node])
    }
}

fn run(v: Value) -> Value {
    let children: Vec<Vec<Option<usize>>> = serde_json::from_value(v["children"].clone()).unwrap();
    let logits: Vec<Vec<f32>> = serde_json::from_value(v["logits"].clone()).unwrap();
    let values: Vec<f32> = serde_json::from_value(v["values"].clone()).unwrap();
    let terminal: Vec<bool> = serde_json::from_value(v["terminal"].clone()).unwrap();
    let actions = children[0].len();
    let nodes = children.len();
    assert!(nodes <= 32768 && (1..=128).contains(&actions));
    assert_eq!(
        (logits.len(), values.len(), terminal.len()),
        (nodes, nodes, nodes)
    );
    for (i, row) in children.iter().enumerate() {
        assert_eq!(row.len(), actions);
        assert_eq!(logits[i].len(), actions);
        assert!(
            row.iter()
                .flatten()
                .all(|&child| child > i && child < nodes)
        );
        assert_eq!(terminal[i], row.iter().all(Option::is_none));
        assert!(values[i].is_finite() && (-1.0..=1.0).contains(&values[i]));
    }
    let fixture = Fixture {
        children,
        logits,
        values,
        terminal,
    };
    let cfg = &v["config"];
    let budget = v["budget"].as_u64().unwrap() as u32;
    assert!(budget <= 4096);
    let gumbel = GumbelConfig {
        max_considered_actions: cfg["max_considered_actions"].as_u64().unwrap() as usize,
        value_scale: cfg["value_scale"].as_f64().unwrap() as f32,
        maxvisit_init: cfg["maxvisit_init"].as_f64().unwrap() as f32,
        rescale_values: cfg["rescale_values"].as_bool().unwrap(),
    };
    let draws = serde_json::from_value(v["gumbel"].clone()).unwrap();
    let mut search = Search::with_gumbel(
        Tree {
            fixture: &fixture,
            path: vec![0],
        },
        Config {
            simulations: budget,
            max_edges: 1_000_000,
            ..Config::default()
        },
        Identity {
            search: 1,
            network: 1,
        },
        gumbel,
        draws,
    )
    .unwrap();
    loop {
        match search.advance().unwrap() {
            Progress::Complete => break,
            Progress::NeedsEvaluation(request) => {
                let node = *search.pending_position().unwrap().path.last().unwrap();
                search
                    .complete(request, &fixture.logits[node], fixture.values[node])
                    .unwrap();
            }
        }
        search.validate().unwrap();
    }
    search.validate().unwrap();
    let stats = search.stats().unwrap();
    let output = json!({"visits":stats.visits,"values":stats.action_values,"policy":stats.policy(),
        "action":stats.best_action(),"root_value":stats.value,"completed":stats.completed_simulations});
    assert_eq!(search.into_position().unwrap().path, vec![0]);
    output
}
fn main() {
    for line in io::stdin().lock().lines() {
        let v = serde_json::from_str(&line.unwrap()).unwrap();
        println!("{}", run(v));
    }
}
