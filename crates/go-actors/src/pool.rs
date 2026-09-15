use crate::{
    Batch, Config, GameRecord, PoolState,
    actor::{Actor, ActorState},
};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

enum Command {
    Start(u64),
    Evaluate(Vec<f32>, Vec<f32>),
    Prefetch(usize),
    EvaluatePrefetch(usize, Vec<f32>, Vec<f32>),
    PrefetchCounters,
    Commit,
    Checkpoint,
    Stop,
}
enum Reply {
    Initialized,
    Batch(Vec<f32>, Vec<u8>),
    Committed(Vec<GameRecord>),
    Checkpoint(Vec<ActorState>),
    PrefetchCounters(u64, u64),
    Failed(String),
}
struct Worker {
    start: usize,
    end: usize,
    tx: SyncSender<Command>,
    rx: Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
}

fn batch(actors: &[Actor], width: usize) -> Reply {
    let mut features = Vec::with_capacity(actors.len() * width);
    let mut active = Vec::with_capacity(actors.len());
    for actor in actors {
        if let Some(row) = actor.features() {
            features.extend(row);
            active.push(1);
        } else {
            features.resize(features.len() + width, 0.0);
            active.push(0);
        }
    }
    Reply::Batch(features, active)
}

fn worker(
    config: Config,
    index: usize,
    start: usize,
    end: usize,
    states: Option<Vec<ActorState>>,
    rx: Receiver<Command>,
    tx: SyncSender<Reply>,
) {
    let initialize = || -> Result<Vec<Actor>, String> {
        if !config.worker_cpus.is_empty() {
            let cpu = config.worker_cpus[index];
            let allowed = core_affinity::get_core_ids().ok_or("cannot inspect CPU affinity")?;
            if !allowed.iter().any(|id| id.id == cpu)
                || !core_affinity::set_for_current(core_affinity::CoreId { id: cpu })
            {
                return Err(format!("cannot pin worker {index} to allowed CPU {cpu}"));
            }
        }
        if let Some(states) = states {
            states
                .into_iter()
                .enumerate()
                .map(|(offset, state)| {
                    Actor::restore(
                        config.actor_offset + (start + offset) as u64,
                        config.clone(),
                        state,
                    )
                })
                .collect()
        } else {
            (start..end)
                .map(|slot| Actor::new(config.actor_offset + slot as u64, config.clone()))
                .collect()
        }
    };
    let mut actors = match initialize() {
        Ok(actors) => actors,
        Err(error) => {
            let _ = tx.send(Reply::Failed(error));
            return;
        }
    };
    if tx.send(Reply::Initialized).is_err() {
        return;
    }
    while let Ok(command) = rx.recv() {
        if matches!(command, Command::Stop) {
            break;
        }
        let result = (|| -> Result<Reply, String> {
            match command {
                Command::Start(network) => {
                    for actor in &mut actors {
                        actor.start(network)?;
                    }
                    Ok(batch(&actors, config.feature_width()))
                }
                Command::Evaluate(policies, values) => {
                    for (i, actor) in actors.iter_mut().enumerate() {
                        actor.evaluate(
                            &policies[i * config.actions()..(i + 1) * config.actions()],
                            values[i],
                        )?;
                    }
                    Ok(batch(&actors, config.feature_width()))
                }
                Command::Commit => {
                    let mut games = Vec::new();
                    for actor in &mut actors {
                        if let Some(game) = actor.commit()? {
                            games.push(game);
                        }
                    }
                    Ok(Reply::Committed(games))
                }
                Command::Prefetch(limit) => {
                    let mut features = Vec::new();
                    let mut active = Vec::new();
                    for actor in &mut actors {
                        let (rows, mask) = actor.prefetch_features(limit)?;
                        features.extend(rows);
                        active.extend(mask);
                    }
                    Ok(Reply::Batch(features, active))
                }
                Command::EvaluatePrefetch(limit, policies, values) => {
                    for (i, actor) in actors.iter_mut().enumerate() {
                        actor.evaluate_prefetch(
                            &policies
                                [i * limit * config.actions()..(i + 1) * limit * config.actions()],
                            &values[i * limit..(i + 1) * limit],
                        )?;
                    }
                    Ok(batch(&actors, config.feature_width()))
                }
                Command::PrefetchCounters => Ok(Reply::PrefetchCounters(
                    actors.iter().map(|a| a.prefetch_evaluations).sum(),
                    actors.iter().map(|a| a.prefetch_hits).sum(),
                )),
                Command::Checkpoint => Ok(Reply::Checkpoint(
                    actors
                        .iter()
                        .map(Actor::checkpoint)
                        .collect::<Result<_, _>>()?,
                )),
                Command::Stop => unreachable!(),
            }
        })();
        match result {
            Ok(reply) => {
                if tx.send(reply).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = tx.send(Reply::Failed(error));
                break;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Searching,
    Prefetching(usize),
    Ready,
    Failed,
    Closed,
}

/// Each long-lived thread owns several games and all mutable search statistics.
/// Commands/completions are bounded, coarse per-worker messages. No Python runs
/// during move generation, search, feature encoding, or outcome labeling.
pub struct Pool {
    config: Config,
    workers: Vec<Worker>,
    phase: Phase,
    round: u64,
    network: u64,
    move_prefetched: bool,
}

impl Pool {
    pub fn new(config: Config) -> Result<Self, String> {
        Self::initialize(config, None)
    }
    pub fn restore(config: Config, state: PoolState) -> Result<Self, String> {
        Self::initialize(config, Some(state))
    }
    fn initialize(config: Config, state: Option<PoolState>) -> Result<Self, String> {
        config.validate()?;
        if let Some(state) = &state {
            let mut expected = state.config.clone();
            expected.workers = config.workers;
            expected.worker_cpus = config.worker_cpus.clone();
            if state.schema_version != 1 || expected != config || state.actors.len() != config.games
            {
                return Err("actor checkpoint schema or scientific configuration differs".into());
            }
        }
        let mut pool = Self {
            config: config.clone(),
            workers: Vec::new(),
            phase: Phase::Idle,
            round: state.as_ref().map_or(0, |s| s.round),
            network: state.as_ref().map_or(0, |s| s.network),
            move_prefetched: false,
        };
        for index in 0..config.workers {
            let start = index * config.games / config.workers;
            let end = (index + 1) * config.games / config.workers;
            let (tx, command_rx) = sync_channel(1);
            let (reply_tx, rx) = sync_channel(1);
            let copy = config.clone();
            let states = state.as_ref().map(|s| s.actors[start..end].to_vec());
            let handle = thread::Builder::new()
                .name(format!("go-actor-{index}"))
                .spawn(move || worker(copy, index, start, end, states, command_rx, reply_tx))
                .map_err(|e| e.to_string())?;
            pool.workers.push(Worker {
                start,
                end,
                tx,
                rx,
                thread: Some(handle),
            });
        }
        for reply in pool.receive()? {
            if !matches!(reply, Reply::Initialized) {
                return Err("unexpected worker initialization response".into());
            }
        }
        Ok(pool)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn checkpoint(&mut self) -> Result<PoolState, String> {
        if self.phase != Phase::Idle {
            return Err("pool checkpoint requires a committed move boundary".into());
        }
        let replies = self.send(
            (0..self.workers.len())
                .map(|_| Command::Checkpoint)
                .collect(),
        )?;
        let mut actors = Vec::with_capacity(self.config.games);
        for reply in replies {
            match reply {
                Reply::Checkpoint(states) => actors.extend(states),
                _ => {
                    self.phase = Phase::Failed;
                    return Err("unexpected checkpoint response".into());
                }
            }
        }
        Ok(PoolState {
            schema_version: 1,
            config: self.config.clone(),
            round: self.round,
            network: self.network,
            actors,
        })
    }
    fn receive(&mut self) -> Result<Vec<Reply>, String> {
        // Drain every worker, including after an error, before allowing cleanup.
        let responses: Vec<_> = self.workers.iter().map(|w| w.rx.recv()).collect();
        let mut output = Vec::new();
        let mut error = None;
        for response in responses {
            match response {
                Ok(Reply::Failed(message)) => {
                    error.get_or_insert(message);
                }
                Ok(reply) => output.push(reply),
                Err(e) => {
                    error.get_or_insert(format!("actor worker disconnected: {e}"));
                }
            }
        }
        if let Some(error) = error {
            self.phase = Phase::Failed;
            Err(error)
        } else {
            Ok(output)
        }
    }
    fn send(&mut self, commands: Vec<Command>) -> Result<Vec<Reply>, String> {
        let mut failed = false;
        for (worker, command) in self.workers.iter().zip(commands) {
            failed |= worker.tx.send(command).is_err();
        }
        let replies = self.receive();
        if failed {
            self.phase = Phase::Failed;
            return Err("actor command channel closed".into());
        }
        replies
    }
    fn batch(&mut self, replies: Vec<Reply>) -> Result<Batch, String> {
        self.sized_batch(replies, 1, false)
    }
    fn sized_batch(
        &mut self,
        replies: Vec<Reply>,
        limit: usize,
        prefetch: bool,
    ) -> Result<Batch, String> {
        let mut features =
            Vec::with_capacity(self.config.games * limit * self.config.feature_width());
        let mut active = Vec::with_capacity(self.config.games * limit);
        for reply in replies {
            match reply {
                Reply::Batch(values, mask) => {
                    features.extend(values);
                    active.extend(mask);
                }
                _ => {
                    self.phase = Phase::Failed;
                    return Err("unexpected actor batch response".into());
                }
            }
        }
        if active.len() != self.config.games * limit
            || features.len() != self.config.games * limit * self.config.feature_width()
        {
            self.phase = Phase::Failed;
            return Err("actor batch shape mismatch".into());
        }
        self.round = self
            .round
            .checked_add(1)
            .ok_or("round identity exhausted")?;
        self.phase = if prefetch {
            Phase::Prefetching(limit)
        } else if active.iter().any(|v| *v != 0) {
            Phase::Searching
        } else {
            Phase::Ready
        };
        Ok(Batch {
            round: self.round,
            network: self.network,
            features,
            active,
        })
    }
    pub fn start(&mut self, network: u64) -> Result<Batch, String> {
        if self.phase != Phase::Idle {
            return Err("start requires an idle actor pool".into());
        }
        self.network = network;
        self.move_prefetched = false;
        let replies = self.send(
            (0..self.workers.len())
                .map(|_| Command::Start(network))
                .collect(),
        )?;
        self.batch(replies)
    }
    /// All slots have a fixed position in the dense batch. Inactive slots are
    /// padding and their predictions are never counted or used for search.
    pub fn evaluate(
        &mut self,
        round: u64,
        network: u64,
        policies: &[f32],
        values: &[f32],
    ) -> Result<Batch, String> {
        if !matches!(self.phase, Phase::Searching | Phase::Ready) {
            return Err("no inference round is active".into());
        }
        if round != self.round || network != self.network {
            return Err("stale inference round or network".into());
        }
        if policies.len() != self.config.games * self.config.actions()
            || values.len() != self.config.games
            || policies.iter().any(|x| !x.is_finite())
            || values
                .iter()
                .any(|x| !x.is_finite() || !(-1.0..=1.0).contains(x))
        {
            return Err("invalid neural output shape or values".into());
        }
        let commands = self
            .workers
            .iter()
            .map(|w| {
                Command::Evaluate(
                    policies[w.start * self.config.actions()..w.end * self.config.actions()]
                        .to_vec(),
                    values[w.start..w.end].to_vec(),
                )
            })
            .collect();
        let replies = self.send(commands)?;
        self.batch(replies)
    }
    /// Prepare root-child features without changing search visits. The new
    /// round invalidates the previous ordinary batch until prefetch completes.
    pub fn prefetch(&mut self, round: u64, network: u64, limit: usize) -> Result<Batch, String> {
        if self.phase != Phase::Searching
            || self.move_prefetched
            || self.config.gumbel.is_none()
            || round != self.round
            || network != self.network
            || !(1..=64).contains(&limit)
        {
            return Err("invalid root prefetch phase, identity, or limit".into());
        }
        if self
            .config
            .games
            .checked_mul(limit)
            .and_then(|n| n.checked_mul(self.config.feature_width()))
            .is_none_or(|floats| floats > 268_435_456)
        {
            return Err("root prefetch exceeds one GiB of feature storage".into());
        }
        let replies = self.send(
            (0..self.workers.len())
                .map(|_| Command::Prefetch(limit))
                .collect(),
        )?;
        self.move_prefetched = true;
        self.sized_batch(replies, limit, true)
    }
    pub fn evaluate_prefetch(
        &mut self,
        round: u64,
        network: u64,
        policies: &[f32],
        values: &[f32],
    ) -> Result<Batch, String> {
        let Phase::Prefetching(limit) = self.phase else {
            return Err("no root prefetch round is active".into());
        };
        if round != self.round || network != self.network {
            return Err("stale root prefetch round or network".into());
        }
        if policies.len() != self.config.games * limit * self.config.actions()
            || values.len() != self.config.games * limit
            || policies.iter().any(|x| !x.is_finite())
            || values
                .iter()
                .any(|x| !x.is_finite() || !(-1.0..=1.0).contains(x))
        {
            return Err("invalid root prefetch neural outputs".into());
        }
        let commands = self
            .workers
            .iter()
            .map(|w| {
                Command::EvaluatePrefetch(
                    limit,
                    policies[w.start * limit * self.config.actions()
                        ..w.end * limit * self.config.actions()]
                        .to_vec(),
                    values[w.start * limit..w.end * limit].to_vec(),
                )
            })
            .collect();
        let replies = self.send(commands)?;
        self.batch(replies)
    }
    /// Lifetime counters for this process. Checkpoints intentionally retain
    /// scientific state only; the caller carries aggregate systems counters.
    pub fn prefetch_counters(&mut self) -> Result<(u64, u64), String> {
        if matches!(self.phase, Phase::Failed | Phase::Closed) {
            return Err("actor pool unavailable".into());
        }
        let replies = self.send(
            (0..self.workers.len())
                .map(|_| Command::PrefetchCounters)
                .collect(),
        )?;
        let mut total = (0, 0);
        for reply in replies {
            match reply {
                Reply::PrefetchCounters(evaluated, used) => {
                    total.0 += evaluated;
                    total.1 += used;
                }
                _ => {
                    self.phase = Phase::Failed;
                    return Err("unexpected root prefetch counters".into());
                }
            }
        }
        Ok(total)
    }
    /// Advances each real game exactly once. Completed games carry terminal
    /// labels; move-limit truncations retain records but expose no training rows.
    pub fn commit(&mut self) -> Result<Vec<GameRecord>, String> {
        if self.phase != Phase::Ready {
            return Err("commit requires completed searches on all actors".into());
        }
        let replies = self.send((0..self.workers.len()).map(|_| Command::Commit).collect())?;
        let mut games = Vec::new();
        for reply in replies {
            match reply {
                Reply::Committed(records) => games.extend(records),
                _ => {
                    self.phase = Phase::Failed;
                    return Err("unexpected commit response".into());
                }
            }
        }
        self.phase = Phase::Idle;
        Ok(games)
    }
    pub fn close(&mut self) {
        if self.phase == Phase::Closed {
            return;
        }
        for worker in &self.workers {
            let _ = worker.tx.send(Command::Stop);
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.thread.take() {
                let _ = handle.join();
            }
        }
        self.phase = Phase::Closed;
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        self.close();
    }
}
