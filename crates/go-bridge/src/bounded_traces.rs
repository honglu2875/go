//! Equal-work trace batches: inactive games retain their full native state.
use go_actors::traces::{Config, Packet, Resolution, StopReason, Ticket, TraceGame};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray3, PyReadonlyArray4,
            PyReadonlyArray5, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::Mutex;

struct Batch {
    games: Vec<TraceGame>,
    config: Config,
    pending: Option<Vec<Option<Ticket>>>,
}

#[pyclass]
pub struct BoundedTraceActors {
    inner: Mutex<Batch>,
}

type Frame<'py> = (String, Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<i32>>,
                  Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<u8>>);

#[pymethods]
impl BoundedTraceActors {
    #[new]
    fn new(config_json: &str, games: usize) -> PyResult<Self> {
        let config: Config = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if !(1..=4096).contains(&games) || !(1..=1024).contains(&config.size)
            || config.max_game_moves > 1_000_000
            || games.checked_mul(config.max_game_moves + 65).is_none_or(|n| n > 1 << 26)
            || games.checked_mul((config.size + 2) * (config.size + 2)).is_none_or(|n| n > 1 << 24)
        {
            return Err(PyValueError::new_err("bounded trace batch exceeds resource limit"));
        }
        let items = (0..games).map(|_| TraceGame::new(config.clone()))
            .collect::<Result<Vec<_>, _>>().map_err(PyValueError::new_err)?;
        Ok(Self { inner: Mutex::new(Batch { games: items, config, pending: None }) })
    }

    fn start<'py>(&self, py: Python<'py>, models: [u64; 2], contexts: [u64; 2],
                  horizon: usize, active: Vec<bool>) -> PyResult<Frame<'py>> {
        if !(1..=64).contains(&horizon) {
            return Err(PyValueError::new_err("invalid horizon"));
        }
        let (tickets, tokens, lengths, legal, stones, batch, width, actions) = py.detach(|| {
            let mut state = self.inner.lock().map_err(|e| e.to_string())?;
            let batch = state.games.len();
            if state.pending.is_some() || active.len() != batch {
                return Err("batch already pending or active mask dimensions differ".into());
            }
            let actions = state.config.size * state.config.size + 1;
            let width = state.config.max_game_moves + horizon + 1;
            let mut tokens = vec![(actions + 1) as i32; batch * width];
            let mut lengths = vec![0i32; batch];
            let mut legal = vec![false; batch * actions];
            let mut stones = vec![0u8; batch * (actions - 1)];
            let mut tickets = Vec::with_capacity(batch);
            for (index, game) in state.games.iter_mut().enumerate() {
                tokens[index * width] = actions as i32;
                legal[(index + 1) * actions - 1] = true;
                if !active[index] {
                    // No recycle, generation increment, or game mutation. The
                    // device receives an explicit dummy row whose work counts
                    // as padding and whose predictions can never be consumed.
                    tickets.push(None);
                    continue;
                }
                game.recycle_finished()?;
                tickets.push(Some(game.start(models, contexts)?));
                legal[index * actions..(index + 1) * actions].copy_from_slice(&game.legal_mask());
                stones[index * (actions - 1)..(index + 1) * (actions - 1)]
                    .copy_from_slice(&game.position().board.stones());
                for (ply, &action) in game.moves().iter().enumerate() {
                    tokens[index * width + ply + 1] = action as i32;
                }
                lengths[index] = game.moves().len() as i32;
            }
            let encoded = serde_json::to_string(&tickets).map_err(|e| e.to_string())?;
            state.pending = Some(tickets);
            Ok::<_, String>((encoded, tokens, lengths, legal, stones, batch, width, actions))
        }).map_err(PyValueError::new_err)?;
        Ok((tickets,
            numpy::ndarray::Array2::from_shape_vec((batch, width), tokens).unwrap().into_pyarray(py),
            lengths.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((batch, actions), legal).unwrap().into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((batch, actions - 1), stones).unwrap().into_pyarray(py)))
    }

    fn resolve(&self, py: Python<'_>, tickets_json: &str,
               proposals: PyReadonlyArray4<'_, i32>, own_logits: PyReadonlyArray5<'_, f32>,
               noise: PyReadonlyArray3<'_, f32>, states: PyReadonlyArray5<'_, u8>,
               allowances: Vec<usize>) -> PyResult<String> {
        let tickets: Vec<Option<Ticket>> = serde_json::from_str(tickets_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (batch, actions) = {
            let state = self.inner.lock().map_err(|e| PyValueError::new_err(e.to_string()))?;
            (state.games.len(), state.config.size * state.config.size + 1)
        };
        let shape = proposals.shape(); let samples = shape[2]; let horizon = shape[3];
        if !(1..=64).contains(&samples) || !(1..=64).contains(&horizon)
            || shape != [batch, 2, samples, horizon] || tickets.len() != batch
            || own_logits.shape() != [batch, 2, samples, horizon, actions]
            || noise.shape() != [batch, horizon, actions]
            || states.shape() != [batch, 2, samples, horizon, actions - 1]
            || allowances.len() != batch
            || allowances.iter().zip(&tickets).any(|(&n, t)| n > horizon || (n > 0) != t.is_some())
            || batch.checked_mul(2 * samples * horizon * actions).is_none_or(|n| n > 1 << 26)
        {
            return Err(PyValueError::new_err("bounded trace dimensions, active mask, or work allowance differ"));
        }
        let proposals = proposals.as_slice()?.to_vec();
        let logits = own_logits.as_slice()?.to_vec(); let noise = noise.as_slice()?.to_vec();
        let states = states.as_slice()?.to_vec();
        py.detach(|| {
            let mut state = self.inner.lock().map_err(|e| e.to_string())?;
            if state.pending.as_ref() != Some(&tickets) {
                return Err("stale bounded batch ticket or active mask".into());
            }
            let rows = 2 * samples * horizon;
            let packet = |i: usize| Packet { samples, horizon,
                actions: &proposals[i * rows..(i + 1) * rows],
                own_logits: &logits[i * rows * actions..(i + 1) * rows * actions],
                noise: &noise[i * horizon * actions..(i + 1) * horizon * actions] };
            let board_inputs = |i: usize| &states[i * rows * (actions - 1)..(i + 1) * rows * (actions - 1)];
            // Validate all active rows and limits before advancing any game.
            for (i, game) in state.games.iter().enumerate() {
                if let Some(ticket) = &tickets[i] {
                    game.validate_state_packet(ticket, &packet(i), board_inputs(i))?;
                }
            }
            let mut resolved = Vec::with_capacity(batch);
            for (i, game) in state.games.iter_mut().enumerate() {
                resolved.push(if let Some(ticket) = &tickets[i] {
                    game.resolve_with_states_limited(ticket, packet(i), board_inputs(i), allowances[i])?
                } else {
                    Resolution { actions: Vec::new(), selected_samples: Vec::new(), legality_corrections: 0,
                                 stop: StopReason::WorkLimit, terminal_white_score: None }
                });
            }
            state.pending = None;
            serde_json::to_string(&resolved).map_err(|e| e.to_string())
        }).map_err(PyValueError::new_err)
    }

    fn inspect(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            let state = self.inner.lock().map_err(|e| e.to_string())?;
            if state.pending.is_some() { return Err("cannot inspect a pending batch".into()); }
            let rows: Vec<_> = state.games.iter().map(|g| serde_json::json!({
                "moves": g.moves(), "stones": g.position().board.stones(), "episode": g.episode(),
                "to_play": g.position().board.to_play() as u8, "terminal": g.position().board.is_terminal(),
            })).collect();
            serde_json::to_string(&rows).map_err(|e| e.to_string())
        }).map_err(PyValueError::new_err)
    }

    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut state = self.inner.lock().map_err(|e| e.to_string())?;
            for game in &mut state.games { game.cancel(); }
            state.pending = None;
            Ok::<_, String>(())
        }).map_err(PyValueError::new_err)
    }
}
