//! One coarse call resolves a batch of real games. No Python move loop.
use go_actors::traces::{Config, Packet, Ticket, TraceGame};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArray5,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::Mutex;

struct Batch {
    games: Vec<TraceGame>,
    config: Config,
    pending: bool,
}

#[pyclass]
pub struct DualTraceActors {
    inner: Mutex<Batch>,
}
type Frame<'py> = (
    String,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray2<bool>>,
);
type StateFrame<'py> = (
    String,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray2<u8>>,
);

#[pymethods]
impl DualTraceActors {
    #[new]
    fn new(config_json: &str, games: usize) -> PyResult<Self> {
        let config: Config =
            serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if !(1..=4096).contains(&games)
            || !(1..=1024).contains(&config.size)
            || config.max_game_moves > 1_000_000
            || games
                .checked_mul(config.max_game_moves + 65)
                .is_none_or(|n| n > 1 << 26)
            || games
                .checked_mul((config.size + 2) * (config.size + 2))
                .is_none_or(|n| n > 1 << 24)
        {
            return Err(PyValueError::new_err("trace batch exceeds resource limit"));
        }
        let items = (0..games)
            .map(|_| TraceGame::new(config.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Mutex::new(Batch {
                games: items,
                config,
                pending: false,
            }),
        })
    }
    fn start<'py>(
        &self,
        py: Python<'py>,
        models: [u64; 2],
        contexts: [u64; 2],
        horizon: usize,
    ) -> PyResult<Frame<'py>> {
        let (tickets, tokens, lengths, legal, _) =
            self.start_with_states(py, models, contexts, horizon)?;
        Ok((tickets, tokens, lengths, legal))
    }
    fn start_with_states<'py>(
        &self,
        py: Python<'py>,
        models: [u64; 2],
        contexts: [u64; 2],
        horizon: usize,
    ) -> PyResult<StateFrame<'py>> {
        if !(1..=64).contains(&horizon) {
            return Err(PyValueError::new_err("invalid horizon"));
        }
        let (tickets, tokens, lengths, legal, stones, batch, width, actions) = py
            .detach(|| {
                let mut state = self.inner.lock().map_err(|e| e.to_string())?;
                if state.pending {
                    return Err("trace batch already pending".into());
                }
                let width = state.config.max_game_moves + horizon + 1;
                let actions = state.config.size * state.config.size + 1;
                let batch = state.games.len();
                let mut tokens = vec![(actions + 1) as i32; batch * width];
                let mut lengths = Vec::new();
                let mut tickets = Vec::new();
                let mut legal = Vec::with_capacity(batch * actions);
                let mut stones = Vec::with_capacity(batch * (actions - 1));
                for (index, game) in state.games.iter_mut().enumerate() {
                    game.recycle_finished()?;
                    tickets.push(game.start(models, contexts)?);
                    legal.extend(game.legal_mask());
                    stones.extend(game.position().board.stones());
                    tokens[index * width] = actions as i32;
                    for (ply, &action) in game.moves().iter().enumerate() {
                        tokens[index * width + ply + 1] = action as i32;
                    }
                    lengths.push(game.moves().len() as i32);
                }
                state.pending = true;
                Ok::<_, String>((
                    serde_json::to_string(&tickets).map_err(|e| e.to_string())?,
                    tokens,
                    lengths,
                    legal,
                    stones,
                    batch,
                    width,
                    actions,
                ))
            })
            .map_err(PyValueError::new_err)?;
        let tokens = numpy::ndarray::Array2::from_shape_vec((batch, width), tokens)
            .unwrap()
            .into_pyarray(py);
        let legal = numpy::ndarray::Array2::from_shape_vec((batch, actions), legal)
            .unwrap()
            .into_pyarray(py);
        let stones = numpy::ndarray::Array2::from_shape_vec((batch, actions - 1), stones)
            .unwrap()
            .into_pyarray(py);
        Ok((tickets, tokens, lengths.into_pyarray(py), legal, stones))
    }
    #[pyo3(signature = (tickets_json, proposals, own_logits, noise, states=None))]
    fn resolve(
        &self,
        py: Python<'_>,
        tickets_json: &str,
        proposals: PyReadonlyArray4<'_, i32>,
        own_logits: PyReadonlyArray5<'_, f32>,
        noise: PyReadonlyArray3<'_, f32>,
        states: Option<PyReadonlyArray5<'_, u8>>,
    ) -> PyResult<String> {
        let tickets: Vec<Ticket> =
            serde_json::from_str(tickets_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (batch, actions) = {
            let state = self
                .inner
                .lock()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            (state.games.len(), state.config.size * state.config.size + 1)
        };
        let shape = proposals.shape();
        let samples = shape[2];
        let horizon = shape[3];
        if !(1..=64).contains(&samples)
            || !(1..=64).contains(&horizon)
            || shape != [batch, 2, samples, horizon]
            || tickets.len() != batch
            || own_logits.shape() != [batch, 2, samples, horizon, actions]
            || noise.shape() != [batch, horizon, actions]
            || states.as_ref().is_some_and(|s| s.shape() != [batch, 2, samples, horizon, actions - 1])
            || batch
                .checked_mul(2 * samples * horizon * actions)
                .is_none_or(|n| n > 1 << 26)
        {
            return Err(PyValueError::new_err(
                "trace batch dimensions or size differ",
            ));
        }
        let proposals = proposals.as_slice()?.to_vec();
        let logits = own_logits.as_slice()?.to_vec();
        let noise = noise.as_slice()?.to_vec();
        let states = states.map(|s| s.as_slice().map(|x| x.to_vec())).transpose()?;
        py.detach(|| {
            let mut state = self.inner.lock().map_err(|e| e.to_string())?;
            if !state.pending {
                return Err("no pending trace batch".into());
            }
            let packet = |i: usize| Packet {
                samples,
                horizon,
                actions: &proposals[i * 2 * samples * horizon..(i + 1) * 2 * samples * horizon],
                own_logits: &logits[i * 2 * samples * horizon * actions
                    ..(i + 1) * 2 * samples * horizon * actions],
                noise: &noise[i * horizon * actions..(i + 1) * horizon * actions],
            };
            // Reject a malformed row anywhere before advancing any real game.
            for (i, game) in state.games.iter().enumerate() {
                if let Some(states) = &states {
                    let stride = 2 * samples * horizon * (actions - 1);
                    game.validate_state_packet(&tickets[i], &packet(i), &states[i * stride..(i + 1) * stride])?;
                } else {
                    game.validate_packet(&tickets[i], &packet(i))?;
                }
            }
            let mut resolved = Vec::new();
            for (i, game) in state.games.iter_mut().enumerate() {
                resolved.push(if let Some(states) = &states {
                    let stride = 2 * samples * horizon * (actions - 1);
                    game.resolve_with_states(&tickets[i], packet(i), &states[i * stride..(i + 1) * stride])?
                } else {
                    game.resolve(&tickets[i], packet(i))?
                });
            }
            state.pending = false;
            serde_json::to_string(&resolved).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }
    fn resolve_with_states(
        &self,
        py: Python<'_>,
        tickets_json: &str,
        proposals: PyReadonlyArray4<'_, i32>,
        own_logits: PyReadonlyArray5<'_, f32>,
        noise: PyReadonlyArray3<'_, f32>,
        states: PyReadonlyArray5<'_, u8>,
    ) -> PyResult<String> {
        self.resolve(py, tickets_json, proposals, own_logits, noise, Some(states))
    }
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut state = self.inner.lock().map_err(|e| e.to_string())?;
            for game in &mut state.games {
                game.cancel();
            }
            state.pending = false;
            Ok::<_, String>(())
        })
        .map_err(PyValueError::new_err)
    }
    fn tapes(&self, py: Python<'_>) -> PyResult<Vec<Vec<usize>>> {
        py.detach(|| {
            let state = self.inner.lock().map_err(|e| e.to_string())?;
            Ok::<_, String>(state.games.iter().map(|g| g.moves().to_vec()).collect())
        })
        .map_err(PyValueError::new_err)
    }
}
