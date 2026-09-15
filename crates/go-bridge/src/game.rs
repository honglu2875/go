use go_actors::game::{Game as NativeGame, GameBatch, GameConfig};
use go_search::{Identity, Request};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::sync::Mutex;

type RequestId = (u64, u64, u64, bool);
type PyFrame<'py> = (Option<RequestId>, Bound<'py, PyArray1<f32>>);
type PyStats<'py> = (usize, Bound<'py, PyArray1<f32>>, f32, u32, u32, u32);
type PyState<'py> = (usize, u8, bool, f64, Bound<'py, PyArray1<u8>>);

fn frame(py: Python<'_>, b: GameBatch) -> PyFrame<'_> {
    (
        b.request
            .map(|r| (r.identity.search, r.identity.network, r.sequence, r.is_root)),
        b.features.into_pyarray(py),
    )
}

#[pyclass]
pub struct Game {
    inner: Mutex<NativeGame>,
}

#[pymethods]
impl Game {
    #[new]
    fn new(py: Python<'_>, config_json: &str) -> PyResult<Self> {
        let c: GameConfig =
            serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let game = py
            .detach(|| NativeGame::new(c))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Mutex::new(game),
        })
    }
    fn play(&self, py: Python<'_>, color: u8, action: usize) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .lock()
                .map_err(|e| e.to_string())?
                .play(color, action)
        })
        .map_err(PyValueError::new_err)
    }
    fn start<'py>(&self, py: Python<'py>, network: u64) -> PyResult<PyFrame<'py>> {
        let b = py
            .detach(|| self.inner.lock().map_err(|e| e.to_string())?.start(network))
            .map_err(PyValueError::new_err)?;
        Ok(frame(py, b))
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        request: RequestId,
        logits: PyReadonlyArray1<'py, f32>,
        value: f32,
    ) -> PyResult<PyFrame<'py>> {
        let logits = logits.as_slice()?.to_vec();
        let request = Request {
            identity: Identity {
                search: request.0,
                network: request.1,
            },
            sequence: request.2,
            is_root: request.3,
        };
        let b = py
            .detach(|| {
                self.inner
                    .lock()
                    .map_err(|e| e.to_string())?
                    .evaluate(request, &logits, value)
            })
            .map_err(PyValueError::new_err)?;
        Ok(frame(py, b))
    }
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<PyStats<'py>> {
        let s = py
            .detach(|| self.inner.lock().map_err(|e| e.to_string())?.finish())
            .map_err(PyValueError::new_err)?;
        Ok((
            s.best_action()
                .ok_or_else(|| PyValueError::new_err("no legal action"))?,
            s.policy().into_pyarray(py),
            s.value,
            s.completed_simulations,
            s.neural_evaluations,
            s.terminal_evaluations,
        ))
    }
    /// Completed search only; this neither finishes nor advances the game.
    fn inspect_search(&self, py: Python<'_>) -> PyResult<String> {
        let d = py
            .detach(|| {
                self.inner
                    .lock()
                    .map_err(|e| e.to_string())?
                    .inspect_search()
            })
            .map_err(PyValueError::new_err)?;
        let s = &d.stats;
        serde_json::to_string(&serde_json::json!({
            "schema_version": 1, "chosen": s.best_action(), "policy": s.policy(),
            "root_value": s.value, "network_value": d.network_value, "root_visits": d.visits,
            "completed_simulations": s.completed_simulations, "neural_evaluations": s.neural_evaluations,
            "terminal_evaluations": s.terminal_evaluations, "visits": s.visits, "priors": s.priors,
            "action_values": s.action_values, "log_priors": d.log_priors, "value_sums": d.value_sums,
            "terminal_child_values": d.terminal_child_values, "nodes": s.nodes, "edges": s.edges
        })).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn state<'py>(&self, py: Python<'py>) -> PyResult<PyState<'py>> {
        let (size, color, terminal, score, stones) = py
            .detach(|| {
                let game = self.inner.lock().map_err(|e| e.to_string())?;
                let board = &game.position.board;
                Ok::<_, String>((
                    board.size(),
                    board.to_play() as u8,
                    board.is_terminal(),
                    game.position.score().white_minus_black,
                    board.stones(),
                ))
            })
            .map_err(PyValueError::new_err)?;
        Ok((size, color, terminal, score, stones.into_pyarray(py)))
    }
    fn legal(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        py.detach(|| Ok::<_, String>(self.inner.lock().map_err(|e| e.to_string())?.legal()))
            .map_err(PyValueError::new_err)
    }
    /// Actual root moves plus the pending leaf path, with no target/future move.
    fn request_history<'py>(
        &self,
        py: Python<'py>,
        request: RequestId,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let request = Request {
            identity: Identity {
                search: request.0,
                network: request.1,
            },
            sequence: request.2,
            is_root: request.3,
        };
        let history = py
            .detach(|| {
                self.inner
                    .lock()
                    .map_err(|e| e.to_string())?
                    .request_history(request)
            })
            .map_err(PyValueError::new_err)?;
        Ok(history
            .into_iter()
            .map(|a| a as i32)
            .collect::<Vec<_>>()
            .into_pyarray(py))
    }
    /// Color labels under this game's explicit scoring profile; no board mutation.
    fn ownership<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let values = py
            .detach(|| {
                let game = self.inner.lock().map_err(|e| e.to_string())?;
                Ok::<_, String>(game.position.board.ownership(game.position.scoring.core()))
            })
            .map_err(PyValueError::new_err)?;
        Ok(values.into_pyarray(py))
    }
}
