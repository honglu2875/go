//! Coarse native batches. Array ownership transfers to NumPy; Rust workers detach
//! from Python during search, transitions, feature encoding, and replay labeling.
use go_actors::{Batch, Config, Pool};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Mutex;
mod bounded_traces;
mod game;
mod observations;
mod traces;

type PyBatch<'py> = (
    u64,
    u64,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u8>>,
);
type PyRows<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<f32>>,
    String,
);
fn output<'py>(py: Python<'py>, batch: Batch) -> PyBatch<'py> {
    (
        batch.round,
        batch.network,
        batch.features.into_pyarray(py),
        batch.active.into_pyarray(py),
    )
}

#[pyclass]
struct Actors {
    pool: Mutex<Pool>,
}

#[pymethods]
impl Actors {
    #[new]
    #[pyo3(signature = (config_json, checkpoint_json=None))]
    fn new(py: Python<'_>, config_json: &str, checkpoint_json: Option<&str>) -> PyResult<Self> {
        let config: Config =
            serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let state = checkpoint_json
            .map(serde_json::from_str::<go_actors::PoolState>)
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pool = py
            .detach(|| match state {
                Some(state) => Pool::restore(config, state),
                None => Pool::new(config),
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            pool: Mutex::new(pool),
        })
    }
    fn checkpoint(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            let state = self.pool.lock().map_err(|e| e.to_string())?.checkpoint()?;
            serde_json::to_string(&state).map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)
    }
    fn start<'py>(&self, py: Python<'py>, network: u64) -> PyResult<PyBatch<'py>> {
        let batch = py
            .detach(|| self.pool.lock().map_err(|e| e.to_string())?.start(network))
            .map_err(PyRuntimeError::new_err)?;
        Ok(output(py, batch))
    }
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        round: u64,
        network: u64,
        policies: PyReadonlyArray2<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<PyBatch<'py>> {
        let (games, actions) = {
            let pool = self
                .pool
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            (pool.config().games, pool.config().actions())
        };
        if policies.shape() != [games, actions] || values.shape() != [games] {
            return Err(PyValueError::new_err(
                "neural output dimensions differ from the actor batch",
            ));
        }
        // Copy small NN outputs before releasing Python, so another Python thread
        // cannot race a native worker by mutating the input array's storage.
        let policies = policies.as_slice()?.to_vec();
        let values = values.as_slice()?.to_vec();
        let batch = py
            .detach(|| {
                self.pool
                    .lock()
                    .map_err(|e| e.to_string())?
                    .evaluate(round, network, &policies, &values)
            })
            .map_err(PyValueError::new_err)?;
        Ok(output(py, batch))
    }
    fn commit<'py>(&self, py: Python<'py>) -> PyResult<PyRows<'py>> {
        let (features, policies, outcomes, root_values, ids, ownership, metadata) = py
            .detach(|| {
                let mut games = self.pool.lock().map_err(|e| e.to_string())?.commit()?;
                let metadata = serde_json::to_string(&games).map_err(|e| e.to_string())?;
                let mut features = Vec::new();
                let mut policies = Vec::new();
                let mut outcomes = Vec::new();
                let mut root_values = Vec::new();
                let mut ids = Vec::new();
                let mut ownership = Vec::new();
                for game in &mut games {
                    if game.truncated {
                        assert!(game.rows.is_empty());
                        continue;
                    }
                    let score = game.white_score.ok_or("terminal score missing")?;
                    let white_value = if score > 0.0 {
                        1.0
                    } else if score < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    for row in game.rows.drain(..) {
                        features.extend(row.features);
                        policies.extend(row.policy);
                        outcomes.push(if row.black_to_play {
                            -white_value
                        } else {
                            white_value
                        });
                        root_values.push(row.root_value);
                        let perspective = if row.black_to_play { -1.0 } else { 1.0 };
                        ownership
                            .extend(game.ownership.iter().map(|&v| f32::from(v) * perspective));
                        ids.extend([
                            game.game_id,
                            row.network,
                            row.action as u64,
                            u64::from(row.simulations),
                            u64::from(row.neural_evaluations),
                            u64::from(row.terminal_evaluations),
                        ]);
                    }
                }
                Ok::<_, String>((
                    features,
                    policies,
                    outcomes,
                    root_values,
                    ids,
                    ownership,
                    metadata,
                ))
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok((
            features.into_pyarray(py),
            policies.into_pyarray(py),
            outcomes.into_pyarray(py),
            root_values.into_pyarray(py),
            ids.into_pyarray(py),
            ownership.into_pyarray(py),
            metadata,
        ))
    }
    fn prefetch<'py>(
        &self,
        py: Python<'py>,
        round: u64,
        network: u64,
        limit: usize,
    ) -> PyResult<PyBatch<'py>> {
        let batch = py
            .detach(|| {
                self.pool
                    .lock()
                    .map_err(|e| e.to_string())?
                    .prefetch(round, network, limit)
            })
            .map_err(PyValueError::new_err)?;
        Ok(output(py, batch))
    }
    fn evaluate_prefetch<'py>(
        &self,
        py: Python<'py>,
        round: u64,
        network: u64,
        policies: PyReadonlyArray2<'py, f32>,
        values: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<PyBatch<'py>> {
        let actions = self
            .pool
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .config()
            .actions();
        if policies.shape() != [values.len(), actions] {
            return Err(PyValueError::new_err(
                "root prefetch output dimensions differ",
            ));
        }
        let policies = policies.as_slice()?.to_vec();
        let values = values.as_slice()?.to_vec();
        let batch = py
            .detach(|| {
                self.pool
                    .lock()
                    .map_err(|e| e.to_string())?
                    .evaluate_prefetch(round, network, &policies, &values)
            })
            .map_err(PyValueError::new_err)?;
        Ok(output(py, batch))
    }
    fn prefetch_counters(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        py.detach(|| {
            self.pool
                .lock()
                .map_err(|e| e.to_string())?
                .prefetch_counters()
        })
        .map_err(PyRuntimeError::new_err)
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.pool.lock().map_err(|e| e.to_string())?.close();
            Ok::<_, String>(())
        })
        .map_err(PyRuntimeError::new_err)
    }
}

#[pymodule]
fn _gozero_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ABI_VERSION", 2)?;
    module.add("DUAL_TRACE_ABI_VERSION", 2)?;
    module.add("STATE_TRACE_ABI_VERSION", 1)?;
    module.add("BOUNDED_STATE_TRACE_ABI_VERSION", 1)?;
    module.add("ROOT_PREFETCH_ABI_VERSION", 1)?;
    module.add("CAUSAL_GAME_ABI_VERSION", 1)?;
    module.add("SEARCH_INSPECTION_ABI_VERSION", 1)?;
    module.add("OBSERVATION_REPLAY_ABI_VERSION", 1)?;
    module.add_function(wrap_pyfunction!(observations::replay_observations, module)?)?;
    module.add("OBSERVATION_SUFFIX_REPLAY_ABI_VERSION", 1)?;
    module.add_function(wrap_pyfunction!(observations::replay_suffix_observations, module)?)?;
    module.add_class::<Actors>()?;
    module.add_class::<game::Game>()?;
    module.add_class::<traces::DualTraceActors>()?;
    module.add_class::<bounded_traces::BoundedTraceActors>()?;
    Ok(())
}
