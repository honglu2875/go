use go_actors::observations::{self, Config};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

type PyObservations<'py> = (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<bool>>,
    String,
);

/// One GIL-free call replays all packed episodes and returns pre-action rows.
#[pyfunction]
pub fn replay_observations<'py>(
    py: Python<'py>,
    config_json: &str,
    actions: PyReadonlyArray1<'py, i32>,
    offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<PyObservations<'py>> {
    let config: Config = serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let actions = actions.as_slice()?.to_vec();
    let offsets = offsets.as_slice()?.to_vec();
    let result = py.detach(|| observations::replay(config, &actions, &offsets)).map_err(PyValueError::new_err)?;
    let outcomes = serde_json::to_string(&result.outcomes).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((result.stones.into_pyarray(py), result.legal.into_pyarray(py), outcomes))
}

/// All actions are validated; only suffix observations cross the language boundary.
#[pyfunction]
pub fn replay_suffix_observations<'py>(
    py: Python<'py>,
    config_json: &str,
    actions: PyReadonlyArray1<'py, i32>,
    offsets: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
) -> PyResult<PyObservations<'py>> {
    let config: Config = serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let actions = actions.as_slice()?.to_vec();
    let offsets = offsets.as_slice()?.to_vec();
    let starts = starts.as_slice()?.to_vec();
    let result = py.detach(|| observations::replay_suffix(config, &actions, &offsets, &starts)).map_err(PyValueError::new_err)?;
    let outcomes = serde_json::to_string(&result.outcomes).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((result.stones.into_pyarray(py), result.legal.into_pyarray(py), outcomes))
}
