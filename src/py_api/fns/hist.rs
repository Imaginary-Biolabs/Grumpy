use crate::hist as hist_ops;
use crate::py_api::types::PyGrumpyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
#[pyo3(signature = (x, weights=None, minlength=0))]
pub fn bincount(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    weights: Option<PyRef<'_, PyGrumpyArray>>,
    minlength: usize,
) -> PyResult<PyGrumpyArray> {
    let out = hist_ops::bincount(py, &x.inner, weights.as_ref().map(|w| &w.inner), minlength)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, bins, right=false))]
pub fn digitize(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    bins: PyRef<'_, PyGrumpyArray>,
    right: bool,
) -> PyResult<PyGrumpyArray> {
    let out = hist_ops::digitize(py, &x.inner, &bins.inner, right)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, bins=10, range=None, density=false, weights=None))]
pub fn histogram(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    bins: usize,
    range: Option<Bound<'_, PyAny>>,
    density: bool,
    weights: Option<PyRef<'_, PyGrumpyArray>>,
) -> PyResult<(PyGrumpyArray, PyGrumpyArray)> {
    let range_parsed: Option<(f64, f64)> = if let Some(r) = range {
        let tup = r.downcast::<pyo3::types::PyTuple>().map_err(|_| PyValueError::new_err("range must be a tuple (lo, hi)."))?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err("range must have length 2."));
        }
        Some((tup.get_item(0)?.extract::<f64>()?, tup.get_item(1)?.extract::<f64>()?))
    } else {
        None
    };
    let (h, edges) = hist_ops::histogram(py, &x.inner, bins, range_parsed, density, weights.as_ref().map(|w| &w.inner))?;
    Ok((PyGrumpyArray { inner: h }, PyGrumpyArray { inner: edges }))
}
