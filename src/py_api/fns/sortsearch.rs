use crate::sortsearch as ss_ops;
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;

#[pyfunction]
pub fn nonzero(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = ss_ops::nonzero(py, &x.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, v, right=false))]
pub fn search_sorted(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    v: PyRef<'_, PyGrumpyArray>,
    right: bool,
) -> PyResult<PyGrumpyArray> {
    let out = ss_ops::search_sorted(py, &x.inner, &v.inner, right)?;
    Ok(PyGrumpyArray { inner: out })
}
