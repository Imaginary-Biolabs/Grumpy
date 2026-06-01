use crate::whereops as where_ops;
use crate::py_api::types::PyGrumpyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (cond, x=None, y=None))]
pub fn where_(
    py: Python<'_>,
    cond: PyRef<'_, PyGrumpyArray>,
    x: Option<PyRef<'_, PyGrumpyArray>>,
    y: Option<PyRef<'_, PyGrumpyArray>>,
) -> PyResult<PyObject> {
    match (x, y) {
        (None, None) => {
            let out = where_ops::where_indices(py, &cond.inner)?;
            Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
        }
        (Some(xx), Some(yy)) => {
            let out = where_ops::where_select(py, &cond.inner, &xx.inner, &yy.inner)?;
            Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
        }
        _ => Err(PyValueError::new_err("where(cond, x, y) requires both x and y.")),
    }
}

#[pyfunction]
pub fn argwhere(py: Python<'_>, cond: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = where_ops::argwhere(py, &cond.inner)?;
    Ok(PyGrumpyArray { inner: out })
}
