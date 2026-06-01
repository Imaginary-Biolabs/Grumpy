use crate::setops as set_ops;
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;

#[pyfunction]
pub fn unique(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::unique(py, &x.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn isin(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>, test_elements: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let a = x.inner.clone();
    let test = test_elements.inner.clone();
    let out = py.allow_threads(|| set_ops::isin(&a, &test))?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn setdiff(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::setdiff(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn setunion(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::setunion(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn setxor(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::setxor(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}
