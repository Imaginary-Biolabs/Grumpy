use crate::linalg as linalg_ops;
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;

#[pyfunction]
pub fn dot(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::dot(py, &a.inner, &b.inner)
}

#[pyfunction]
pub fn inner(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::inner(py, &a.inner, &b.inner)
}

#[pyfunction]
pub fn norm(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::norm(py, &a.inner)
}

#[pyfunction]
pub fn trace(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::trace(py, &a.inner)
}

#[pyfunction]
pub fn outer(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = linalg_ops::outer(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn cross(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = linalg_ops::cross(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn det(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::det(py, &a.inner)
}

#[pyfunction]
pub fn inv(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = linalg_ops::inv(py, &a.inner)?;
    Ok(PyGrumpyArray { inner: out })
}
