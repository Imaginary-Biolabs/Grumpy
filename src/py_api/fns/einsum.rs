use crate::einsum as einsum_ops;
use crate::layout::GrumpyArray;
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
#[pyo3(signature = (subscripts, *operands))]
pub fn einsum(py: Python<'_>, subscripts: String, operands: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyObject> {
    let mut ops: Vec<GrumpyArray> = Vec::with_capacity(operands.len());
    for i in 0..operands.len() {
        let item = operands.get_item(i)?;
        let arr: PyRef<'_, PyGrumpyArray> = item.extract()?;
        ops.push(arr.inner.clone());
    }
    let result = match einsum_ops::einsum(py, &subscripts, &ops) {
        Ok(r) => r,
        Err(_) => einsum_ops::einsum_numpy_fallback(py, &subscripts, &ops)?,
    };
    match result {
        einsum_ops::TensorOut::Scalar(o) => Ok(o),
        einsum_ops::TensorOut::Array(a) => Ok(Py::new(py, PyGrumpyArray { inner: a })?.into_py(py)),
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, axes=2usize))]
pub fn tensordot(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>, axes: usize) -> PyResult<PyObject> {
    match einsum_ops::tensordot(py, &a.inner, &b.inner, axes)? {
        einsum_ops::TensorOut::Scalar(o) => Ok(o),
        einsum_ops::TensorOut::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
    }
}
