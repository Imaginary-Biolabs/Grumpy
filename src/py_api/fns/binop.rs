use crate::ops::{self, BinOp};
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn multiply(
    a: PyRef<'_, PyGrumpyArray>,
    b: PyRef<'_, PyGrumpyArray>,
    mut out: Option<PyRefMut<'_, PyGrumpyArray>>,
) -> PyResult<PyGrumpyArray> {
    if let Some(ref mut o) = out {
        ops::elementwise_into(&mut o.inner, &a.inner, &b.inner, BinOp::Mul)?;
        Ok(PyGrumpyArray { inner: o.inner.clone() })
    } else {
        let inner = ops::elementwise(&a.inner, &b.inner, BinOp::Mul)?;
        Ok(PyGrumpyArray { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn add_arrays(
    a: PyRef<'_, PyGrumpyArray>,
    b: PyRef<'_, PyGrumpyArray>,
    mut out: Option<PyRefMut<'_, PyGrumpyArray>>,
) -> PyResult<PyGrumpyArray> {
    if let Some(ref mut o) = out {
        ops::elementwise_into(&mut o.inner, &a.inner, &b.inner, BinOp::Add)?;
        Ok(PyGrumpyArray { inner: o.inner.clone() })
    } else {
        let inner = ops::elementwise(&a.inner, &b.inner, BinOp::Add)?;
        Ok(PyGrumpyArray { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn subtract(
    a: PyRef<'_, PyGrumpyArray>,
    b: PyRef<'_, PyGrumpyArray>,
    mut out: Option<PyRefMut<'_, PyGrumpyArray>>,
) -> PyResult<PyGrumpyArray> {
    if let Some(ref mut o) = out {
        ops::elementwise_into(&mut o.inner, &a.inner, &b.inner, BinOp::Sub)?;
        Ok(PyGrumpyArray { inner: o.inner.clone() })
    } else {
        let inner = ops::elementwise(&a.inner, &b.inner, BinOp::Sub)?;
        Ok(PyGrumpyArray { inner })
    }
}
