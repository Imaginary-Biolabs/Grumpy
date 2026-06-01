use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::unary as unary_ops;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Sin)? })
    }

    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Cos)? })
    }

    fn tan(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Tan)? })
    }

    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Exp)? })
    }

    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Log)? })
    }

    fn log10(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Log10)? })
    }

    fn log2(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Log2)? })
    }

    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Sqrt)? })
    }

    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Abs)? })
    }

    fn sign(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Sign)? })
    }

    fn floor(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Floor)? })
    }

    fn ceil(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Ceil)? })
    }

    fn round(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Round)? })
    }

    fn reciprocal(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Reciprocal)? })
    }

    fn angle(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Angle)? })
    }
}
