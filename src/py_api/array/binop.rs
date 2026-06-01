use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::ops::BinOp;
use crate::py_api::binop::apply_elementwise_binop;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    fn __add__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Add).map(|inner| Self { inner })
    }

    fn __sub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Sub).map(|inner| Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Mul).map(|inner| Self { inner })
    }

    fn __truediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Div).map(|inner| Self { inner })
    }

    fn __mod__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Mod).map(|inner| Self { inner })
    }

    fn remainder(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Remainder).map(|inner| Self { inner })
    }

    fn mod_(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Mod).map(|inner| Self { inner })
    }
}
