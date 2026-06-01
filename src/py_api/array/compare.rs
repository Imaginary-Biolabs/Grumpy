use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::compare as cmp_ops;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    fn isnan(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::predicate(py, &self.inner, cmp_ops::PredOp::IsNan)? })
    }

    fn isfinite(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::predicate(py, &self.inner, cmp_ops::PredOp::IsFinite)? })
    }

    fn isinf(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::predicate(py, &self.inner, cmp_ops::PredOp::IsInf)? })
    }

    fn equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Eq)? })
    }

    fn not_equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Ne)? })
    }

    fn less(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Lt)? })
    }

    fn less_equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Le)? })
    }

    fn greater(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Gt)? })
    }

    fn greater_equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Ge)? })
    }

    fn logical_and(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_bin(py, &self.inner, &other.inner, cmp_ops::LogicOp::And)? })
    }

    fn logical_or(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_bin(py, &self.inner, &other.inner, cmp_ops::LogicOp::Or)? })
    }

    fn logical_xor(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_bin(py, &self.inner, &other.inner, cmp_ops::LogicOp::Xor)? })
    }

    fn logical_not(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_not(py, &self.inner)? })
    }
}
