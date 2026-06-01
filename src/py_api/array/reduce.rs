use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::layout::GrumpyArray;
use crate::reduce::{self, ReduceOp, ReduceOutput};
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    #[pyo3(signature = (dim=None))]
    fn sum(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, dim, ReduceOp::Sum)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    #[pyo3(signature = (dim=None))]
    fn mean(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, dim, ReduceOp::Mean)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    #[pyo3(signature = (dim=None))]
    fn min(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, dim, ReduceOp::Min)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    #[pyo3(signature = (dim=None))]
    fn max(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, dim, ReduceOp::Max)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    #[pyo3(signature = (dim=None))]
    fn ptp(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, dim, ReduceOp::Ptp)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }
}
