use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::sortsearch as ss_ops;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    #[pyo3(signature = (dim=None))]
    fn sort(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<Self> {
        let dim = dim.unwrap_or(-1);
        Ok(Self { inner: ss_ops::sort_axis(py, &self.inner, dim)? })
    }

    #[pyo3(signature = (dim=None))]
    fn argsort(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<Self> {
        let dim = dim.unwrap_or(-1);
        Ok(Self { inner: ss_ops::argsort_axis(py, &self.inner, dim)? })
    }

    #[pyo3(signature = (dim=None))]
    fn argmax(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::ArgMax)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::ArgMax)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (dim=None))]
    fn argmin(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::ArgMin)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::ArgMin)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (dim=None))]
    fn nanargmax(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::NanArgMax)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::NanArgMax)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (dim=None))]
    fn nanargmin(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::NanArgMin)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::NanArgMin)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (kth, dim=None))]
    fn partition(&self, py: Python<'_>, kth: usize, dim: Option<isize>) -> PyResult<Self> {
        match dim {
            None => Ok(Self { inner: ss_ops::partition(py, &self.inner, kth)? }),
            Some(1) | Some(-1) => Ok(Self { inner: ss_ops::partition_dim1(py, &self.inner, kth)? }),
            Some(_) => Err(PyValueError::new_err("partition: only dim=1 is implemented for 2D arrays.")),
        }
    }

    #[pyo3(signature = (kth, dim=None))]
    fn argpartition(&self, py: Python<'_>, kth: usize, dim: Option<isize>) -> PyResult<Self> {
        match dim {
            None => Ok(Self { inner: ss_ops::argpartition(py, &self.inner, kth)? }),
            Some(1) | Some(-1) => Ok(Self { inner: ss_ops::argpartition_dim1(py, &self.inner, kth)? }),
            Some(_) => Err(PyValueError::new_err("argpartition: only dim=1 is implemented for 2D arrays.")),
        }
    }
}
