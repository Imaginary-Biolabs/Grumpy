use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::layout::Layout;
use crate::stats as stats_ops;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    #[pyo3(signature = (dim=None, ddof=0))]
    fn var(&self, py: Python<'_>, dim: Option<isize>, ddof: isize) -> PyResult<PyObject> {
        let arr = stats_ops::var(py, &self.inner, dim, ddof, false)?;
        if arr.len() == 1 {
            if let Layout::Leaf(l) = &arr.layout {
                return l.scalar_to_py(py, 0);
            }
        }
        Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py))
    }

    #[pyo3(signature = (dim=None, ddof=0))]
    fn std(&self, py: Python<'_>, dim: Option<isize>, ddof: isize) -> PyResult<PyObject> {
        let arr = stats_ops::std(py, &self.inner, dim, ddof, false)?;
        if arr.len() == 1 {
            if let Layout::Leaf(l) = &arr.layout {
                return l.scalar_to_py(py, 0);
            }
        }
        Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py))
    }

    #[pyo3(signature = (dim=None, ddof=0))]
    fn nanvar(&self, py: Python<'_>, dim: Option<isize>, ddof: isize) -> PyResult<PyObject> {
        let arr = stats_ops::var(py, &self.inner, dim, ddof, true)?;
        if arr.len() == 1 {
            if let Layout::Leaf(l) = &arr.layout {
                return l.scalar_to_py(py, 0);
            }
        }
        Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py))
    }

    #[pyo3(signature = (dim=None, ddof=0))]
    fn nanstd(&self, py: Python<'_>, dim: Option<isize>, ddof: isize) -> PyResult<PyObject> {
        let arr = stats_ops::std(py, &self.inner, dim, ddof, true)?;
        if arr.len() == 1 {
            if let Layout::Leaf(l) = &arr.layout {
                return l.scalar_to_py(py, 0);
            }
        }
        Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py))
    }

    fn quantile(&self, py: Python<'_>, q: f64, dim: isize) -> PyResult<Self> {
        Ok(Self {
            inner: stats_ops::quantile(py, &self.inner, dim, vec![q], stats_ops::QuantileMode::Quantile, false)?,
        })
    }

    fn nanquantile(&self, py: Python<'_>, q: f64, dim: isize) -> PyResult<Self> {
        Ok(Self {
            inner: stats_ops::quantile(py, &self.inner, dim, vec![q], stats_ops::QuantileMode::Quantile, true)?,
        })
    }

    fn percentile(&self, py: Python<'_>, q: f64, dim: isize) -> PyResult<Self> {
        Ok(Self {
            inner: stats_ops::quantile(py, &self.inner, dim, vec![q], stats_ops::QuantileMode::Percentile, false)?,
        })
    }

    fn nanpercentile(&self, py: Python<'_>, q: f64, dim: isize) -> PyResult<Self> {
        Ok(Self {
            inner: stats_ops::quantile(py, &self.inner, dim, vec![q], stats_ops::QuantileMode::Percentile, true)?,
        })
    }

    fn median(&self, py: Python<'_>, dim: isize) -> PyResult<Self> {
        Ok(Self { inner: stats_ops::median(py, &self.inner, dim, false)? })
    }

    fn nanmedian(&self, py: Python<'_>, dim: isize) -> PyResult<Self> {
        Ok(Self { inner: stats_ops::median(py, &self.inner, dim, true)? })
    }
}
