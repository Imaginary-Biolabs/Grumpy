use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::py_api::convert::shape_or_nshape;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let lst = self.inner.to_py_list(py)?;
        Ok(format!("GrumpyArray(dtype={}, data={})", self.inner.dtype, lst))
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType { dt: self.inner.dtype }
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner.to_py_list(py)
    }

    fn astype(&self, dtype: PyDType, casting: Option<&str>) -> PyResult<Self> {
        let mode = crate::cast::CastMode::parse(casting.unwrap_or("safe"))?;
        let inner = crate::cast::cast_array_with_mode(&self.inner, dtype.dt, mode)?;
        Ok(Self { inner })
    }

    fn shape(&self, py: Python<'_>, dim: usize) -> PyResult<PyObject> {
        shape_or_nshape(py, &self.inner, dim, false)
    }

    fn nshape(&self, py: Python<'_>, dim: usize) -> PyResult<PyObject> {
        shape_or_nshape(py, &self.inner, dim, true)
    }

    fn nanshape(&self, py: Python<'_>, dim: usize) -> PyResult<PyObject> {
        // Alias; for milestone-1 we treat nanshape == nshape (NaN is a value, not null).
        shape_or_nshape(py, &self.inner, dim, true)
    }

    fn copy(&self) -> Self {
        let mut inner = self.inner.clone();
        inner.uniquify_buffers();
        Self { inner }
    }
}
