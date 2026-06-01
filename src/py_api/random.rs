use crate::dtype::{DType, PyDType};
use crate::random::{self, GrumpyRng};
use crate::py_api::types::{PyGenerator, PyGrumpyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;

pub(crate) fn with_rng<F, R>(seed: Option<u64>, rng: Option<PyRef<'_, PyGenerator>>, f: F) -> PyResult<R>
where
    F: FnOnce(&mut GrumpyRng) -> PyResult<R>,
{
    if let Some(r) = rng {
        return f(&mut *r.inner.borrow_mut());
    }
    if let Some(s) = seed {
        let mut local = GrumpyRng::new(s);
        return f(&mut local);
    }
    Err(PyValueError::new_err(
        "Provide seed=... or rng=... for this operation.",
    ))
}

#[pymethods]
impl PyGenerator {
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        Self {
            inner: RefCell::new(GrumpyRng::new(seed)),
        }
    }

    #[pyo3(signature = (a, size, replace=true, dim=0))]
    fn choice(
        &self,
        py: Python<'_>,
        a: PyRef<'_, PyGrumpyArray>,
        size: Bound<'_, PyAny>,
        replace: bool,
        dim: isize,
    ) -> PyResult<PyGrumpyArray> {
        let parsed = random::parse_choice_size(py, &size)?;
        let out = random::choice(
            &mut *self.inner.borrow_mut(),
            &a.inner,
            dim,
            parsed,
            replace,
        )?;
        Ok(PyGrumpyArray { inner: out })
    }

    #[pyo3(signature = (a, low=0.0, high=1.0))]
    fn uniform_like(&self, a: PyRef<'_, PyGrumpyArray>, low: f64, high: f64) -> PyResult<PyGrumpyArray> {
        let out = random::uniform_like(&mut *self.inner.borrow_mut(), &a.inner, low, high)?;
        Ok(PyGrumpyArray { inner: out })
    }

    fn random_like(&self, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
        let out = random::random_like(&mut *self.inner.borrow_mut(), &a.inner)?;
        Ok(PyGrumpyArray { inner: out })
    }

    #[pyo3(signature = (a, loc=0.0, scale=1.0))]
    fn normal_like(&self, a: PyRef<'_, PyGrumpyArray>, loc: f64, scale: f64) -> PyResult<PyGrumpyArray> {
        let out = random::normal_like(&mut *self.inner.borrow_mut(), &a.inner, loc, scale)?;
        Ok(PyGrumpyArray { inner: out })
    }

    #[pyo3(signature = (a, low, high, dtype=None))]
    fn integers_like(
        &self,
        a: PyRef<'_, PyGrumpyArray>,
        low: i64,
        high: i64,
        dtype: Option<PyDType>,
    ) -> PyResult<PyGrumpyArray> {
        let dt = dtype.map(|d| d.dt).unwrap_or(DType::Int64);
        let out = random::integers_like(
            &mut *self.inner.borrow_mut(),
            &a.inner,
            low,
            high,
            dt,
        )?;
        Ok(PyGrumpyArray { inner: out })
    }

    #[pyo3(signature = (low, high, size, dtype=None))]
    fn integers(
        &self,
        low: i64,
        high: i64,
        size: usize,
        dtype: Option<PyDType>,
    ) -> PyResult<PyGrumpyArray> {
        let dt = dtype.map(|d| d.dt).unwrap_or(DType::Int64);
        let out = random::integers(
            &mut *self.inner.borrow_mut(),
            low,
            high,
            size,
            dt,
        )?;
        Ok(PyGrumpyArray { inner: out })
    }

    #[pyo3(signature = (a, dim=0))]
    fn permutation(&self, a: PyRef<'_, PyGrumpyArray>, dim: isize) -> PyResult<PyGrumpyArray> {
        let out = random::permutation(&mut *self.inner.borrow_mut(), &a.inner, dim)?;
        Ok(PyGrumpyArray { inner: out })
    }

    #[pyo3(signature = (a, dim=0))]
    fn shuffle(&self, mut a: PyRefMut<'_, PyGrumpyArray>, dim: isize) -> PyResult<()> {
        random::shuffle(&mut *self.inner.borrow_mut(), &mut a.inner, dim)
    }
}

#[pyfunction]
#[pyo3(name = "rng", signature = (seed=0))]
pub fn py_rng(seed: u64) -> PyGenerator {
    PyGenerator::new(seed)
}
