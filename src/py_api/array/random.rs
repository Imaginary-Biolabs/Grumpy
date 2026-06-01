use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::random;
use crate::py_api::random::with_rng;
use crate::py_api::types::PyGenerator;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    #[pyo3(signature = (size, replace=true, dim=0, seed=None, rng=None))]
    fn choice(
        &self,
        py: Python<'_>,
        size: Bound<'_, PyAny>,
        replace: bool,
        dim: isize,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<Self> {
        let parsed = random::parse_choice_size(py, &size)?;
        with_rng(seed, rng, |g| {
            random::choice(g, &self.inner, dim, parsed, replace).map(|inner| Self { inner })
        })
    }

    #[pyo3(signature = (dim=0, seed=None, rng=None))]
    fn permutation(
        &self,
        dim: isize,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<Self> {
        with_rng(seed, rng, |g| {
            random::permutation(g, &self.inner, dim).map(|inner| Self { inner })
        })
    }

    #[pyo3(signature = (dim=0, seed=None, rng=None))]
    fn shuffle(
        &mut self,
        dim: isize,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<()> {
        with_rng(seed, rng, |g| random::shuffle(g, &mut self.inner, dim))
    }

    #[pyo3(signature = (low=0.0, high=1.0, seed=None, rng=None))]
    fn uniform_like(
        &self,
        low: f64,
        high: f64,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<Self> {
        with_rng(seed, rng, |g| {
            random::uniform_like(g, &self.inner, low, high).map(|inner| Self { inner })
        })
    }

    #[pyo3(signature = (seed=None, rng=None))]
    fn random_like(
        &self,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<Self> {
        with_rng(seed, rng, |g| {
            random::random_like(g, &self.inner).map(|inner| Self { inner })
        })
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, seed=None, rng=None))]
    fn normal_like(
        &self,
        loc: f64,
        scale: f64,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<Self> {
        with_rng(seed, rng, |g| {
            random::normal_like(g, &self.inner, loc, scale).map(|inner| Self { inner })
        })
    }

    #[pyo3(signature = (low, high, dtype=None, seed=None, rng=None))]
    fn integers_like(
        &self,
        low: i64,
        high: i64,
        dtype: Option<PyDType>,
        seed: Option<u64>,
        rng: Option<PyRef<'_, PyGenerator>>,
    ) -> PyResult<Self> {
        let dt = dtype.map(|d| d.dt).unwrap_or(DType::Int64);
        with_rng(seed, rng, |g| {
            random::integers_like(g, &self.inner, low, high, dt).map(|inner| Self { inner })
        })
    }
}
