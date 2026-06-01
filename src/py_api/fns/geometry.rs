use crate::geometry::grid_pool::{self as grid_pool_ops, GridSpec};
use crate::geometry::pairwise;
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (x, dim=1))]
pub fn pairwise_distances(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    dim: isize,
) -> PyResult<PyGrumpyArray> {
    let inner = x.inner.clone();
    let out = py.allow_threads(move || pairwise::pairwise_distances(&inner, dim))?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, grid_size, origin=None, voxel_size=None, dim=1))]
pub fn grid_pool(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    grid_size: (usize, usize, usize),
    origin: Option<(f64, f64, f64)>,
    voxel_size: Option<(f64, f64, f64)>,
    dim: isize,
) -> PyResult<PyGrumpyArray> {
    let (nx, ny, nz) = grid_size;
    let origin = origin.unwrap_or((0.0, 0.0, 0.0));
    let voxel_size = voxel_size.unwrap_or((1.0, 1.0, 1.0));
    let spec = GridSpec {
        nx,
        ny,
        nz,
        origin: [origin.0, origin.1, origin.2],
        voxel_size: [voxel_size.0, voxel_size.1, voxel_size.2],
    };
    let inner = x.inner.clone();
    let out = py.allow_threads(move || grid_pool_ops::grid_pool(&inner, spec, dim))?;
    Ok(PyGrumpyArray { inner: out })
}
