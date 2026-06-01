use crate::neighbors as neigh_ops;
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
#[pyo3(signature = (query, data, k=None, radius=None, dim=0, loop_=true, return_distances=false, gpu="never"))]
pub fn neighbors(
    py: Python<'_>,
    query: PyRef<'_, PyGrumpyArray>,
    data: PyRef<'_, PyGrumpyArray>,
    k: Option<usize>,
    radius: Option<f64>,
    dim: isize,
    loop_: bool,
    return_distances: bool,
    gpu: &str,
) -> PyResult<PyObject> {
    let gpu_pref = crate::gpu::GpuPreference::parse(gpu)?;
    // Release the GIL: neighbors is a pure Rust compute kernel and can run in parallel threads.
    let q = query.inner.clone();
    let d = data.inner.clone();
    let (edge, dist) = py.allow_threads(move || {
        crate::neighbors::neighbors_edge_index_and_distances(&q, &d, k, radius, dim, loop_, return_distances, gpu_pref)
    })?;
    let edge_obj = Py::new(py, PyGrumpyArray { inner: edge })?.into_py(py);
    if let Some(dd) = dist {
        let dist_obj = Py::new(py, PyGrumpyArray { inner: dd })?.into_py(py);
        Ok(pyo3::types::PyTuple::new_bound(py, [edge_obj, dist_obj]).into_py(py))
    } else {
        Ok(edge_obj)
    }
}
