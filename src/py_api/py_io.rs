use crate::io as io_ops;
use crate::py_api::types::{PyGrumpyArray, PyGrumpyDataFrame};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn stored_len(path: String) -> PyResult<usize> {
    io_ops::stored_axis0_len(&path)
}

#[pyfunction]
pub fn load_slice(py: Python<'_>, path: String, start: usize, stop: usize) -> PyResult<PyObject> {
    let handle = io_ops::DatasetHandle::open(&path)?;
    match &handle.meta.root {
        io_ops::RootMeta::Array { .. } => {
            let sliced = io_ops::load_array_axis0_slice(&handle, start, stop)?;
            Ok(Py::new(py, PyGrumpyArray { inner: sliced })?.into_py(py))
        }
        io_ops::RootMeta::DataFrame { .. } => {
            let sliced = io_ops::load_dataframe_axis0_slice(&handle, start, stop)?;
            Ok(Py::new(py, PyGrumpyDataFrame { inner: sliced })?.into_py(py))
        }
    }
}

#[pyfunction]
pub fn io_bytes_read() -> usize {
    io_ops::io_bytes_read()
}

#[pyfunction]
pub fn reset_io_bytes_read() {
    io_ops::reset_io_bytes_read();
}

#[pyfunction]
#[pyo3(signature = (obj, path, chunk_size=1024usize, chunk_dim=None))]
pub fn save(py: Python<'_>, obj: Bound<'_, PyAny>, path: String, chunk_size: usize, chunk_dim: Option<String>) -> PyResult<()> {
    let depth = chunk_dim
        .as_deref()
        .map(|s| {
            if obj.extract::<PyRef<'_, PyGrumpyArray>>().is_ok() {
                io_ops::resolve_chunk_dim_depth(None, s)
            } else if let Ok(df) = obj.extract::<PyRef<'_, PyGrumpyDataFrame>>() {
                io_ops::resolve_chunk_dim_depth(df.inner.schema.as_ref(), s)
            } else {
                Err(PyValueError::new_err(
                    "gr.save expects a GrumpyArray or GrumpyDataFrame.",
                ))
            }
        })
        .transpose()?;
    if let Ok(arr) = obj.extract::<PyRef<'_, PyGrumpyArray>>() {
        return io_ops::save_array(py, &arr.inner, &path, chunk_size, depth);
    }
    if let Ok(df) = obj.extract::<PyRef<'_, PyGrumpyDataFrame>>() {
        return io_ops::save_dataframe(py, &df.inner, &path, chunk_size, depth);
    }
    Err(PyValueError::new_err("gr.save expects a GrumpyArray or GrumpyDataFrame."))
}

#[pyfunction]
#[pyo3(signature = (obj, path, chunk_size=1024usize, chunk_dim=None))]
pub fn append_batch(py: Python<'_>, obj: Bound<'_, PyAny>, path: String, chunk_size: usize, chunk_dim: Option<String>) -> PyResult<()> {
    let depth = chunk_dim
        .as_deref()
        .map(|s| {
            if obj.extract::<PyRef<'_, PyGrumpyArray>>().is_ok() {
                io_ops::resolve_chunk_dim_depth(None, s)
            } else if let Ok(df) = obj.extract::<PyRef<'_, PyGrumpyDataFrame>>() {
                io_ops::resolve_chunk_dim_depth(df.inner.schema.as_ref(), s)
            } else {
                Err(PyValueError::new_err(
                    "gr.append_batch expects a GrumpyArray or GrumpyDataFrame.",
                ))
            }
        })
        .transpose()?;
    if let Ok(arr) = obj.extract::<PyRef<'_, PyGrumpyArray>>() {
        return io_ops::append_array_axis0(py, &path, &arr.inner, chunk_size, depth);
    }
    if let Ok(df) = obj.extract::<PyRef<'_, PyGrumpyDataFrame>>() {
        return io_ops::append_dataframe_axis0(py, &path, &df.inner, chunk_size, depth);
    }
    Err(PyValueError::new_err(
        "gr.append_batch expects a GrumpyArray or GrumpyDataFrame.",
    ))
}

#[pyfunction]
pub fn load(py: Python<'_>, path: String) -> PyResult<PyObject> {
    // Try array first (fast); if metadata says dataframe, this errors and we fall back.
    if let Ok(arr) = io_ops::load_array(py, &path) {
        return Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py));
    }
    let df = io_ops::load_dataframe(py, &path)?;
    Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
}
