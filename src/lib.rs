mod cast;
mod error;
mod dtype;
mod kernels;
mod layout;
mod layout_ops;
mod ops;
mod reduce;
mod unary;
mod compare;
mod setops;
mod stats;
mod hist;
mod sortsearch;
mod whereops;
mod linalg;
mod einsum;
mod neighbors;
mod geometry;
mod gpu;
mod dataframe;
mod dataframe_indexing;
mod io;
mod io_cache;
mod random;
mod stream;
mod py_api;

use pyo3::prelude::*;

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_api::register(py, m)?;
    Ok(())
}


