mod dtype;
mod kernels;
mod layout;
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
mod dataframe;
mod io;
mod py_api;

use pyo3::prelude::*;

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_api::register(py, m)?;
    Ok(())
}


