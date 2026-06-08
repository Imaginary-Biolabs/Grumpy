//! Python bindings (pyo3) for the Rust core.
//!
//! Conventions:
//! - Keep the Python layer thin: parse args, call Rust kernels, convert outputs.
//! - Hot paths must avoid Python loops and avoid building Python lists (use typed buffers + kernels).
//! - Compiled pipelines:
//!   - `PyCompiledPlan` executes a restricted IR with `py.allow_threads` where possible.
//!
//! When adding a new op:
//! - Add the Rust kernel in a dedicated module (`src/<opgroup>.rs`) and make it **no-GIL** if possible.
//! - Expose it here via a method on `PyGrumpyArray` / `PyGrumpyDataFrame` or as a free function.
//! - If it's performance-critical in `Stream.apply`, consider extending:
//!   - `PlanOp` + `python/grumpy/compiler.py` compilation rules
//!   - `run_plan_*_rust` for Rust scheduling of fully compiled pipelines

mod array;
mod bench;
mod binop;
mod cast;
mod compile;
mod convert;
mod dataframe;
mod fns;
mod indexing;
mod interop;
mod open;
mod py_io;
mod random;
mod types;

pub use types::*;

use crate::dtype::PyDType;
use cast::{py_can_cast, py_promote_types};
use open::open_dataset;
use fns::array::{array as gr_array, cat, full_like, ones_like, zeros_like};
use interop::{from_numpy, from_tensorflow, from_torch, is_rectangular};
use fns::binop::{add_arrays, multiply, subtract};
use fns::dataframe::dataframe as gr_dataframe;
use fns::einsum::{einsum, tensordot};
use fns::hist::{bincount, digitize, histogram};
use fns::linalg::{cross, det, dot, inner, inv, norm, outer, trace};
use fns::geometry::{grid_pool as grid_pool_fn, pairwise_distances};
use fns::gpu::{gpu_available, gpu_backend};
use fns::neighbors::neighbors;
use fns::setops::{isin, setdiff, setunion, setxor, unique};
use fns::sortsearch::{nonzero, search_sorted};
use fns::whereops::{argwhere, where_};
use py_io::{
    append_batch, clear_path_caches, io_bytes_read, io_cache_stats, load, load_slice,
    reset_io_bytes_read, save, stored_len,
};
use pyo3::prelude::*;
use random::py_rng;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyGenerator>()?;
    m.add_class::<PyGrumpyArray>()?;
    m.add_class::<PyGrumpyDataFrame>()?;
    m.add_class::<PyDataFrameAccessor>()?;
    m.add_class::<PyCompiledPlan>()?;
    m.add_class::<PyCompiledBatchesIter>()?;
    m.add_class::<PyOpenDataFrame>()?;
    m.add_class::<PyOpenColumn>()?;
    m.add_class::<open::PyOpenDataFrameAccessor>()?;
    m.add_function(wrap_pyfunction!(py_can_cast, m)?)?;
    m.add_function(wrap_pyfunction!(py_promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(py_rng, m)?)?;
    m.add_function(wrap_pyfunction!(gr_array, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(add_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(isin, m)?)?;
    m.add_function(wrap_pyfunction!(setdiff, m)?)?;
    m.add_function(wrap_pyfunction!(setunion, m)?)?;
    m.add_function(wrap_pyfunction!(setxor, m)?)?;
    m.add_function(wrap_pyfunction!(bincount, m)?)?;
    m.add_function(wrap_pyfunction!(digitize, m)?)?;
    m.add_function(wrap_pyfunction!(histogram, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(search_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(where_, m)?)?;
    m.add_function(wrap_pyfunction!(argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(einsum, m)?)?;
    m.add_function(wrap_pyfunction!(tensordot, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_backend, m)?)?;
    m.add_function(wrap_pyfunction!(neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distances, m)?)?;
    m.add_function(wrap_pyfunction!(grid_pool_fn, m)?)?;
    m.add_function(wrap_pyfunction!(gr_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(append_batch, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(stored_len, m)?)?;
    m.add_function(wrap_pyfunction!(load_slice, m)?)?;
    m.add_function(wrap_pyfunction!(open_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(io_bytes_read, m)?)?;
    m.add_function(wrap_pyfunction!(reset_io_bytes_read, m)?)?;
    m.add_function(wrap_pyfunction!(clear_path_caches, m)?)?;
    m.add_function(wrap_pyfunction!(io_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(from_tensorflow, m)?)?;
    m.add_function(wrap_pyfunction!(is_rectangular, m)?)?;
    Ok(())
}
