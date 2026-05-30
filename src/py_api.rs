//! Python bindings (pyo3) for the Rust core.
//!
//! Conventions:
//! - Keep the Python layer thin: parse args, call Rust kernels, convert outputs.
//! - Hot paths must avoid Python loops and avoid building Python lists (use typed buffers + kernels).
//! - Compiled pipelines:
//!   - `PyCompiledPlan` executes a restricted IR with `py.allow_threads` where possible.
//!   - `_core.compiled_stream_apply` runs fully-compiled pipelines with Rust scheduling (rayon thread pool).
//!
//! When adding a new op:
//! - Add the Rust kernel in a dedicated module (`src/<opgroup>.rs`) and make it **no-GIL** if possible.
//! - Expose it here via a method on `PyGrumpyArray` / `PyGrumpyDataFrame` or as a free function.
//! - If it’s performance-critical in `Stream.apply`, consider extending:
//!   - `PlanOp` + `python/grumpy/compiler.py` compilation rules
//!   - `run_plan_*_rust` for Rust scheduling of fully compiled pipelines

use crate::dtype::{infer_dtype, inferclass_to_dtype, DType, PyDType};
use crate::layout::{
    build_array, concat_to_py_list, coord_to_leaf_index, drop_axis0_select_element, fill_layout_like,
    gather_2d_fancy_leaf, gather_2d_fancy_sum_i64, scatter_2d_fancy_i32, scatter_2d_fancy_numeric,
    GrumpyArray, Layout, LeafBuffer, OffsetView,
};
use crate::ops::{self, BinOp};
use crate::reduce::{self, ReduceOp, ReduceOutput};
use crate::unary as unary_ops;
use crate::compare as cmp_ops;
use crate::setops as set_ops;
use crate::stats as stats_ops;
use crate::hist as hist_ops;
use crate::sortsearch as ss_ops;
use crate::whereops as where_ops;
use crate::linalg as linalg_ops;
use crate::einsum as einsum_ops;
use crate::neighbors as neigh_ops;
use crate::dataframe as df_ops;
use crate::io as io_ops;
use std::sync::Arc;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyAnyMethods, PyBool, PyFloat, PyInt, PySlice, PyTuple};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

#[pyclass(name = "GrumpyArray")]
pub struct PyGrumpyArray {
    inner: GrumpyArray,
}

#[pyclass(name = "GrumpyDataFrame")]
pub struct PyGrumpyDataFrame {
    inner: df_ops::GrumpyDataFrame,
}

#[pyclass(name = "DataFrameAccessor")]
pub struct PyDataFrameAccessor {
    parent: Py<PyGrumpyDataFrame>,
    // Schema levels path, e.g. ["residue"] or ["molecule","residue"].
    path: Vec<String>,
}

#[derive(Clone, Debug)]
enum PlanOp {
    AddScalar { value: f64, is_int: bool },
    SubScalar { value: f64, is_int: bool },
    MulScalar { value: f64, is_int: bool },
    DivScalar { value: f64, is_int: bool },
    ModScalar { value: f64, is_int: bool },
    MulScalarSumAll { value: f64, is_int: bool },
    NeighborsKnnSelf { k: usize, dim: isize, loop_: bool },
    ReduceCur { op: ReduceOp, dim: Option<isize> },
    DfGetTmp { level0: String, col: String },
    ReduceTmp { op: ReduceOp, dim: isize },
    DfSetTmp { level0: String, col: String },
}

#[pyclass(name = "CompiledPlan")]
pub struct PyCompiledPlan {
    ops: Vec<PlanOp>,
}

#[pyclass(name = "CompiledBatchesIter")]
pub struct PyCompiledBatchesIter {
    arr_batches: Option<Vec<GrumpyArray>>,
    df_batches: Option<Vec<df_ops::GrumpyDataFrame>>,
    pos: usize,
}

#[pymethods]
impl PyCompiledBatchesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if let Some(b) = slf.arr_batches.as_ref() {
            if slf.pos >= b.len() {
                return Ok(None);
            }
            let out = b[slf.pos].clone();
            slf.pos += 1;
            return Ok(Some(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py)));
        }
        if let Some(b) = slf.df_batches.as_ref() {
            if slf.pos >= b.len() {
                return Ok(None);
            }
            let out = b[slf.pos].clone();
            slf.pos += 1;
            return Ok(Some(
                Py::new(py, PyGrumpyDataFrame { inner: out })?.into_py(py),
            ));
        }
        Ok(None)
    }
}

#[pymethods]
impl PyCompiledPlan {
    #[new]
    fn new(spec: Bound<'_, PyAny>) -> PyResult<Self> {
        let seq = spec
            .downcast::<pyo3::types::PyList>()
            .map_err(|_| PyValueError::new_err("CompiledPlan spec must be a list of dicts."))?;
        let mut ops: Vec<PlanOp> = Vec::with_capacity(seq.len());
        for i in 0..seq.len() {
            let item = seq.get_item(i)?;
            let d = item
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("CompiledPlan spec entries must be dicts."))?;
            let op_obj = d
                .get_item("op")?
                .ok_or_else(|| PyValueError::new_err("CompiledPlan op dict missing 'op' field."))?;
            let op_name: String = op_obj
                .extract()
                .map_err(|_| PyValueError::new_err("CompiledPlan op 'op' must be a string."))?;
            let is_int = d
                .get_item("is_int")?
                .and_then(|x| x.extract::<bool>().ok())
                .unwrap_or(false);
            let val_f64 = || -> PyResult<f64> {
                let v = d
                    .get_item("value")?
                    .ok_or_else(|| PyValueError::new_err("Scalar op missing 'value'."))?;
                v.extract::<f64>()
                    .map_err(|_| PyValueError::new_err("Scalar op 'value' must be a number."))
            };
            match op_name.as_str() {
                "add_scalar" => ops.push(PlanOp::AddScalar { value: val_f64()?, is_int }),
                "sub_scalar" => ops.push(PlanOp::SubScalar { value: val_f64()?, is_int }),
                "mul_scalar" => ops.push(PlanOp::MulScalar { value: val_f64()?, is_int }),
                "div_scalar" => ops.push(PlanOp::DivScalar { value: val_f64()?, is_int }),
                "mod_scalar" => ops.push(PlanOp::ModScalar { value: val_f64()?, is_int }),
                "mul_scalar_sum_all" => {
                    ops.push(PlanOp::MulScalarSumAll { value: val_f64()?, is_int });
                }
                "neighbors_knn_self" => {
                    let k: usize = d
                        .get_item("k")?
                        .ok_or_else(|| PyValueError::new_err("neighbors op missing 'k'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("neighbors op 'k' must be int."))?;
                    let dim: isize = d
                        .get_item("dim")?
                        .and_then(|x| x.extract::<isize>().ok())
                        .unwrap_or(0);
                    let loop_: bool = d
                        .get_item("loop")?
                        .and_then(|x| x.extract::<bool>().ok())
                        .unwrap_or(true);
                    ops.push(PlanOp::NeighborsKnnSelf { k, dim, loop_ });
                }
                "reduce" => {
                    let which: String = d
                        .get_item("reduce")?
                        .ok_or_else(|| PyValueError::new_err("reduce op missing 'reduce'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("reduce op 'reduce' must be a string."))?;
                    let dim: Option<isize> = d
                        .get_item("dim")?
                        .map(|x| x.extract::<isize>())
                        .transpose()
                        .map_err(|_| PyValueError::new_err("reduce op 'dim' must be an int."))?;
                    let rop = match which.as_str() {
                        "sum" => ReduceOp::Sum,
                        "mean" => ReduceOp::Mean,
                        "min" => ReduceOp::Min,
                        "max" => ReduceOp::Max,
                        "ptp" => ReduceOp::Ptp,
                        _ => return Err(PyValueError::new_err("reduce op: unsupported reduction.")),
                    };
                    ops.push(PlanOp::ReduceCur { op: rop, dim });
                }
                "df_get" => {
                    let level0: String = d
                        .get_item("level0")?
                        .ok_or_else(|| PyValueError::new_err("df_get missing 'level0'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("df_get 'level0' must be a string."))?;
                    let col: String = d
                        .get_item("col")?
                        .ok_or_else(|| PyValueError::new_err("df_get missing 'col'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("df_get 'col' must be a string."))?;
                    ops.push(PlanOp::DfGetTmp { level0, col });
                }
                "reduce_tmp" => {
                    let which: String = d
                        .get_item("reduce")?
                        .ok_or_else(|| PyValueError::new_err("reduce_tmp missing 'reduce'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("reduce_tmp 'reduce' must be a string."))?;
                    let dim: isize = d
                        .get_item("dim")?
                        .ok_or_else(|| PyValueError::new_err("reduce_tmp missing 'dim'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("reduce_tmp 'dim' must be an int."))?;
                    let rop = match which.as_str() {
                        "sum" => ReduceOp::Sum,
                        "mean" => ReduceOp::Mean,
                        "min" => ReduceOp::Min,
                        "max" => ReduceOp::Max,
                        "ptp" => ReduceOp::Ptp,
                        _ => return Err(PyValueError::new_err("reduce_tmp: unsupported reduction.")),
                    };
                    ops.push(PlanOp::ReduceTmp { op: rop, dim });
                }
                "df_set" => {
                    let level0: String = d
                        .get_item("level0")?
                        .ok_or_else(|| PyValueError::new_err("df_set missing 'level0'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("df_set 'level0' must be a string."))?;
                    let col: String = d
                        .get_item("col")?
                        .ok_or_else(|| PyValueError::new_err("df_set missing 'col'."))?
                        .extract()
                        .map_err(|_| PyValueError::new_err("df_set 'col' must be a string."))?;
                    ops.push(PlanOp::DfSetTmp { level0, col });
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown compiled op '{op_name}'."
                    )))
                }
            }
        }
        Ok(Self { ops })
    }

    fn __repr__(&self) -> String {
        format!("CompiledPlan(n_ops={})", self.ops.len())
    }

    fn run(&self, py: Python<'_>, batch: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let mut cur_arr: Option<GrumpyArray> = None;
        let mut cur_df: Option<df_ops::GrumpyDataFrame> = None;
        let mut cur_scalar: Option<PyObject> = None;

        if let Ok(a) = batch.extract::<PyRef<'_, PyGrumpyArray>>() {
            cur_arr = Some(a.inner.clone());
        } else if let Ok(df) = batch.extract::<PyRef<'_, PyGrumpyDataFrame>>() {
            cur_df = Some(df.inner.clone());
        } else {
            return Err(PyValueError::new_err(
                "CompiledPlan.run expects a GrumpyArray or GrumpyDataFrame batch.",
            ));
        }
        let mut tmp: Option<GrumpyArray> = None;

        for op in &self.ops {
            match op {
                PlanOp::AddScalar { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("add_scalar requires array batch."))?;
                    let a = py.allow_threads(move || -> PyResult<GrumpyArray> {
                        let mut a = a0;
                        let did = ops::elementwise_scalar_inplace(&mut a, BinOp::Add, *value, *is_int)?;
                        if !did {
                            let rhs = scalar_like(a.dtype, *value, *is_int)?;
                            a = ops::elementwise(&a, &rhs, BinOp::Add)?;
                        }
                        Ok(a)
                    })?;
                    cur_arr = Some(a);
                }
                PlanOp::SubScalar { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("sub_scalar requires array batch."))?;
                    let a = py.allow_threads(move || -> PyResult<GrumpyArray> {
                        let mut a = a0;
                        let did = ops::elementwise_scalar_inplace(&mut a, BinOp::Sub, *value, *is_int)?;
                        if !did {
                            let rhs = scalar_like(a.dtype, *value, *is_int)?;
                            a = ops::elementwise(&a, &rhs, BinOp::Sub)?;
                        }
                        Ok(a)
                    })?;
                    cur_arr = Some(a);
                }
                PlanOp::MulScalar { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("mul_scalar requires array batch."))?;
                    let a = py.allow_threads(move || -> PyResult<GrumpyArray> {
                        let mut a = a0;
                        let did = ops::elementwise_scalar_inplace(&mut a, BinOp::Mul, *value, *is_int)?;
                        if !did {
                            let rhs = scalar_like(a.dtype, *value, *is_int)?;
                            a = ops::elementwise(&a, &rhs, BinOp::Mul)?;
                        }
                        Ok(a)
                    })?;
                    cur_arr = Some(a);
                }
                PlanOp::DivScalar { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("div_scalar requires array batch."))?;
                    let a = py.allow_threads(move || -> PyResult<GrumpyArray> {
                        let mut a = a0;
                        let did = ops::elementwise_scalar_inplace(&mut a, BinOp::Div, *value, *is_int)?;
                        if !did {
                            let rhs = scalar_like(a.dtype, *value, *is_int)?;
                            a = ops::elementwise(&a, &rhs, BinOp::Div)?;
                        }
                        Ok(a)
                    })?;
                    cur_arr = Some(a);
                }
                PlanOp::ModScalar { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("mod_scalar requires array batch."))?;
                    let a = py.allow_threads(move || -> PyResult<GrumpyArray> {
                        let mut a = a0;
                        let did = ops::elementwise_scalar_inplace(&mut a, BinOp::Mod, *value, *is_int)?;
                        if !did {
                            let rhs = scalar_like(a.dtype, *value, *is_int)?;
                            a = ops::elementwise(&a, &rhs, BinOp::Mod)?;
                        }
                        Ok(a)
                    })?;
                    cur_arr = Some(a);
                }
                PlanOp::MulScalarSumAll { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("mul_scalar_sum_all requires array batch."))?;
                    if !is_int {
                        return Err(PyValueError::new_err("mul_scalar_sum_all requires int scalar."));
                    }
                    let s = *value as i32;
                    let sum = py.allow_threads(move || ops::mul_scalar_sum_all_i64(&a0, s))?;
                    cur_scalar = Some(sum.to_object(py));
                }
                PlanOp::NeighborsKnnSelf { k, dim, loop_ } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("neighbors requires array batch."))?;
                    let kk = *k;
                    let dd = *dim;
                    let lp = *loop_;
                    let a = py.allow_threads(move || neigh_ops::neighbors(&a0, &a0, Some(kk), None, dd, lp))?;
                    cur_arr = Some(a);
                }
                PlanOp::ReduceCur { op, dim } => {
                    let a0 = cur_arr.take().ok_or_else(|| PyValueError::new_err("reduce requires array batch."))?;
                    match reduce::reduce(py, &a0, *dim, *op)? {
                        ReduceOutput::Array(out) => cur_arr = Some(out),
                        ReduceOutput::Scalar(obj) => cur_scalar = Some(obj),
                    }
                }
                PlanOp::DfGetTmp { level0, col } => {
                    let df = cur_df
                        .as_ref()
                        .ok_or_else(|| PyValueError::new_err("df_get requires dataframe batch."))?
                        .clone();
                    tmp = Some(df_get_level0_column(&df, level0, col)?);
                }
                PlanOp::ReduceTmp { op, dim } => {
                    let a = tmp.take().ok_or_else(|| PyValueError::new_err("reduce_tmp: missing tmp value."))?;
                    match reduce::reduce(py, &a, Some(*dim), *op)? {
                        ReduceOutput::Array(out) => tmp = Some(out),
                        ReduceOutput::Scalar(_) => return Err(PyValueError::new_err("reduce_tmp produced scalar; not supported for df assignment.")),
                    }
                }
                PlanOp::DfSetTmp { level0, col } => {
                    let mut df = cur_df
                        .take()
                        .ok_or_else(|| PyValueError::new_err("df_set requires dataframe batch."))?;
                    let rhs = tmp.take().ok_or_else(|| PyValueError::new_err("df_set: missing tmp value."))?;
                    df_set_level0_column(&mut df, level0, col, rhs)?;
                    cur_df = Some(df);
                }
            }
        }

        if let Some(a) = cur_arr {
            return Ok(Py::new(py, PyGrumpyArray { inner: a })?.into_py(py));
        }
        if let Some(df) = cur_df {
            return Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py));
        }
        if let Some(o) = cur_scalar {
            return Ok(o);
        }
        Err(PyValueError::new_err("CompiledPlan.run produced no result."))
    }
}

fn slice_axis0_view(arr: &GrumpyArray, start: usize, stop: usize) -> PyResult<GrumpyArray> {
    if start > stop || stop > arr.len() {
        return Err(PyValueError::new_err("Slice out of bounds."));
    }
    let layout = match &arr.layout {
        Layout::ListOffset(lo) => Layout::OffsetView(OffsetView {
            offsets: lo.offsets.clone(),
            content: lo.content.clone(),
            start,
            stop,
        }),
        Layout::OffsetView(v) => {
            let s = v.start + start;
            let e = v.start + stop;
            if e > v.stop {
                return Err(PyValueError::new_err("Slice out of bounds."));
            }
            Layout::OffsetView(OffsetView {
                offsets: v.offsets.clone(),
                content: v.content.clone(),
                start: s,
                stop: e,
            })
        }
        _ => crate::layout::take_range(&arr.layout, start, stop)?,
    };
    Ok(GrumpyArray { dtype: arr.dtype, layout })
}

fn run_plan_array_rust(ops_plan: &[PlanOp], mut cur: GrumpyArray) -> PyResult<GrumpyArray> {
    for op in ops_plan {
        match op {
            PlanOp::AddScalar { value, is_int } => {
                if !ops::elementwise_scalar_inplace(&mut cur, BinOp::Add, *value, *is_int)? {
                    let rhs = scalar_like(cur.dtype, *value, *is_int)?;
                    cur = ops::elementwise(&cur, &rhs, BinOp::Add)?;
                }
            }
            PlanOp::SubScalar { value, is_int } => {
                if !ops::elementwise_scalar_inplace(&mut cur, BinOp::Sub, *value, *is_int)? {
                    let rhs = scalar_like(cur.dtype, *value, *is_int)?;
                    cur = ops::elementwise(&cur, &rhs, BinOp::Sub)?;
                }
            }
            PlanOp::MulScalar { value, is_int } => {
                if !ops::elementwise_scalar_inplace(&mut cur, BinOp::Mul, *value, *is_int)? {
                    let rhs = scalar_like(cur.dtype, *value, *is_int)?;
                    cur = ops::elementwise(&cur, &rhs, BinOp::Mul)?;
                }
            }
            PlanOp::DivScalar { value, is_int } => {
                if !ops::elementwise_scalar_inplace(&mut cur, BinOp::Div, *value, *is_int)? {
                    let rhs = scalar_like(cur.dtype, *value, *is_int)?;
                    cur = ops::elementwise(&cur, &rhs, BinOp::Div)?;
                }
            }
            PlanOp::ModScalar { value, is_int } => {
                if !ops::elementwise_scalar_inplace(&mut cur, BinOp::Mod, *value, *is_int)? {
                    let rhs = scalar_like(cur.dtype, *value, *is_int)?;
                    cur = ops::elementwise(&cur, &rhs, BinOp::Mod)?;
                }
            }
            PlanOp::MulScalarSumAll { value, is_int } => {
                if !is_int {
                    return Err(PyValueError::new_err("mul_scalar_sum_all requires int scalar."));
                }
                let s = *value as i32;
                let sum = ops::mul_scalar_sum_all_i64(&cur, s)?;
                return Err(PyValueError::new_err(format!(
                    "mul_scalar_sum_all produced scalar {sum}; array pipeline cannot consume scalars."
                )));
            }
            PlanOp::NeighborsKnnSelf { k, dim, loop_ } => {
                cur = neigh_ops::neighbors(&cur, &cur, Some(*k), None, *dim, *loop_)?;
            }
            PlanOp::ReduceCur { op, dim } => {
                match dim {
                    Some(d) => cur = reduce::reduce_array(&cur, *d, *op)?,
                    None => {
                        return Err(PyValueError::new_err(
                            "reduce without dim in array Rust plan is not supported.",
                        ));
                    }
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Rust scheduled compiled pipelines currently support scalar ops, neighbors, and reductions on arrays.",
                ));
            }
        }
    }
    Ok(cur)
}

fn df_slice_axis0_view(df: &df_ops::GrumpyDataFrame, start: usize, stop: usize) -> PyResult<df_ops::GrumpyDataFrame> {
    df.row_slice_view(start, stop)
}

fn run_plan_df_rust(ops_plan: &[PlanOp], mut cur: df_ops::GrumpyDataFrame) -> PyResult<df_ops::GrumpyDataFrame> {
    let mut tmp: Option<GrumpyArray> = None;
    for op in ops_plan {
        match op {
            PlanOp::DfGetTmp { level0, col } => {
                tmp = Some(df_get_level0_column(&cur, level0, col)?);
            }
            PlanOp::ReduceTmp { op, dim } => {
                let a = tmp.take().ok_or_else(|| PyValueError::new_err("reduce_tmp: missing tmp value."))?;
                tmp = Some(reduce::reduce_array(&a, *dim, *op)?);
            }
            PlanOp::DfSetTmp { level0, col } => {
                let rhs = tmp.take().ok_or_else(|| PyValueError::new_err("df_set: missing tmp value."))?;
                df_set_level0_column(&mut cur, level0, col, rhs)?;
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Rust scheduled dataframe pipelines currently support df_get/reduce_tmp/df_set (compiled dot-notation assignment).",
                ));
            }
        }
    }
    Ok(cur)
}

#[pyfunction]
#[pyo3(signature = (path, batch_size, drop_last, cpu, _prefetch, spec))]
fn compiled_stream_apply(
    py: Python<'_>,
    path: String,
    batch_size: usize,
    drop_last: bool,
    cpu: usize,
    _prefetch: usize,
    spec: Bound<'_, PyAny>,
) -> PyResult<PyCompiledBatchesIter> {
    if cpu < 1 {
        return Err(PyValueError::new_err("cpu must be >= 1"));
    }
    if batch_size == 0 {
        return Err(PyValueError::new_err("batch_size must be > 0"));
    }
    let plan = PyCompiledPlan::new(spec)?;
    // Try array first; if it fails, try dataframe.
    if let Ok(arr) = io_ops::load_array(py, &path) {
        let n = arr.len();
        let end = if drop_last && (n % batch_size != 0) {
            n - (n % batch_size)
        } else {
            n
        };
        let mut batches: Vec<GrumpyArray> = Vec::new();
        let mut i = 0usize;
        while i < end {
            let j = (i + batch_size).min(end);
            batches.push(slice_axis0_view(&arr, i, j)?);
            i = j;
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(cpu)
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool ({e}).")))?;
        let results: Vec<PyResult<GrumpyArray>> = pool.install(|| {
            batches
                .into_par_iter()
                .map(|b| run_plan_array_rust(&plan.ops, b))
                .collect()
        });
        let mut outs: Vec<GrumpyArray> = Vec::with_capacity(results.len());
        for r in results {
            outs.push(r?);
        }
        return Ok(PyCompiledBatchesIter { arr_batches: Some(outs), df_batches: None, pos: 0 });
    }

    let df = io_ops::load_dataframe(py, &path)?;
    let n = df.nrows();
    let end = if drop_last && (n % batch_size != 0) {
        n - (n % batch_size)
    } else {
        n
    };
    let mut batches: Vec<df_ops::GrumpyDataFrame> = Vec::new();
    let mut i = 0usize;
    while i < end {
        let j = (i + batch_size).min(end);
        batches.push(df_slice_axis0_view(&df, i, j)?);
        i = j;
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(cpu)
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool ({e}).")))?;
    let results: Vec<PyResult<df_ops::GrumpyDataFrame>> = pool.install(|| {
        batches
            .into_par_iter()
            .map(|b| run_plan_df_rust(&plan.ops, b))
            .collect()
    });
    let mut outs: Vec<df_ops::GrumpyDataFrame> = Vec::with_capacity(results.len());
    for r in results {
        outs.push(r?);
    }
    Ok(PyCompiledBatchesIter { arr_batches: None, df_batches: Some(outs), pos: 0 })
}

fn df_get_level0_column(df: &df_ops::GrumpyDataFrame, level0: &str, colname: &str) -> PyResult<GrumpyArray> {
    let schema = df.schema.as_ref().ok_or_else(|| PyValueError::new_err("Dot-notation requires a schema."))?;
    let level = *schema
        .name_to_level
        .get(level0)
        .ok_or_else(|| PyValueError::new_err("Invalid schema path."))?;
    // Find column
    let mut col: Option<GrumpyArray> = None;
    for (n, c) in df.names.iter().zip(df.cols.iter()) {
        if n == colname {
            col = Some(c.clone());
            break;
        }
    }
    let col = col.ok_or_else(|| PyValueError::new_err(format!("Unknown column '{colname}'.")))?;
    // Drop outer axes according to requested level.
    let mut layout = col.layout.clone();
    let mut drops = level;
    while drops > 0 {
        match layout {
            Layout::ListOffset(lo) => layout = *lo.content,
            Layout::OffsetView(v) => {
                let start = v.offsets[v.start] as usize;
                let end = v.offsets[v.stop] as usize;
                layout = crate::layout::take_range(v.content.as_ref(), start, end)?;
            }
            Layout::Indexed(ix) => layout = *ix.content,
            Layout::Leaf(_) => break,
            Layout::UnionScalarList(_) => return Err(PyValueError::new_err("Dot-notation not supported for union layouts.")),
        }
        drops -= 1;
    }
    Ok(GrumpyArray { dtype: col.dtype, layout })
}

fn df_set_level0_column(df: &mut df_ops::GrumpyDataFrame, level0: &str, colname: &str, rhs: GrumpyArray) -> PyResult<()> {
    let schema = df.schema.as_ref().ok_or_else(|| PyValueError::new_err("Dot-notation assignment requires a schema."))?;
    let level = *schema
        .name_to_level
        .get(level0)
        .ok_or_else(|| PyValueError::new_err("Invalid schema path."))?;
    let col_level = schema.level_for_column(colname)?;

    let rhs2 = if level == col_level {
        df.renest_rhs_for_level(level, level0, rhs)?
    } else {
        rhs
    };
    df.set_column_array(colname.to_string(), rhs2)
}

fn scalar_like(dt: DType, value: f64, is_int: bool) -> PyResult<GrumpyArray> {
    use crate::layout::{Leaf, LeafBuffer};
    use bitvec::bitvec;
    use bitvec::order::Lsb0;

    let mut leaf = Leaf::new(dt);
    leaf.len = 1;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; 1]);
    leaf.buffer = match dt {
        DType::Int8 => LeafBuffer::I8(Arc::new(vec![value as i8])),
        DType::Int16 => LeafBuffer::I16(Arc::new(vec![value as i16])),
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![value as i32])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![value as i64])),
        DType::UInt8 => LeafBuffer::U8(Arc::new(vec![value as u8])),
        DType::UInt16 => LeafBuffer::U16(Arc::new(vec![value as u16])),
        DType::UInt32 => LeafBuffer::U32(Arc::new(vec![value as u32])),
        DType::UInt64 => LeafBuffer::U64(Arc::new(vec![value as u64])),
        DType::Float16 => LeafBuffer::F16(Arc::new(vec![half::f16::from_f64(value).to_bits()])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![value as f32])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![value])),
        DType::Bool => return Err(PyValueError::new_err("Compiled scalar ops do not support dtype=bool.")),
        DType::Char | DType::String => return Err(PyValueError::new_err("Compiled scalar ops require numeric dtype.")),
    };
    if is_int {
        // ok; just a hint today
    }
    Ok(GrumpyArray { dtype: dt, layout: Layout::Leaf(leaf) })
}

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

    fn astype(&self, py: Python<'_>, dtype: PyDType) -> PyResult<Self> {
        let target = dtype.dt;
        let obj = self.inner.to_py_list(py)?;
        let bound = obj.bind(py);
        let arr = build_array(py, &bound, target)?;
        Ok(Self { inner: arr })
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

    fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Fast leaf (1D) export: create a typed NumPy array and memcpy from the contiguous leaf buffer.
        // This avoids materializing Python lists for common 1D results (e.g. fancy gather).
        if let Layout::Leaf(leaf) = &self.inner.layout {
            if !leaf.has_nulls {
                if let Some(arr) = leaf_to_numpy_1d_typed(py, leaf, self.inner.dtype)? {
                    return Ok(arr);
                }
            }
        }

        let np = PyModule::import_bound(py, "numpy")?;
        let lst = self.inner.to_py_list(py)?;

        // Typed path: if there are no unions and no nulls, ask NumPy to materialize a typed array.
        // (We still need to validate rectangularity; NumPy will error if the nesting is ragged.)
        if layout_all_valid_no_union(&self.inner.layout) {
            if let Some((dtype_obj, expected_name)) = numpy_dtype(&np, self.inner.dtype)? {
                let kwargs = [("dtype", dtype_obj)].into_py_dict_bound(py);
                if let Ok(arr) = np.call_method("array", (lst.clone_ref(py),), Some(&kwargs)) {
                    // Ensure NumPy actually produced the dtype we requested (and not object).
                    let dtype = arr.getattr("dtype")?;
                    let name: String = dtype.getattr("name")?.extract()?;
                    if name == expected_name {
                        return Ok(arr.into());
                    }
                }
            }
        }

        // Fallback: object array via numpy.array(py_list, dtype=object)
        let dtype_obj = np.getattr("object_")?;
        let kwargs = [("dtype", dtype_obj)].into_py_dict_bound(py);
        let arr = np.call_method("array", (lst,), Some(&kwargs))?;
        Ok(arr.into())
    }

    fn copy(&self) -> Self {
        let mut inner = self.inner.clone();
        inner.uniquify_buffers();
        Self { inner }
    }

    #[pyo3(signature = (dim=None))]
    fn sum(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, dim, ReduceOp::Sum)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    fn mean(&self, py: Python<'_>, dim: isize) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, Some(dim), ReduceOp::Mean)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    fn min(&self, py: Python<'_>, dim: isize) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, Some(dim), ReduceOp::Min)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    fn max(&self, py: Python<'_>, dim: isize) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, Some(dim), ReduceOp::Max)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    fn ptp(&self, py: Python<'_>, dim: isize) -> PyResult<PyObject> {
        match reduce::reduce(py, &self.inner, Some(dim), ReduceOp::Ptp)? {
            ReduceOutput::Scalar(x) => Ok(x),
            ReduceOutput::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
        }
    }

    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Sin)? })
    }

    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Cos)? })
    }

    fn tan(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Tan)? })
    }

    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Exp)? })
    }

    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Log)? })
    }

    fn log10(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Log10)? })
    }

    fn log2(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Log2)? })
    }

    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Sqrt)? })
    }

    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Abs)? })
    }

    fn sign(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Sign)? })
    }

    fn floor(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Floor)? })
    }

    fn ceil(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Ceil)? })
    }

    fn round(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Round)? })
    }

    fn reciprocal(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Reciprocal)? })
    }

    fn angle(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: unary_ops::unary(py, &self.inner, unary_ops::UnaryOp::Angle)? })
    }

    fn isnan(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::predicate(py, &self.inner, cmp_ops::PredOp::IsNan)? })
    }

    fn isfinite(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::predicate(py, &self.inner, cmp_ops::PredOp::IsFinite)? })
    }

    fn isinf(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::predicate(py, &self.inner, cmp_ops::PredOp::IsInf)? })
    }

    fn equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Eq)? })
    }

    fn not_equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Ne)? })
    }

    fn less(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Lt)? })
    }

    fn less_equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Le)? })
    }

    fn greater(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Gt)? })
    }

    fn greater_equal(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::compare(py, &self.inner, &other.inner, cmp_ops::CmpOp::Ge)? })
    }

    fn logical_and(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_bin(py, &self.inner, &other.inner, cmp_ops::LogicOp::And)? })
    }

    fn logical_or(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_bin(py, &self.inner, &other.inner, cmp_ops::LogicOp::Or)? })
    }

    fn logical_xor(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_bin(py, &self.inner, &other.inner, cmp_ops::LogicOp::Xor)? })
    }

    fn logical_not(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self { inner: cmp_ops::logical_not(py, &self.inner)? })
    }

    fn var(&self, py: Python<'_>, dim: isize, ddof: isize) -> PyResult<Self> {
        Ok(Self { inner: stats_ops::var(py, &self.inner, dim, ddof, false)? })
    }

    fn std(&self, py: Python<'_>, dim: isize, ddof: isize) -> PyResult<Self> {
        Ok(Self { inner: stats_ops::std(py, &self.inner, dim, ddof, false)? })
    }

    fn nanvar(&self, py: Python<'_>, dim: isize, ddof: isize) -> PyResult<Self> {
        Ok(Self { inner: stats_ops::var(py, &self.inner, dim, ddof, true)? })
    }

    fn nanstd(&self, py: Python<'_>, dim: isize, ddof: isize) -> PyResult<Self> {
        Ok(Self { inner: stats_ops::std(py, &self.inner, dim, ddof, true)? })
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

    #[pyo3(signature = (dim=None))]
    fn sort(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<Self> {
        let dim = dim.unwrap_or(-1);
        Ok(Self { inner: ss_ops::sort_axis(py, &self.inner, dim)? })
    }

    #[pyo3(signature = (dim=None))]
    fn argsort(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<Self> {
        let dim = dim.unwrap_or(-1);
        Ok(Self { inner: ss_ops::argsort_axis(py, &self.inner, dim)? })
    }

    #[pyo3(signature = (dim=None))]
    fn argmax(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::ArgMax)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::ArgMax)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (dim=None))]
    fn argmin(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::ArgMin)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::ArgMin)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (dim=None))]
    fn nanargmax(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::NanArgMax)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::NanArgMax)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (dim=None))]
    fn nanargmin(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<PyObject> {
        match dim {
            None => match ss_ops::argreduce(py, &self.inner, ss_ops::ArgOp::NanArgMin)? {
                ss_ops::ArgOut::Scalar(o) => Ok(o),
            },
            Some(d) => {
                let out = ss_ops::argreduce_axis_array(py, &self.inner, d, ss_ops::ArgOp::NanArgMin)?;
                Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
            }
        }
    }

    #[pyo3(signature = (kth, dim=None))]
    fn partition(&self, py: Python<'_>, kth: usize, dim: Option<isize>) -> PyResult<Self> {
        match dim {
            None => Ok(Self { inner: ss_ops::partition(py, &self.inner, kth)? }),
            Some(1) | Some(-1) => Ok(Self { inner: ss_ops::partition_dim1(py, &self.inner, kth)? }),
            Some(_) => Err(PyValueError::new_err("partition: only dim=1 is implemented for 2D arrays.")),
        }
    }

    #[pyo3(signature = (kth, dim=None))]
    fn argpartition(&self, py: Python<'_>, kth: usize, dim: Option<isize>) -> PyResult<Self> {
        match dim {
            None => Ok(Self { inner: ss_ops::argpartition(py, &self.inner, kth)? }),
            Some(1) | Some(-1) => Ok(Self { inner: ss_ops::argpartition_dim1(py, &self.inner, kth)? }),
            Some(_) => Err(PyValueError::new_err("argpartition: only dim=1 is implemented for 2D arrays.")),
        }
    }

    /// Kernel-only: mul for 2D rectangular int32 arrays, returning an i64 checksum.
    /// This avoids allocating an output array so benchmarks can measure compute only.
    fn _mul2d_i32_sum_i64(&self, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 || other.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_mul2d_i32_sum_i64 requires int32 arrays."));
        }
        let (a_off, a_vals) = rect2d_i32_view(&self.inner.layout)?;
        let (b_off, b_vals) = rect2d_i32_view(&other.inner.layout)?;
        if a_off != b_off {
            return Err(PyValueError::new_err("Rectangular shapes differ."));
        }
        Ok(sum_i32_mul_neon(a_vals, b_vals))
    }

    fn _add2d_i32_sum_i64(&self, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 || other.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_add2d_i32_sum_i64 requires int32 arrays."));
        }
        let (a_off, a_vals) = rect2d_i32_view(&self.inner.layout)?;
        let (b_off, b_vals) = rect2d_i32_view(&other.inner.layout)?;
        if a_off != b_off {
            return Err(PyValueError::new_err("Rectangular shapes differ."));
        }
        Ok(sum_i32_add_neon(a_vals, b_vals))
    }

    /// Sum all elements for 2D pure list-chain int32 arrays (leaf sum). Used for benchmarking
    /// full elementwise ops without converting to Python/NumPy.
    fn _sum2d_i32_i64(&self) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_sum2d_i32_i64 requires int32 arrays."));
        }
        let (_off, vals) = rect2d_i32_view(&self.inner.layout)?;
        Ok(sum_i32_to_i64_neon(vals))
    }

    /// Kernel-only: sum over dim=1 for 2D rectangular int32 arrays, returning a checksum (sum of row sums).
    fn _sum2d_dim1_i32_sum_i64(&self) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_sum2d_dim1_i32_sum_i64 requires int32 array."));
        }
        let (off, vals) = rect2d_i32_view(&self.inner.layout)?;
        if off.len() < 2 {
            return Ok(0);
        }
        let nrows = off.len() - 1;
        let mut acc: i64 = 0;
        for i in 0..nrows {
            let s = off[i] as usize;
            let e = off[i + 1] as usize;
            acc = acc.wrapping_add(sum_i32_to_i64_neon(&vals[s..e]));
        }
        Ok(acc)
    }

    /// Kernel-only: mean over dim=1 for 2D rectangular int32 arrays, returning a checksum (sum of row means).
    fn _mean2d_dim1_i32_sum_f64(&self) -> PyResult<f64> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_mean2d_dim1_i32_sum_f64 requires int32 array."));
        }
        let (off, vals) = rect2d_i32_view(&self.inner.layout)?;
        if off.len() < 2 {
            return Ok(0.0);
        }
        let nrows = off.len() - 1;
        let mut acc: f64 = 0.0;
        for i in 0..nrows {
            let s = off[i] as usize;
            let e = off[i + 1] as usize;
            let len = (e - s) as f64;
            if len == 0.0 {
                continue;
            }
            let sum = sum_i32_to_i64_neon(&vals[s..e]) as f64;
            acc += sum / len;
        }
        Ok(acc)
    }

    /// Full op benchmark helper: compute (self * other).sum() for 2D int32 arrays via Grumpy's
    /// own elementwise implementation (exercises rectangular fast-path vs generic ragged path).
    fn _mul2d_i32_sum_via_op_i64(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        let _ = py;
        let out = ops::elementwise(&self.inner, &other.inner, BinOp::Mul)?;
        if out.dtype != DType::Int32 {
            return Err(PyValueError::new_err("Internal error: expected int32 output."));
        }
        let (_off, vals) = rect2d_i32_view(&out.layout)?;
        Ok(sum_i32_to_i64_neon(vals))
    }

    fn _add2d_i32_sum_via_op_i64(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        let _ = py;
        let out = ops::elementwise(&self.inner, &other.inner, BinOp::Add)?;
        if out.dtype != DType::Int32 {
            return Err(PyValueError::new_err("Internal error: expected int32 output."));
        }
        let (_off, vals) = rect2d_i32_view(&out.layout)?;
        Ok(sum_i32_to_i64_neon(vals))
    }

    /// Internal kernel: gather 2D fancy indices and return an i64 checksum (sum) without allocating output.
    /// This exists to benchmark just the gather kernel (no to_list/to_numpy overhead).
    fn _gather2d_sum_i64(&self, _py: Python<'_>, rows: Bound<'_, PyAny>, cols: Bound<'_, PyAny>) -> PyResult<i64> {
        if !self.inner.is_pure_list_chain() {
            return Err(PyValueError::new_err("gather kernel only supports pure list-chain arrays."));
        }
        let r = rows
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("rows must be a 1D NumPy int64 array."))?;
        let c = cols
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("cols must be a 1D NumPy int64 array."))?;
        let rs = r.as_slice()?;
        let cs = c.as_slice()?;
        gather_2d_fancy_sum_i64(&self.inner.layout, rs, cs, self.inner.dtype)
    }

    /// Internal kernel: scatter 2D fancy indices from NumPy arrays without Python conversions (int32 only).
    fn _scatter2d_i32(
        &mut self,
        _py: Python<'_>,
        rows: Bound<'_, PyAny>,
        cols: Bound<'_, PyAny>,
        values: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_scatter2d_i32 requires dtype=int32."));
        }
        let r = rows
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("rows must be a 1D NumPy int64 array."))?;
        let c = cols
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("cols must be a 1D NumPy int64 array."))?;
        let v = values
            .extract::<PyReadonlyArray1<'_, i32>>()
            .map_err(|_| PyValueError::new_err("values must be a 1D NumPy int32 array."))?;
        let rs = r.as_slice()?;
        let cs = c.as_slice()?;
        let vs = v.as_slice()?;
        crate::layout::scatter_2d_fancy_i32(&mut self.inner.layout, rs, cs, vs)?;
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, index: Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Some(out) = fast_getitem(py, &self.inner, &index)? {
            return Ok(out);
        }

        // Fallback (correctness): Python-list based indexing.
        let base = self.inner.to_py_list(py)?;
        let out = if index.downcast::<PyTuple>().is_ok() {
            getitem_coordinate(py, &base.bind(py), &index, self.inner.dtype)?
        } else if crate::dtype::is_sequence_like(py, &index)? {
            getitem_array_indexing(py, &base.bind(py), &index, self.inner.dtype)?
        } else {
            // scalar int or slice = coordinate indexing on dim 0
            getitem_coordinate(py, &base.bind(py), &index, self.inner.dtype)?
        };
        wrap_result(py, out, self.inner.dtype)
    }

    fn __setitem__(&mut self, py: Python<'_>, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if fast_setitem(py, &mut self.inner, &index, &value)? {
            return Ok(());
        }

        // Fallback (correctness): mutate Python list and rebuild.
        let base = self.inner.to_py_list(py)?;
        let base_b = base.bind(py);
        if index.downcast::<PyTuple>().is_ok() {
            setitem_coordinate(py, &base_b, &index, &value)?;
        } else if crate::dtype::is_sequence_like(py, &index)? {
            setitem_array(py, &base_b, &index, &value)?;
        } else {
            setitem_coordinate(py, &base_b, &index, &value)?;
        }
        let rebuilt = build_array(py, &base_b, self.inner.dtype)?;
        self.inner = rebuilt;
        Ok(())
    }

    #[pyo3(signature = (dim=None, but=None))]
    fn flatten(
        &self,
        py: Python<'_>,
        dim: Option<Bound<'_, PyAny>>,
        but: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Fast path: pure list-chain + default flatten (all axes) => just return the leaf view.
        // This matches the Awkward-style "avoid materialization": we reuse the contiguous leaf buffers
        // (Arc + copy-on-write) and only change metadata.
        if dim.is_none() && but.is_none() && self.inner.is_pure_list_chain() {
            if let Ok(leaf) = find_leaf_fast(&self.inner.layout) {
                return Ok(Self {
                    inner: GrumpyArray {
                        dtype: self.inner.dtype,
                        layout: Layout::Leaf(leaf.clone()),
                    },
                });
            }
        }

        let data = self.inner.to_py_list(py)?;
        let data_b = data.bind(py);
        let max_depth = max_list_depth(py, &data_b)?;

        let dim_is_none = dim.is_none();
        let mut dims = if let Some(ref d) = dim {
            parse_dims(py, d, max_depth)?
        } else {
            Vec::new()
        };

        if let Some(b) = but {
            let excluded = parse_dims(py, &b, max_depth)?;
            let mut set = std::collections::BTreeSet::new();
            for x in 0..max_depth {
                set.insert(x as isize);
            }
            for e in excluded {
                set.remove(&e);
            }
            dims = set.into_iter().collect();
        } else if dim_is_none {
            // default: flatten all axes
            dims = (0..max_depth).map(|x| x as isize).collect();
        }

        let mut axes = std::collections::BTreeSet::<usize>::new();
        for d in dims {
            axes.insert(d as usize);
        }
        let flat_elems = flatten_collect(py, &data_b, 0, &axes)?;
        let cur = if axes.contains(&0) {
            let out_list = pyo3::types::PyList::empty_bound(py);
            for e in flat_elems {
                out_list.append(e)?;
            }
            out_list.into_py(py)
        } else {
            if flat_elems.len() != 1 {
                return Err(PyValueError::new_err(
                    "Internal error: flatten root expected a single list result.",
                ));
            }
            flat_elems[0].clone_ref(py)
        };

        let out = build_array(py, &cur.bind(py), self.inner.dtype)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (sizes, dim=0))]
    fn unflatten(&self, py: Python<'_>, sizes: Bound<'_, PyAny>, dim: isize) -> PyResult<Self> {
        // Fast path: unflatten a 1D leaf along dim=0 using a 1D sizes vector (Python list or GrumpyArray[int64]).
        // Produces a ListOffset view sharing the leaf buffer (Arc + copy-on-write).
        if self.inner.is_pure_list_chain() && dim == 0 {
            if let Layout::Leaf(leaf) = &self.inner.layout {
                if let Some(szs) = sizes_to_vec_usize_fast(py, &sizes)? {
                    let mut offsets: Vec<i64> = Vec::with_capacity(szs.len() + 1);
                    offsets.push(0);
                    let mut acc: i64 = 0;
                    for s in szs {
                        acc += s as i64;
                        offsets.push(acc);
                    }
                    if acc as usize != leaf.len {
                        return Err(PyValueError::new_err(
                            "unflatten sizes must sum to the number of elements.",
                        ));
                    }
                    return Ok(Self {
                        inner: GrumpyArray {
                            dtype: self.inner.dtype,
                            layout: Layout::ListOffset(crate::layout::ListOffset {
                                offsets: std::sync::Arc::new(offsets),
                                content: Box::new(Layout::Leaf(leaf.clone())),
                            }),
                        },
                    });
                }
            }
        }

        let data = self.inner.to_py_list(py)?;
        let data_b = data.bind(py);
        let max_depth = max_list_depth(py, &data_b)?;
        let dim_u = normalize_dim(dim, max_depth as isize)?;

        let sizes_obj = sizes_to_list_any(py, &sizes)?;
        let out_py = unflatten_rec(py, &data_b, &sizes_obj.bind(py), dim_u as usize)?;
        let out = build_array(py, &out_py.bind(py), self.inner.dtype)?;
        Ok(Self { inner: out })
    }

    fn __add__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Add).map(|inner| Self { inner })
    }

    fn __sub__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Sub).map(|inner| Self { inner })
    }

    fn __mul__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Mul).map(|inner| Self { inner })
    }

    fn __truediv__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Div).map(|inner| Self { inner })
    }

    fn __mod__(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Mod).map(|inner| Self { inner })
    }

    fn remainder(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Remainder).map(|inner| Self { inner })
    }

    fn mod_(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Self> {
        apply_elementwise_binop(py, &self.inner, &other, BinOp::Mod).map(|inner| Self { inner })
    }
}

fn coerce_to_array(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype_hint: DType) -> PyResult<GrumpyArray> {
    if let Ok(arr) = obj.extract::<PyRef<'_, PyGrumpyArray>>() {
        return Ok(arr.inner.clone());
    }
    // Scalars / python lists: build GrumpyArray with the dtype of lhs.
    // This enables scalar broadcasting like x * 2.
    build_array(py, obj, dtype_hint)
}

fn try_extract_broadcast_scalar(obj: &Bound<'_, PyAny>, dtype: DType) -> Option<(f64, bool)> {
    if obj.is_none() {
        return None;
    }
    if obj.is_instance_of::<PyBool>() {
        let v = obj.extract::<bool>().ok()?;
        return Some((if v { 1.0 } else { 0.0 }, true));
    }
    if obj.is_instance_of::<PyInt>() {
        let v = obj.extract::<i64>().ok()? as f64;
        return Some((v, true));
    }
    if obj.is_instance_of::<PyFloat>() {
        let v = obj.extract::<f64>().ok()?;
        return Some((v, false));
    }
    let _ = dtype;
    None
}

fn apply_elementwise_binop(
    py: Python<'_>,
    lhs: &GrumpyArray,
    other: &Bound<'_, PyAny>,
    op: BinOp,
) -> PyResult<GrumpyArray> {
    if let Ok(rhs) = other.extract::<PyRef<'_, PyGrumpyArray>>() {
        return ops::elementwise(lhs, &rhs.inner, op);
    }
    if let Some((value, is_int)) = try_extract_broadcast_scalar(other, lhs.dtype) {
        return ops::elementwise_with_scalar(lhs, op, value, is_int);
    }
    let rhs = coerce_to_array(py, other, lhs.dtype)?;
    ops::elementwise(lhs, &rhs, op)
}

#[pymethods]
impl PyGrumpyDataFrame {
    fn __repr__(&self) -> String {
        format!("grumpy.dataframe({})", self.inner.names.join(", "))
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner.to_pydict(py)
    }

    fn max(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner.max_all(py)
    }

    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Column selection by string or tuple of strings.
        if let Ok(s) = key.extract::<String>() {
            let df = self.inner.column_subset(&[s])?;
            return Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py));
        }
        if let Ok(tup) = key.downcast::<PyTuple>() {
            let mut names: Vec<String> = Vec::new();
            for i in 0..tup.len() {
                names.push(tup.get_item(i)?.extract::<String>().map_err(|_| {
                    PyValueError::new_err("Column selection must be strings.")
                })?);
            }
            let df = self.inner.column_subset(&names)?;
            return Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py));
        }

        // Row selection: int/slice/bool mask.
        let idx = df_ops::parse_row_index(py, &key, self.inner.nrows())?;
        let df = self.inner.row_select_indexed(idx)?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn __setitem__(&mut self, py: Python<'_>, key: String, value: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(arr) = value.extract::<PyRef<'_, PyGrumpyArray>>() {
            self.inner.set_column_array(key, arr.inner.clone())
        } else {
            // Infer dtype from value and build.
            let inferred = crate::dtype::infer_dtype(py, &value)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = crate::dtype::inferclass_to_dtype(inferred);
            self.inner.set_column(py, key, &value, Some(dt))
        }
    }

    fn __getattr__(slf: PyRef<'_, Self>, py: Python<'_>, name: String) -> PyResult<PyObject> {
        // If this is a schema level, return accessor.
        if let Some(schema) = &slf.inner.schema {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let parent: Py<PyGrumpyDataFrame> = slf.into_py(py).extract(py)?;
                    let acc = PyDataFrameAccessor { parent, path: vec![name] };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        // Otherwise, treat as column name and return fully flattened array (default dot-notation behavior).
        for (n, c) in slf.inner.names.iter().zip(slf.inner.cols.iter()) {
            if n == &name {
                // default df.col: fully flatten all axes
                let leaf = crate::py_api::find_leaf_fast(&c.layout)?;
                let out = PyGrumpyArray { inner: GrumpyArray { dtype: c.dtype, layout: Layout::Leaf(leaf.clone()) } };
                return Ok(Py::new(py, out)?.into_py(py));
            }
        }
        Err(PyValueError::new_err(format!("Unknown attribute '{}'.", name)))
    }
}

#[pymethods]
impl PyDataFrameAccessor {
    fn __getattr__(&self, py: Python<'_>, name: String) -> PyResult<PyObject> {
        let df_ref = self.parent.borrow(py);
        // Chain schema levels if applicable.
        if let Some(schema) = &df_ref.inner.schema {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let mut p = self.path.clone();
                    p.push(name);
                    let acc = PyDataFrameAccessor { parent: self.parent.clone_ref(py), path: p };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        // Column access under a path: currently require path to start at a valid schema level.
        let schema = df_ref.inner.schema.as_ref().ok_or_else(|| PyValueError::new_err("Dot-notation requires a schema."))?;
        let level0 = *schema
            .name_to_level
            .get(&self.path[0])
            .ok_or_else(|| PyValueError::new_err("Invalid schema path."))?;

        // Find column
        let mut col: Option<GrumpyArray> = None;
        for (n, c) in df_ref.inner.names.iter().zip(df_ref.inner.cols.iter()) {
            if n == &name {
                col = Some(c.clone());
                break;
            }
        }
        let col = col.ok_or_else(|| PyValueError::new_err(format!("Unknown column '{}'.", name)))?;

        // Drop outer axes according to requested level0: molecule->drop0, residue->drop1, atom->drop2, ...
        let mut layout = col.layout.clone();
        let mut drops = level0;
        while drops > 0 {
            match layout {
                Layout::ListOffset(lo) => layout = *lo.content,
                Layout::OffsetView(v) => {
                    let start = v.offsets[v.start] as usize;
                    let end = v.offsets[v.stop] as usize;
                    layout = crate::layout::take_range(v.content.as_ref(), start, end)?;
                }
                Layout::Indexed(ix) => layout = *ix.content,
                Layout::Leaf(_) => break,
                Layout::UnionScalarList(_) => return Err(PyValueError::new_err("Dot-notation not supported for union layouts.")),
            }
            drops -= 1;
        }
        let out = PyGrumpyArray { inner: GrumpyArray { dtype: col.dtype, layout } };
        Ok(Py::new(py, out)?.into_py(py))
    }

    fn __setattr__(&mut self, py: Python<'_>, name: String, value: Bound<'_, PyAny>) -> PyResult<()> {
        // Only handle setting columns; allow normal attribute sets for internal fields.
        if name == "parent" || name == "path" {
            return Err(PyValueError::new_err("Cannot overwrite accessor internals."));
        }
        let mut df_mut = self.parent.borrow_mut(py);
        let schema = df_mut.inner.schema.as_ref().ok_or_else(|| PyValueError::new_err("Dot-notation assignment requires a schema."))?;
        let level0 = *schema
            .name_to_level
            .get(&self.path[0])
            .ok_or_else(|| PyValueError::new_err("Invalid schema path."))?;
        let col_level = schema.level_for_column(&name)?;

        // Build RHS array from Python value with inferred dtype.
        let arr = if let Ok(g) = value.extract::<PyRef<'_, PyGrumpyArray>>() {
            g.inner.clone()
        } else {
            let inferred = crate::dtype::infer_dtype(py, &value)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = crate::dtype::inferclass_to_dtype(inferred);
            crate::layout::build_array(py, &value, dt)?
        };

        // If setting at schema level `level0` for a column belonging to the same schema level,
        // accept a flat-by-level RHS (outer len == total elements at that level) and re-nest it
        // back to axis-0 using canonical offsets at all intermediate levels.
        let arr2 = if level0 >= 1 && level0 == col_level {
            if arr.len() == df_mut.inner.nrows() {
                arr
            } else {
                let canon_off_level = df_mut
                    .inner
                    .canon
                    .offsets
                    .get(level0)
                    .and_then(|x| x.as_ref())
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Cannot re-nest: missing canonical offsets for schema level {level0} ('{}').",
                            self.path[0]
                        ))
                    })?;
                let total = *canon_off_level.last().unwrap() as usize;
                if arr.len() != total {
                    return Err(PyValueError::new_err(format!(
                        "Dot-notation assignment at '{}': RHS must have outer length {total} (total elements at that level) or {} (axis-0 length), but has {}.",
                        self.path[0],
                        df_mut.inner.nrows(),
                        arr.len()
                    )));
                }
                let mut cur = arr.layout;
                for lev in (1..=level0).rev() {
                    let canon_off = df_mut
                        .inner
                        .canon
                        .offsets
                        .get(lev)
                        .and_then(|x| x.as_ref())
                        .ok_or_else(|| PyValueError::new_err(format!("Cannot re-nest: missing canonical offsets for schema level {lev}.")))?
                        .clone();
                    cur = Layout::ListOffset(crate::layout::ListOffset {
                        offsets: Arc::new(canon_off),
                        content: Box::new(cur),
                    });
                }
                GrumpyArray { dtype: arr.dtype, layout: cur }
            }
        } else {
            arr
        };

        // Delegate to dataframe set column array (schema validation happens there).
        df_mut.inner.set_column_array(name, arr2)
    }
}
fn sizes_to_vec_usize_fast(_py: Python<'_>, sizes: &Bound<'_, PyAny>) -> PyResult<Option<Vec<usize>>> {
    // Accept:
    // - Python sequence of ints
    // - GrumpyArray[int64] leaf (e.g. shape(dim=1))
    if let Ok(seq) = sizes.downcast::<pyo3::types::PySequence>() {
        let n = seq.len()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let v = seq.get_item(i)?;
            let x: i64 = v.extract()?;
            if x < 0 {
                return Err(PyValueError::new_err("sizes must be non-negative."));
            }
            out.push(x as usize);
        }
        return Ok(Some(out));
    }
    if let Ok(gr_sizes) = sizes.extract::<PyRef<'_, PyGrumpyArray>>() {
        if gr_sizes.inner.dtype != DType::Int64 {
            return Ok(None);
        }
        if let Layout::Leaf(leaf) = &gr_sizes.inner.layout {
            if leaf.has_nulls {
                return Err(PyValueError::new_err("sizes array must not contain nulls."));
            }
            if let crate::layout::LeafBuffer::I64(v) = &leaf.buffer {
                let mut out = Vec::with_capacity(leaf.len);
                for &x in v.as_slice().iter().take(leaf.len) {
                    if x < 0 {
                        return Err(PyValueError::new_err("sizes must be non-negative."));
                    }
                    out.push(x as usize);
                }
                return Ok(Some(out));
            }
        }
    }
    Ok(None)
}

fn rect2d_i32_view<'a>(layout: &'a Layout) -> PyResult<(&'a [i64], &'a [i32])> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        Layout::OffsetView(v) => {
            // treat as a view into an existing offset buffer
            return Ok((
                v.offsets.as_slice(),
                match v.content.as_ref() {
                    Layout::Leaf(leaf) => match &leaf.buffer {
                        crate::layout::LeafBuffer::I32(buf) => buf.as_slice(),
                        _ => return Err(PyValueError::new_err("Expected int32 leaf buffer.")),
                    },
                    _ => return Err(PyValueError::new_err("Expected leaf values.")),
                },
            ));
        }
        _ => return Err(PyValueError::new_err("Expected 2D list layout.")),
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf values.")),
    };
    if leaf.has_nulls {
        return Err(PyValueError::new_err("Kernel requires all-valid leaf (no nulls)."));
    }
    let vals = match &leaf.buffer {
        crate::layout::LeafBuffer::I32(v) => v.as_slice(),
        _ => return Err(PyValueError::new_err("Expected int32 leaf buffer.")),
    };
    Ok((lo.offsets.as_slice(), vals))
}

#[cfg(target_arch = "aarch64")]
fn sum_i32_to_i64_neon(a: &[i32]) -> i64 {
    crate::kernels::sum_i32_to_i64(a)
}

#[cfg(not(target_arch = "aarch64"))]
fn sum_i32_to_i64_neon(a: &[i32]) -> i64 {
    crate::kernels::sum_i32_to_i64(a)
}

#[cfg(target_arch = "aarch64")]
fn sum_i32_mul_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_mul_to_i64(a, b)
}

#[cfg(not(target_arch = "aarch64"))]
fn sum_i32_mul_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_mul_to_i64(a, b)
}

#[cfg(target_arch = "aarch64")]
fn sum_i32_add_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_add_to_i64(a, b)
}

#[cfg(not(target_arch = "aarch64"))]
fn sum_i32_add_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_add_to_i64(a, b)
}

fn leaf_to_numpy_1d_typed(
    py: Python<'_>,
    leaf: &crate::layout::Leaf,
    dt: DType,
) -> PyResult<Option<PyObject>> {
    let n = leaf.len;
    match dt {
        DType::Int32 => Ok(Some(bytes_to_numpy_1d::<i32>(py, leaf.buffer.as_bytes(), n)?)),
        DType::Int64 => Ok(Some(bytes_to_numpy_1d::<i64>(py, leaf.buffer.as_bytes(), n)?)),
        DType::Float32 => Ok(Some(bytes_to_numpy_1d::<f32>(py, leaf.buffer.as_bytes(), n)?)),
        DType::Float64 => Ok(Some(bytes_to_numpy_1d::<f64>(py, leaf.buffer.as_bytes(), n)?)),
        // extend later (uint/bool/float16) once benchmarks show it's needed
        _ => Ok(None),
    }
}

fn bytes_to_numpy_1d<T: numpy::Element>(py: Python<'_>, bytes: &[u8], n: usize) -> PyResult<PyObject> {
    let expected = n * std::mem::size_of::<T>();
    if bytes.len() != expected {
        return Err(PyValueError::new_err("Internal error: leaf byte size mismatch."));
    }

    let arr = PyArray1::<T>::zeros_bound(py, n, false);
    unsafe {
        let dst = arr.data() as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    Ok(arr.into_py(py))
}

fn sizes_to_list_any(py: Python<'_>, sizes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    // If sizes is a GrumpyArray, use its to_list.
    if let Ok(arr) = sizes.extract::<PyRef<'_, PyGrumpyArray>>() {
        return arr.inner.to_py_list(py);
    }
    Ok(sizes.clone().unbind())
}

fn max_list_depth(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if crate::dtype::is_sequence_like(py, obj)? {
        let seq = obj.downcast::<pyo3::types::PySequence>()?;
        let mut m = 0usize;
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            m = m.max(max_list_depth(py, &it)?);
        }
        Ok(m + 1)
    } else {
        Ok(0)
    }
}

fn normalize_dim(dim: isize, max_depth: isize) -> PyResult<usize> {
    let mut d = dim;
    if d < 0 {
        d += max_depth;
    }
    if d < 0 || d > max_depth {
        return Err(PyValueError::new_err("dim out of range."));
    }
    Ok(d as usize)
}

fn parse_dims(py: Python<'_>, obj: &Bound<'_, PyAny>, max_depth: usize) -> PyResult<Vec<isize>> {
    let md = max_depth as isize;
    if let Ok(i) = obj.extract::<isize>() {
        let d = normalize_dim(i, md)? as isize;
        return Ok(vec![d]);
    }
    if crate::dtype::is_sequence_like(py, obj)? {
        let seq = obj.downcast::<pyo3::types::PySequence>()?;
        let mut out = Vec::with_capacity(seq.len()? as usize);
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            let v = it.extract::<isize>()?;
            out.push(normalize_dim(v, md)? as isize);
        }
        return Ok(out);
    }
    Err(PyValueError::new_err("dim/but must be an int or a sequence of ints."))
}

fn flatten_collect(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    level: usize,
    axes_to_remove: &std::collections::BTreeSet<usize>,
) -> PyResult<Vec<PyObject>> {
    if !crate::dtype::is_sequence_like(py, obj)? {
        return Ok(vec![obj.clone().unbind()]);
    }
    let seq = obj.downcast::<pyo3::types::PySequence>()?;
    if axes_to_remove.contains(&level) {
        // Remove this list level: splice children into parent.
        let mut out: Vec<PyObject> = Vec::new();
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            out.extend(flatten_collect(py, &it, level + 1, axes_to_remove)?);
        }
        Ok(out)
    } else {
        // Keep this list level, but allow deeper levels to be removed (splicing into this list).
        let out_list = pyo3::types::PyList::empty_bound(py);
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            for e in flatten_collect(py, &it, level + 1, axes_to_remove)? {
                out_list.append(e)?;
            }
        }
        Ok(vec![out_list.into_py(py)])
    }
}

fn unflatten_rec(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    dim: usize,
) -> PyResult<PyObject> {
    if dim == 0 {
        let data_seq = data.downcast::<pyo3::types::PySequence>().map_err(|_| {
            PyValueError::new_err("unflatten(dim=0) requires a sequence of values.")
        })?;
        let sizes_seq = sizes.downcast::<pyo3::types::PySequence>().map_err(|_| {
            PyValueError::new_err("sizes must be a sequence of integers.")
        })?;
        let total = data_seq.len()? as usize;
        let out = pyo3::types::PyList::empty_bound(py);
        let mut pos = 0usize;
        for i in 0..sizes_seq.len()? {
            let s = sizes_seq.get_item(i as usize)?.extract::<isize>()?;
            if s < 0 {
                return Err(PyValueError::new_err("sizes entries must be non-negative."));
            }
            let s = s as usize;
            if pos + s > total {
                return Err(PyValueError::new_err(
                    "sizes do not sum to the number of elements being unflattened.",
                ));
            }
            let chunk = pyo3::types::PyList::empty_bound(py);
            for j in 0..s {
                chunk.append(data_seq.get_item((pos + j) as usize)?)?;
            }
            pos += s;
            out.append(chunk)?;
        }
        if pos != total {
            return Err(PyValueError::new_err(
                "sizes do not sum to the number of elements being unflattened.",
            ));
        }
        return Ok(out.into());
    }

    let data_seq = data.downcast::<pyo3::types::PySequence>().map_err(|_| {
        PyValueError::new_err("unflatten requires list-like structure at the target axis.")
    })?;
    let sizes_seq = sizes.downcast::<pyo3::types::PySequence>().map_err(|_| {
        PyValueError::new_err("sizes must match the structure of the array along outer axes.")
    })?;
    if data_seq.len()? != sizes_seq.len()? {
        return Err(PyValueError::new_err(
            "sizes must have the same outer length as the array along axes above dim.",
        ));
    }
    let out = pyo3::types::PyList::empty_bound(py);
    for i in 0..data_seq.len()? {
        let sub_data = data_seq.get_item(i as usize)?;
        let sub_sizes = sizes_seq.get_item(i as usize)?;
        out.append(unflatten_rec(py, &sub_data, &sub_sizes, dim - 1)?)?;
    }
    Ok(out.into())
}

fn layout_all_valid_no_union(layout: &Layout) -> bool {
    match layout {
        Layout::Leaf(l) => !l.has_nulls,
        Layout::ListOffset(lo) => layout_all_valid_no_union(lo.content.as_ref()),
        Layout::Indexed(ix) => layout_all_valid_no_union(ix.content.as_ref()),
        Layout::OffsetView(v) => layout_all_valid_no_union(v.content.as_ref()),
        Layout::UnionScalarList(_) => false,
    }
}

fn numpy_dtype<'py>(
    np: &Bound<'py, PyModule>,
    dt: DType,
) -> PyResult<Option<(Bound<'py, PyAny>, &'static str)>> {
    let (attr, expected) = match dt {
        DType::Int8 => ("int8", "int8"),
        DType::Int16 => ("int16", "int16"),
        DType::Int32 => ("int32", "int32"),
        DType::Int64 => ("int64", "int64"),
        DType::UInt8 => ("uint8", "uint8"),
        DType::UInt16 => ("uint16", "uint16"),
        DType::UInt32 => ("uint32", "uint32"),
        DType::UInt64 => ("uint64", "uint64"),
        DType::Float16 => ("float16", "float16"),
        DType::Float32 => ("float32", "float32"),
        DType::Float64 => ("float64", "float64"),
        DType::Bool => ("bool_", "bool"),
        DType::Char | DType::String => return Ok(None),
    };
    Ok(Some((np.getattr(attr)?, expected)))
}

fn shape_or_nshape(
    py: Python<'_>,
    arr: &GrumpyArray,
    dim: usize,
    count_non_null_scalars: bool,
) -> PyResult<PyObject> {
    if dim == 0 {
        return Ok((arr.len() as i64).into_py(py));
    }
    let target_axis = dim - 1;
    let out_list = pyo3::types::PyList::empty_bound(py);
    for i in 0..arr.len() {
        let obj = collect_shape_for_element(
            py,
            &arr.layout,
            i,
            0,
            target_axis,
            count_non_null_scalars,
        )?;
        out_list.append(obj)?;
    }
    let out_obj = out_list.into_py(py);
    let built = build_array(py, &out_obj.bind(py), DType::Int64)?;
    Ok(PyGrumpyArray { inner: built }.into_py(py))
}

fn collect_shape_for_element(
    py: Python<'_>,
    layout: &Layout,
    idx: usize,
    axis: usize,
    target_axis: usize,
    count_non_null_scalars: bool,
) -> PyResult<PyObject> {
    if axis == target_axis {
        // At target axis we want the "length at next axis" of list elements only.
        if let Layout::ListOffset(lo) = layout {
            let len = if count_non_null_scalars {
                lo.child_len_non_null_scalars(idx)?
            } else {
                lo.child_len_total(idx)?
            };
            return Ok((len as i64).into_py(py));
        }
        if let Layout::OffsetView(v) = layout {
            let abs = v.start + idx;
            let start = v.offsets[abs] as usize;
            let end = v.offsets[abs + 1] as usize;
            let len = end - start;
            return Ok((len as i64).into_py(py));
        }
        if let Layout::Indexed(ix) = layout {
            // resolve and retry
            let n = ix.content.len() as i64;
            let mut j = ix.index[idx];
            if j < 0 { j += n; }
            if j < 0 || j >= n { return Err(PyValueError::new_err("Index out of bounds.")); }
            return collect_shape_for_element(py, ix.content.as_ref(), j as usize, axis, target_axis, count_non_null_scalars);
        }
        // A scalar (or union element that is scalar) at target axis does not contribute.
        return Ok(py.None());
    }

    match layout {
        Layout::ListOffset(lo) => {
            // We are above the target axis: build a list from children that have a path to the target.
            let start = lo.offsets[idx] as usize;
            let end = lo.offsets[idx + 1] as usize;
            let out = pyo3::types::PyList::empty_bound(py);
            for j in start..end {
                let child = collect_shape_for_any(
                    py,
                    lo.content.as_ref(),
                    j,
                    axis + 1,
                    target_axis,
                    count_non_null_scalars,
                )?;
                if !child.is_none(py) {
                    out.append(child)?;
                }
            }
            Ok(out.into())
        }
        Layout::OffsetView(v) => {
            let abs = v.start + idx;
            let start = v.offsets[abs] as usize;
            let end = v.offsets[abs + 1] as usize;
            let out = pyo3::types::PyList::empty_bound(py);
            for j in start..end {
                let child = collect_shape_for_any(
                    py,
                    v.content.as_ref(),
                    j,
                    axis + 1,
                    target_axis,
                    count_non_null_scalars,
                )?;
                if !child.is_none(py) {
                    out.append(child)?;
                }
            }
            Ok(out.into())
        }
        Layout::Indexed(ix) => {
            // Resolve the indexed element then continue.
            let n = ix.content.len() as i64;
            let mut j = ix.index[idx];
            if j < 0 { j += n; }
            if j < 0 || j >= n {
                return Err(PyValueError::new_err("Index out of bounds."));
            }
            collect_shape_for_element(py, ix.content.as_ref(), j as usize, axis, target_axis, count_non_null_scalars)
        }
        Layout::UnionScalarList(u) => {
            // A union element at this axis: we only descend if it is a list.
            // If scalar, it yields empty list for axes above target (no deeper list nodes).
            let tag = u.tags[idx];
            let ix = u.index[idx] as usize;
            match tag {
                0 => Ok(pyo3::types::PyList::empty_bound(py).into()),
                1 => collect_shape_for_any(
                    py,
                    &Layout::ListOffset(u.lists.clone()),
                    ix,
                    axis,
                    target_axis,
                    count_non_null_scalars,
                ),
                _ => Err(PyValueError::new_err("Invalid union tag.")),
            }
        }
        Layout::Leaf(_) => Ok(pyo3::types::PyList::empty_bound(py).into()),
    }
}

fn collect_shape_for_any(
    py: Python<'_>,
    layout: &Layout,
    idx: usize,
    axis: usize,
    target_axis: usize,
    count_non_null_scalars: bool,
) -> PyResult<PyObject> {
    if axis == target_axis {
        match layout {
            Layout::ListOffset(lo) => {
                let len = if count_non_null_scalars {
                    lo.child_len_non_null_scalars(idx)?
                } else {
                    lo.child_len_total(idx)?
                };
                Ok((len as i64).into_py(py))
            }
            Layout::OffsetView(v) => {
                let abs = v.start + idx;
                let start = v.offsets[abs] as usize;
                let end = v.offsets[abs + 1] as usize;
                Ok(((end - start) as i64).into_py(py))
            }
            Layout::Indexed(ix) => {
                let n = ix.content.len() as i64;
                let mut j = ix.index[idx];
                if j < 0 { j += n; }
                if j < 0 || j >= n {
                    return Err(PyValueError::new_err("Index out of bounds."));
                }
                collect_shape_for_any(py, ix.content.as_ref(), j as usize, axis, target_axis, count_non_null_scalars)
            }
            Layout::UnionScalarList(u) => {
                // At target axis, only list elements contribute.
                let tag = u.tags[idx];
                let ix = u.index[idx] as usize;
                if tag == 1 {
                    let lo = &u.lists;
                    let len = if count_non_null_scalars {
                        lo.child_len_non_null_scalars(ix)?
                    } else {
                        lo.child_len_total(ix)?
                    };
                    Ok((len as i64).into_py(py))
                } else {
                    Ok(py.None())
                }
            }
            Layout::Leaf(_) => Ok(py.None()),
        }
    } else {
        collect_shape_for_element(
            py,
            layout,
            idx,
            axis,
            target_axis,
            count_non_null_scalars,
        )
    }
}

fn wrap_result(py: Python<'_>, out: PyObject, dtype: DType) -> PyResult<PyObject> {
    let bound = out.bind(py);
    if crate::dtype::is_sequence_like(py, &bound)? {
        let arr = build_array(py, &bound, dtype)?;
        Ok(PyGrumpyArray { inner: arr }.into_py(py))
    } else {
        Ok(out)
    }
}

fn fast_getitem(py: Python<'_>, arr: &GrumpyArray, index: &Bound<'_, PyAny>) -> PyResult<Option<PyObject>> {
    // Only optimize pure list chains (no unions); unions fall back.
    if !arr.is_pure_list_chain() {
        return Ok(None);
    }

    // Coordinate tuple indexing
    if let Ok(tup) = index.downcast::<PyTuple>() {
        // Hot path: x[int, int] on 2D int32 arrays (skip fancy-index probes).
        if tup.len() == 2 {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.is_instance_of::<PyInt>() && a1.is_instance_of::<PyInt>() {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    if arr.dtype == DType::Int32 {
                        if let Layout::ListOffset(lo) = &arr.layout {
                            if let Layout::Leaf(leaf) = lo.content.as_ref() {
                                if !leaf.has_nulls {
                                    if let LeafBuffer::I32(buf) = &leaf.buffer {
                                        let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                                        return Ok(Some((buf[leaf_ix] as i64).into_py(py)));
                                    }
                                }
                            }
                        }
                    }
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_fast(&arr.layout)?;
                    let out = leaf.scalar_to_py(py, leaf_ix)?;
                    return Ok(Some(out));
                }
            }
        }

        // Fancy coordinate indexing if any part is sequence-like (and no slices).
        let mut has_seq = false;
        let mut has_slice = false;
        for i in 0..tup.len() {
            let p = tup.get_item(i)?;
            if p.downcast::<PySlice>().is_ok() {
                has_slice = true;
            } else if is_index_vec_like(py, &p)? {
                has_seq = true;
            }
        }
        if has_slice {
            return Ok(None);
        }
        if tup.len() == 2 && !has_seq {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.downcast::<PySlice>().is_err() && a1.downcast::<PySlice>().is_err() {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    if arr.dtype == DType::Int32 {
                        if let Layout::ListOffset(lo) = &arr.layout {
                            if let Layout::Leaf(leaf) = lo.content.as_ref() {
                                if !leaf.has_nulls {
                                    if let LeafBuffer::I32(buf) = &leaf.buffer {
                                        let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                                        return Ok(Some((buf[leaf_ix] as i64).into_py(py)));
                                    }
                                }
                            }
                        }
                    }
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_fast(&arr.layout)?;
                    let out = leaf.scalar_to_py(py, leaf_ix)?;
                    return Ok(Some(out));
                }
            }
        }
        if has_seq {
            // Support 2D fancy: (rows, cols) with optional scalar broadcast.
            if tup.len() != 2 {
                return Ok(None);
            }
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            let rows_d = extract_index_data(py, &a0)?;
            let cols_d = extract_index_data(py, &a1)?;
            let rows: &[i64] = match &rows_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };
            let cols: &[i64] = match &cols_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };

            if !rows.is_empty() && !cols.is_empty() {
                let leaf = gather_2d_fancy_leaf(&arr.layout, &rows, &cols)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            if !rows.is_empty() && cols.is_empty() {
                let col = a1.extract::<i64>()?;
                let cols2 = vec![col; rows.len()];
                let leaf = gather_2d_fancy_leaf(&arr.layout, &rows, &cols2)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            if rows.is_empty() && !cols.is_empty() {
                let row = a0.extract::<i64>()?;
                let rows2 = vec![row; cols.len()];
                let leaf = gather_2d_fancy_leaf(&arr.layout, &rows2, &cols)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            return Ok(None);
        }

        // Pure coordinate (ints only)
        if tup.len() == 1 {
            let i0 = tup.get_item(0)?.extract::<i64>()?;
            return fast_getitem(py, arr, &i0.into_py(py).into_bound(py));
        }
        // Scalar coordinate: return scalar
        let mut coords: Vec<i64> = Vec::with_capacity(tup.len());
        for i in 0..tup.len() {
            coords.push(tup.get_item(i)?.extract::<i64>()?);
        }
        // Only support scalar selection ending in leaf scalar.
        let leaf_ix = coord_to_leaf_index(&arr.layout, &coords)?;
        let leaf = find_leaf_fast(&arr.layout)?;
        let out = leaf.scalar_to_py(py, leaf_ix)?;
        return Ok(Some(out));
    }

    // Axis-0 int selection (drops axis)
    if let Ok(i0) = index.extract::<i64>() {
        let root_len = arr.len() as i64;
        let mut i = i0;
        if i < 0 {
            i += root_len;
        }
        if i < 0 || i >= root_len {
            return Err(PyValueError::new_err("Index out of bounds."));
        }
        let layout = drop_axis0_select_element(&arr.layout, i as usize)?;
        let out = PyGrumpyArray {
            inner: GrumpyArray {
                dtype: arr.dtype,
                layout,
            },
        };
        return Ok(Some(out.into_py(py)));
    }

    // Axis-0 slice view (no copy) for list-chains: return a new ListOffset with trimmed offsets,
    // sharing underlying leaf buffers via Arc/COW.
    if let Ok(slc) = index.downcast::<PySlice>() {
        let root_lo = match &arr.layout {
            Layout::ListOffset(lo) => lo,
            _ => return Ok(None),
        };
        // Only step=1 for now (fast and common). Other steps fall back.
        let n: isize = root_lo
            .len()
            .try_into()
            .map_err(|_| PyValueError::new_err("Slice length too large."))?;
        let indices = slc.indices(n)?;
        let (start, stop, step) = (indices.start, indices.stop, indices.step);
        if step != 1 {
            return Ok(None);
        }
        let start_u = start as usize;
        let stop_u = stop as usize;
        if start_u > stop_u || stop_u > root_lo.len() {
            return Err(PyValueError::new_err("Slice out of bounds."));
        }
        let layout = Layout::OffsetView(crate::layout::OffsetView {
            offsets: root_lo.offsets.clone(),
            start: start_u,
            stop: stop_u,
            content: root_lo.content.clone(),
        });
        let out = PyGrumpyArray {
            inner: GrumpyArray {
                dtype: arr.dtype,
                layout,
            },
        };
        return Ok(Some(out.into_py(py)));
    }

    Ok(None)
}

fn find_leaf_fast<'a>(layout: &'a Layout) -> PyResult<&'a crate::layout::Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::ListOffset(lo) => find_leaf_fast(lo.content.as_ref()),
        Layout::Indexed(ix) => find_leaf_fast(ix.content.as_ref()),
        Layout::OffsetView(v) => find_leaf_fast(v.content.as_ref()),
        Layout::UnionScalarList(_) => Err(PyValueError::new_err("Union not supported.")),
    }
}

fn fast_setitem(
    py: Python<'_>,
    arr: &mut GrumpyArray,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !arr.is_pure_list_chain() {
        return Ok(false);
    }

    // Only support leaf-mutation assignments (no structural changes).
    // Supported:
    // - x[i,j] = scalar
    // - x[[i...],[j...]] = list/scalar
    // - x[[i...], j] scalar broadcast
    if let Ok(tup) = index.downcast::<PyTuple>() {
        if tup.len() == 0 {
            return Ok(false);
        }

        // Hot path: x[int, int] = int on 2D int32 arrays (skip fancy-index probes).
        if tup.len() == 2 {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.is_instance_of::<PyInt>() && a1.is_instance_of::<PyInt>() {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    if arr.dtype == DType::Int32 {
                        if let Ok(v) = value.extract::<i32>() {
                            let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                            if let Layout::ListOffset(lo) = &mut arr.layout {
                                if let Layout::Leaf(leaf) = lo.content.as_mut() {
                                    if !leaf.has_nulls {
                                        if let LeafBuffer::I32(buf) = &mut leaf.buffer {
                                            Arc::make_mut(buf)[leaf_ix] = v;
                                            return Ok(true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_mut_fast(&mut arr.layout)?;
                    if arr.dtype == DType::Int32 {
                        if let Ok(v) = value.extract::<i32>() {
                            leaf.set_i32(leaf_ix, v)?;
                            return Ok(true);
                        }
                    }
                }
            }
        }

        let mut has_seq = false;
        let mut has_slice = false;
        for i in 0..tup.len() {
            let p = tup.get_item(i)?;
            if p.downcast::<PySlice>().is_ok() {
                has_slice = true;
            } else if is_index_vec_like(py, &p)? {
                has_seq = true;
            }
        }
        if has_slice {
            return Ok(false);
        }
        if has_seq {
            if tup.len() != 2 {
                return Ok(false);
            }
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            let rows_d = extract_index_data(py, &a0)?;
            let cols_d = extract_index_data(py, &a1)?;
            let rows: &[i64] = match &rows_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };
            let cols: &[i64] = match &cols_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };
            let n = if !rows.is_empty() { rows.len() } else { cols.len() };
            if n == 0 {
                return Ok(false);
            }
            let rows2_owned;
            let cols2_owned;
            let rows2: &[i64] = if rows.is_empty() {
                rows2_owned = vec![a0.extract::<i64>()?; n];
                rows2_owned.as_slice()
            } else {
                rows
            };
            let cols2: &[i64] = if cols.is_empty() {
                cols2_owned = vec![a1.extract::<i64>()?; n];
                cols2_owned.as_slice()
            } else {
                cols
            };

            // If dtype=int32 and values is a NumPy i32 array, use fully typed scatter (no per-element Python extraction).
            if arr.dtype == DType::Int32 {
                if let Ok(vro) = value.extract::<PyReadonlyArray1<'_, i32>>() {
                    let vs = vro.as_slice()?;
                    if vs.len() != n {
                        return Err(PyValueError::new_err(
                            "Assignment value length must match number of selected coordinates.",
                        ));
                    }
                    scatter_2d_fancy_i32(&mut arr.layout, rows2, cols2, vs)?;
                    return Ok(true);
                }
            }

            // Otherwise fall back to generic vectorized scatter (still avoids rebuilding structures).
            scatter_2d_fancy_numeric(py, &mut arr.layout, rows2, cols2, value, arr.dtype)?;
            return Ok(true);
        }

        // Pure scalar coordinate assignment (2D+ supported if it ends in leaf)
        if tup.len() == 2 {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.downcast::<PySlice>().is_err()
                && a1.downcast::<PySlice>().is_err()
                && !is_index_vec_like(py, &a0)?
                && !is_index_vec_like(py, &a1)?
            {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_mut_fast(&mut arr.layout)?;
                    if arr.dtype == DType::Int32 {
                        if let Ok(v) = value.extract::<i32>() {
                            leaf.set_i32(leaf_ix, v)?;
                            return Ok(true);
                        }
                    }
                }
            }
        }
        let mut coords: Vec<i64> = Vec::with_capacity(tup.len());
        for i in 0..tup.len() {
            coords.push(tup.get_item(i)?.extract::<i64>()?);
        }
        let leaf_ix = coord_to_leaf_index(&arr.layout, &coords)?;
        let (valid, bytes) = crate::layout::Leaf::encode_scalar(py, value, arr.dtype)?;
        let leaf = find_leaf_mut_fast(&mut arr.layout)?;
        leaf.set_encoded(leaf_ix, valid, &bytes)?;
        return Ok(true);
    }

    Ok(false)
}

fn find_leaf_mut_fast<'a>(layout: &'a mut Layout) -> PyResult<&'a mut crate::layout::Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::ListOffset(lo) => find_leaf_mut_fast(lo.content.as_mut()),
        Layout::Indexed(ix) => find_leaf_mut_fast(ix.content.as_mut()),
        Layout::OffsetView(v) => find_leaf_mut_fast(v.content.as_mut()),
        Layout::UnionScalarList(_) => Err(PyValueError::new_err("Union not supported.")),
    }
}

enum IndexData<'py> {
    Empty,
    NpI64(PyReadonlyArray1<'py, i64>),
    Owned(Vec<i64>),
}

fn extract_index_data<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<IndexData<'py>> {
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<'py, i64>>() {
        return Ok(IndexData::NpI64(ro));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<'py, i32>>() {
        let slice = ro.as_slice()?;
        return Ok(IndexData::Owned(slice.iter().map(|&x| x as i64).collect()));
    }
    if crate::dtype::is_sequence_like(py, obj)? {
        let s = obj.downcast::<pyo3::types::PySequence>()?;
        let mut out = Vec::with_capacity(s.len()? as usize);
        for i in 0..s.len()? {
            out.push(s.get_item(i as usize)?.extract::<i64>()?);
        }
        return Ok(IndexData::Owned(out));
    }
    Ok(IndexData::Empty)
}

fn is_index_vec_like(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.extract::<PyReadonlyArray1<'_, i64>>().is_ok() {
        return Ok(true);
    }
    if obj.extract::<PyReadonlyArray1<'_, i32>>().is_ok() {
        return Ok(true);
    }
    crate::dtype::is_sequence_like(py, obj)
}

fn getitem_coordinate(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    _dtype: DType,
) -> PyResult<PyObject> {
    // Coordinate indexing:
    // - int/slice: applied on axis 0
    // - tuple: multiple axes, with fancy coordinates if any axis is a sequence of ints.
    if let Ok(tup) = index.downcast::<PyTuple>() {
        if tup.len() == 0 {
            return Err(PyValueError::new_err("Empty index tuple is not allowed."));
        }
        let parts: Vec<Bound<'_, PyAny>> = (0..tup.len()).map(|i| tup.get_item(i).unwrap()).collect();
        return getitem_coordinate_tuple(py, base, &parts);
    }

    // Single-axis coordinate indexing
    getitem_axis(py, base, index)
}

enum CoordPart {
    Int(i64),
    Fancy(Vec<i64>),
}

fn getitem_coordinate_tuple(py: Python<'_>, base: &Bound<'_, PyAny>, parts: &[Bound<'_, PyAny>]) -> PyResult<PyObject> {
    // If there are no fancy parts (sequences), just apply axis-by-axis with int/slice.
    let mut has_fancy = false;
    for p in parts {
        if crate::dtype::is_sequence_like(py, p)? {
            has_fancy = true;
            break;
        }
    }
    if !has_fancy {
        let mut cur = base.clone().unbind();
        let mut cur_b = cur.bind(py);
        for p in parts {
            cur = getitem_axis(py, &cur_b, p)?;
            cur_b = cur.bind(py);
        }
        return Ok(cur);
    }

    // Fancy coordinate indexing: allow scalars and 1D int sequences; disallow slices.
    let mut parsed: Vec<CoordPart> = Vec::with_capacity(parts.len());
    let mut fancy_lens: Vec<usize> = Vec::new();
    for p in parts {
        if p.downcast::<PySlice>().is_ok() {
            return Err(PyValueError::new_err(
                "Slice is not supported together with fancy coordinate arrays (yet).",
            ));
        }
        if let Ok(i) = p.extract::<i64>() {
            parsed.push(CoordPart::Int(i));
        } else if crate::dtype::is_sequence_like(py, p)? {
            let seq = p.downcast::<pyo3::types::PySequence>()?;
            let mut v = Vec::with_capacity(seq.len()? as usize);
            for j in 0..seq.len()? {
                v.push(seq.get_item(j as usize)?.extract::<i64>()?);
            }
            fancy_lens.push(v.len());
            parsed.push(CoordPart::Fancy(v));
        } else {
            return Err(PyValueError::new_err(
                "Unsupported index component in coordinate indexing.",
            ));
        }
    }

    let n = *fancy_lens
        .iter()
        .max()
        .ok_or_else(|| PyValueError::new_err("Internal error."))?;
    for l in &fancy_lens {
        if *l != n {
            return Err(PyValueError::new_err(
                "Coordinate indexing with multiple index arrays requires same length (broadcasting not yet supported except scalars).",
            ));
        }
    }

    let out = pyo3::types::PyList::empty_bound(py);
    for k in 0..n {
        let mut cur = base.clone().unbind();
        let mut cur_b = cur.bind(py);
        for p in &parsed {
            let ix = match p {
                CoordPart::Int(i) => *i,
                CoordPart::Fancy(v) => v[k],
            };
            cur = getitem_axis(py, &cur_b, &ix.into_py(py).into_bound(py))?;
            cur_b = cur.bind(py);
        }
        out.append(cur)?;
    }
    Ok(out.into())
}

fn getitem_axis(py: Python<'_>, base: &Bound<'_, PyAny>, index: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    // base must be sequence-like for int/slice selection
    let seq = base.downcast::<pyo3::types::PySequence>().map_err(|_| {
        PyValueError::new_err("Attempted to index into a scalar value.")
    })?;
    let len = seq.len()? as i64;

    if let Ok(slc) = index.downcast::<PySlice>() {
        let (start, stop, step) = parse_slice(py, slc, len)?;
        let out = pyo3::types::PyList::empty_bound(py);
        let mut i = start;
        if step > 0 {
            while i < stop {
                out.append(seq.get_item(i as usize)?)?;
                i += step;
            }
        } else {
            while i > stop {
                out.append(seq.get_item(i as usize)?)?;
                i += step;
            }
        }
        return Ok(out.into());
    }

    let mut i = index.extract::<i64>().map_err(|_| PyValueError::new_err("Index must be int or slice."))?;
    if i < 0 {
        i += len;
    }
    if i < 0 || i >= len {
        return Err(PyValueError::new_err("Index out of bounds."));
    }
    Ok(seq.get_item(i as usize)?.into())
}

fn parse_slice(
    _py: Python<'_>,
    slc: &Bound<'_, PySlice>,
    len: i64,
) -> PyResult<(i64, i64, i64)> {
    let indices = slc.call_method1("indices", (len,))?;
    let t = indices.downcast::<PyTuple>()?;
    let start = t.get_item(0)?.extract::<i64>()?;
    let stop = t.get_item(1)?.extract::<i64>()?;
    let step = t.get_item(2)?.extract::<i64>()?;
    Ok((start, stop, step))
}

fn getitem_array_indexing(py: Python<'_>, base: &Bound<'_, PyAny>, index: &Bound<'_, PyAny>, _dtype: DType) -> PyResult<PyObject> {
    let base_seq = base.downcast::<pyo3::types::PySequence>()?;
    let n = base_seq.len()? as usize;
    let idx_seq = index.downcast::<pyo3::types::PySequence>()?;
    let m = idx_seq.len()? as usize;

    // Determine if this is a boolean mask on axis 0.
    let mut is_all_bool = true;
    let mut bools: Vec<bool> = Vec::new();
    for i in 0..m {
        let it = idx_seq.get_item(i)?;
        if it.is_instance_of::<pyo3::types::PyBool>() {
            bools.push(it.extract::<bool>()?);
        } else {
            is_all_bool = false;
            break;
        }
    }
    if is_all_bool {
        if m != n {
            return Err(PyValueError::new_err(
                "Boolean indexing requires mask length to match outer dimension.",
            ));
        }
        let out = pyo3::types::PyList::empty_bound(py);
        for i in 0..n {
            if bools[i] {
                out.append(base_seq.get_item(i as usize)?)?;
            }
        }
        return Ok(out.into());
    }

    // If index length != outer length, treat as outer fancy selection on axis 0.
    if m != n {
        let out = pyo3::types::PyList::empty_bound(py);
        for i in 0..m {
            let it = idx_seq.get_item(i)?;
            let mut ix = it.extract::<i64>()?;
            if ix < 0 {
                ix += n as i64;
            }
            if ix < 0 || ix >= n as i64 {
                return Err(PyValueError::new_err("Index out of bounds."));
            }
            out.append(base_seq.get_item(ix as usize)?)?;
        }
        return Ok(out.into());
    }

    // Per-row indexing: apply each index element to corresponding row.
    let out = pyo3::types::PyList::empty_bound(py);
    for i in 0..n {
        let row = base_seq.get_item(i as usize)?;
        let sub = idx_seq.get_item(i as usize)?;
        // For int -> wrap as single-element list (matches example [[1],[5]])
        if let Ok(slc) = sub.downcast::<PySlice>() {
            let got = getitem_axis(py, &row, &slc.clone().into_any())?;
            out.append(got)?;
        } else if sub.extract::<i64>().is_ok() {
            let v = getitem_axis(py, &row, &sub)?;
            let wrap = pyo3::types::PyList::empty_bound(py);
            wrap.append(v)?;
            out.append(wrap)?;
        } else if crate::dtype::is_sequence_like(py, &sub)? {
            // sequence of ints for this row
            let sseq = sub.downcast::<pyo3::types::PySequence>()?;
            let wrap = pyo3::types::PyList::empty_bound(py);
            for j in 0..sseq.len()? {
                let jx = sseq.get_item(j as usize)?;
                let v = getitem_axis(py, &row, &jx)?;
                wrap.append(v)?;
            }
            out.append(wrap)?;
        } else {
            return Err(PyValueError::new_err("Unsupported per-row index element."));
        }
    }
    Ok(out.into())
}

fn setitem_coordinate(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(tup) = index.downcast::<PyTuple>() {
        let parts: Vec<Bound<'_, PyAny>> = (0..tup.len()).map(|i| tup.get_item(i).unwrap()).collect();
        return setitem_coordinate_tuple(py, base, &parts, value);
    }
    setitem_axis(py, base, index, value)
}

fn setitem_coordinate_tuple(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    parts: &[Bound<'_, PyAny>],
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if parts.is_empty() {
        return Err(PyValueError::new_err("Empty index tuple is not allowed."));
    }

    let mut has_fancy = false;
    for p in parts {
        if crate::dtype::is_sequence_like(py, p)? {
            has_fancy = true;
            break;
        }
    }

    if !has_fancy {
        // Walk down to the last axis, mutating at the end.
        let mut cur_obj = base.clone().unbind();
        for ax in 0..parts.len() - 1 {
            let p = &parts[ax];
            if p.downcast::<PySlice>().is_ok() {
                return Err(PyValueError::new_err(
                    "Slice in coordinate tuple assignment is not supported yet.",
                ));
            }
            let child = getitem_axis(py, &cur_obj.bind(py), p)?;
            cur_obj = child;
        }
        let last = &parts[parts.len() - 1];
        return setitem_axis(py, &cur_obj.bind(py), last, value);
    }

    // Fancy coordinate assignment: allow scalar ints and 1D int sequences; disallow slices.
    // All index arrays must have same length; scalar ints are broadcast.
    let mut fancy_lens: Vec<usize> = Vec::new();
    let mut parsed_ints: Vec<Option<Vec<i64>>> = Vec::with_capacity(parts.len());
    let mut parsed_scalars: Vec<Option<i64>> = Vec::with_capacity(parts.len());
    for p in parts {
        if p.downcast::<PySlice>().is_ok() {
            return Err(PyValueError::new_err(
                "Slice is not supported together with fancy coordinate arrays (yet).",
            ));
        }
        if let Ok(i) = p.extract::<i64>() {
            parsed_scalars.push(Some(i));
            parsed_ints.push(None);
        } else if crate::dtype::is_sequence_like(py, p)? {
            let seq = p.downcast::<pyo3::types::PySequence>()?;
            let mut v = Vec::with_capacity(seq.len()? as usize);
            for j in 0..seq.len()? {
                v.push(seq.get_item(j as usize)?.extract::<i64>()?);
            }
            fancy_lens.push(v.len());
            parsed_scalars.push(None);
            parsed_ints.push(Some(v));
        } else {
            return Err(PyValueError::new_err(
                "Unsupported index component in coordinate indexing assignment.",
            ));
        }
    }
    let n = *fancy_lens
        .iter()
        .max()
        .ok_or_else(|| PyValueError::new_err("Internal error."))?;
    for l in &fancy_lens {
        if *l != n {
            return Err(PyValueError::new_err(
                "Coordinate assignment with multiple index arrays requires same length (broadcasting not yet supported except scalars).",
            ));
        }
    }

    // Values: scalar broadcast or sequence length n
    let (values_is_scalar, values_seq) = if crate::dtype::is_sequence_like(py, value)? {
        let vseq = value.downcast::<pyo3::types::PySequence>()?;
        if vseq.len()? as usize != n {
            return Err(PyValueError::new_err(
                "Assignment value length must match number of selected coordinates.",
            ));
        }
        (false, Some(vseq))
    } else {
        (true, None)
    };

    for k in 0..n {
        // Walk to parent of last axis
        let mut cur_obj = base.clone().unbind();
        for ax in 0..parts.len() - 1 {
            let ix = if let Some(s) = parsed_scalars[ax] {
                s
            } else {
                parsed_ints[ax].as_ref().unwrap()[k]
            };
            let child = getitem_axis(py, &cur_obj.bind(py), &ix.into_py(py).into_bound(py))?;
            cur_obj = child;
        }
        let last_ax = parts.len() - 1;
        let last_ix = if let Some(s) = parsed_scalars[last_ax] {
            s
        } else {
            parsed_ints[last_ax].as_ref().unwrap()[k]
        };
        let v_k = if values_is_scalar {
            value.clone()
        } else {
            values_seq
                .as_ref()
                .unwrap()
                .get_item(k as usize)?
        };
        setitem_axis(py, &cur_obj.bind(py), &last_ix.into_py(py).into_bound(py), &v_k)?;
    }

    Ok(())
}

fn setitem_axis(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let seq = base.downcast::<pyo3::types::PySequence>().map_err(|_| {
        PyValueError::new_err("Attempted to assign into a scalar value.")
    })?;
    let len = seq.len()? as i64;

    if let Ok(slc) = index.downcast::<PySlice>() {
        let (start, stop, step) = parse_slice(py, slc, len)?;
        if step == 0 {
            return Err(PyValueError::new_err("Slice step cannot be zero."));
        }
        // Collect target indices
        let mut idxs: Vec<usize> = Vec::new();
        let mut i = start;
        if step > 0 {
            while i < stop {
                idxs.push(i as usize);
                i += step;
            }
        } else {
            while i > stop {
                idxs.push(i as usize);
                i += step;
            }
        }
        if crate::dtype::is_sequence_like(py, value)? {
            let vseq = value.downcast::<pyo3::types::PySequence>()?;
            if vseq.len()? as usize != idxs.len() {
                return Err(PyValueError::new_err(
                    "Slice assignment value length mismatch.",
                ));
            }
            for (k, ix) in idxs.iter().enumerate() {
                let v = vseq.get_item(k)?;
                seq.set_item(*ix, v)?;
            }
        } else {
            for ix in idxs {
                seq.set_item(ix, value.clone())?;
            }
        }
        return Ok(());
    }

    let mut i = index
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err("Index must be int or slice."))?;
    if i < 0 {
        i += len;
    }
    if i < 0 || i >= len {
        return Err(PyValueError::new_err("Index out of bounds."));
    }
    seq.set_item(i as usize, value.clone())?;
    Ok(())
}

fn setitem_array(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let base_seq = base.downcast::<pyo3::types::PySequence>()?;
    let n = base_seq.len()? as usize;
    let idx_seq = index.downcast::<pyo3::types::PySequence>()?;
    let m = idx_seq.len()? as usize;

    // boolean mask on axis 0?
    let mut is_all_bool = true;
    let mut bools: Vec<bool> = Vec::new();
    for i in 0..m {
        let it = idx_seq.get_item(i)?;
        if it.is_instance_of::<pyo3::types::PyBool>() {
            bools.push(it.extract::<bool>()?);
        } else {
            is_all_bool = false;
            break;
        }
    }
    if is_all_bool {
        if m != n {
            return Err(PyValueError::new_err(
                "Boolean indexing assignment requires mask length to match outer dimension.",
            ));
        }
        let targets: Vec<usize> = (0..n).filter(|i| bools[*i]).collect();
        return assign_outer_positions(py, &base_seq, &targets, value);
    }

    // Outer fancy if len != outer dim.
    if m != n {
        let mut targets: Vec<usize> = Vec::with_capacity(m);
        for i in 0..m {
            let it = idx_seq.get_item(i)?;
            let mut ix = it.extract::<i64>()?;
            if ix < 0 {
                ix += n as i64;
            }
            if ix < 0 || ix >= n as i64 {
                return Err(PyValueError::new_err("Index out of bounds."));
            }
            targets.push(ix as usize);
        }
        return assign_outer_positions(py, &base_seq, &targets, value);
    }

    // Per-row assignment: m == n
    let values_per_row: Option<Vec<PyObject>> = if crate::dtype::is_sequence_like(py, value)? {
        let vseq = value.downcast::<pyo3::types::PySequence>()?;
        if vseq.len()? as usize == n {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(vseq.get_item(i)?.into());
            }
            Some(out)
        } else {
            None
        }
    } else {
        None
    };

    for i in 0..n {
        let row = base_seq.get_item(i)?;
        let sub = idx_seq.get_item(i)?;
        let apply = |v: &Bound<'_, PyAny>| -> PyResult<()> {
            if let Ok(slc) = sub.downcast::<PySlice>() {
                setitem_axis(py, &row, &slc.clone().into_any(), v)?;
            } else if sub.extract::<i64>().is_ok() {
                // Assign into single position
                setitem_axis(py, &row, &sub, v)?;
            } else if crate::dtype::is_sequence_like(py, &sub)? {
                // Sequence of indices for this row: allow scalar broadcast or sequence matching index count.
                let sseq = sub.downcast::<pyo3::types::PySequence>()?;
                if crate::dtype::is_sequence_like(py, v)? {
                    let vseqq = v.downcast::<pyo3::types::PySequence>()?;
                    if vseqq.len()? != sseq.len()? {
                        return Err(PyValueError::new_err(
                            "Per-row assignment value length mismatch.",
                        ));
                    }
                    for j in 0..sseq.len()? {
                        let jx = sseq.get_item(j as usize)?;
                        let vv = vseqq.get_item(j as usize)?;
                        setitem_axis(py, &row, &jx, &vv)?;
                    }
                } else {
                    for j in 0..sseq.len()? {
                        let jx = sseq.get_item(j as usize)?;
                        setitem_axis(py, &row, &jx, v)?;
                    }
                }
            } else {
                return Err(PyValueError::new_err("Unsupported per-row index element."));
            }
            Ok(())
        };

        if let Some(vs) = &values_per_row {
            let v_i = vs[i].bind(py);
            apply(&v_i)?;
        } else {
            apply(value)?;
        }
    }
    Ok(())
}

fn assign_outer_positions(
    py: Python<'_>,
    base_seq: &Bound<'_, pyo3::types::PySequence>,
    targets: &[usize],
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if crate::dtype::is_sequence_like(py, value)? {
        let vseq = value.downcast::<pyo3::types::PySequence>()?;
        if vseq.len()? as usize != targets.len() {
            return Err(PyValueError::new_err(
                "Assignment value length must match number of selected elements.",
            ));
        }
        for (k, ix) in targets.iter().enumerate() {
            base_seq.set_item(*ix, vseq.get_item(k)?)?;
        }
    } else {
        for ix in targets {
            base_seq.set_item(*ix, value.clone())?;
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn array(
    py: Python<'_>,
    obj: Bound<'_, pyo3::types::PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let dt = if let Some(d) = dtype {
        d.dt
    } else {
        let inf = infer_dtype(py, &obj)?;
        let cls = inf.ok_or_else(|| {
            PyValueError::new_err("Cannot infer dtype from all-null input. Specify dtype explicitly.")
        })?;
        inferclass_to_dtype(cls)
    };
    let inner = build_array(py, &obj, dt)?;
    Ok(PyGrumpyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn multiply(
    a: PyRef<'_, PyGrumpyArray>,
    b: PyRef<'_, PyGrumpyArray>,
    mut out: Option<PyRefMut<'_, PyGrumpyArray>>,
) -> PyResult<PyGrumpyArray> {
    if let Some(ref mut o) = out {
        ops::elementwise_into(&mut o.inner, &a.inner, &b.inner, BinOp::Mul)?;
        Ok(PyGrumpyArray { inner: o.inner.clone() })
    } else {
        let inner = ops::elementwise(&a.inner, &b.inner, BinOp::Mul)?;
        Ok(PyGrumpyArray { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn add_arrays(
    a: PyRef<'_, PyGrumpyArray>,
    b: PyRef<'_, PyGrumpyArray>,
    mut out: Option<PyRefMut<'_, PyGrumpyArray>>,
) -> PyResult<PyGrumpyArray> {
    if let Some(ref mut o) = out {
        ops::elementwise_into(&mut o.inner, &a.inner, &b.inner, BinOp::Add)?;
        Ok(PyGrumpyArray { inner: o.inner.clone() })
    } else {
        let inner = ops::elementwise(&a.inner, &b.inner, BinOp::Add)?;
        Ok(PyGrumpyArray { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
pub fn subtract(
    a: PyRef<'_, PyGrumpyArray>,
    b: PyRef<'_, PyGrumpyArray>,
    mut out: Option<PyRefMut<'_, PyGrumpyArray>>,
) -> PyResult<PyGrumpyArray> {
    if let Some(ref mut o) = out {
        ops::elementwise_into(&mut o.inner, &a.inner, &b.inner, BinOp::Sub)?;
        Ok(PyGrumpyArray { inner: o.inner.clone() })
    } else {
        let inner = ops::elementwise(&a.inner, &b.inner, BinOp::Sub)?;
        Ok(PyGrumpyArray { inner })
    }
}

#[pyfunction]
#[pyo3(signature = (arrays, dim=0))]
pub fn cat(py: Python<'_>, arrays: Vec<PyRef<'_, PyGrumpyArray>>, dim: isize) -> PyResult<PyGrumpyArray> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("cat() requires at least one array."));
    }
    let dim_u = if dim < 0 {
        return Err(PyValueError::new_err("Negative dim is not supported yet."));
    } else {
        dim as usize
    };
    let dtype = arrays[0].inner.dtype;
    for a in &arrays[1..] {
        if a.inner.dtype != dtype {
            return Err(PyValueError::new_err(
                "cat() requires all input arrays to have the same dtype for milestone-2.",
            ));
        }
    }
    let rust_arrays: Vec<GrumpyArray> = arrays.into_iter().map(|a| a.inner.clone()).collect();
    let merged_list = concat_to_py_list(py, &rust_arrays, dim_u)?;
    let merged = build_array(py, &merged_list.bind(py), dtype)?;
    Ok(PyGrumpyArray { inner: merged })
}

#[pyfunction]
#[pyo3(signature = (x, fill_value, dtype=None))]
pub fn full_like(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    fill_value: Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let dt = dtype.map(|d| d.dt).unwrap_or(x.inner.dtype);
    let layout = fill_layout_like(py, &x.inner.layout, dt, &fill_value)?;
    Ok(PyGrumpyArray {
        inner: GrumpyArray { dtype: dt, layout },
    })
}

#[pyfunction]
#[pyo3(signature = (x, dtype=None))]
pub fn zeros_like(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>, dtype: Option<PyDType>) -> PyResult<PyGrumpyArray> {
    let dt = dtype.map(|d| d.dt).unwrap_or(x.inner.dtype);
    let fill = match dt {
        DType::Bool => false.into_py(py),
        DType::Char => return Err(PyValueError::new_err("zeros_like(dtype=char) is not supported.")),
        _ => 0.into_py(py),
    };
    full_like(py, x, fill.into_bound(py), Some(PyDType { dt }))
}

#[pyfunction]
#[pyo3(signature = (x, dtype=None))]
pub fn ones_like(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>, dtype: Option<PyDType>) -> PyResult<PyGrumpyArray> {
    let dt = dtype.map(|d| d.dt).unwrap_or(x.inner.dtype);
    let fill = match dt {
        DType::Bool => true.into_py(py),
        DType::Char => return Err(PyValueError::new_err("ones_like(dtype=char) is not supported.")),
        _ => 1.into_py(py),
    };
    full_like(py, x, fill.into_bound(py), Some(PyDType { dt }))
}

#[pyfunction]
pub fn unique(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::unique(py, &x.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn isin(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>, test_elements: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let a = x.inner.clone();
    let test = test_elements.inner.clone();
    let out = py.allow_threads(|| set_ops::isin(&a, &test))?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn setdiff(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::setdiff(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn setunion(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::setunion(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn setxor(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = set_ops::setxor(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, weights=None, minlength=0))]
pub fn bincount(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    weights: Option<PyRef<'_, PyGrumpyArray>>,
    minlength: usize,
) -> PyResult<PyGrumpyArray> {
    let out = hist_ops::bincount(py, &x.inner, weights.as_ref().map(|w| &w.inner), minlength)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, bins, right=false))]
pub fn digitize(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    bins: PyRef<'_, PyGrumpyArray>,
    right: bool,
) -> PyResult<PyGrumpyArray> {
    let out = hist_ops::digitize(py, &x.inner, &bins.inner, right)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, bins=10, range=None, density=false, weights=None))]
pub fn histogram(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    bins: usize,
    range: Option<Bound<'_, PyAny>>,
    density: bool,
    weights: Option<PyRef<'_, PyGrumpyArray>>,
) -> PyResult<(PyGrumpyArray, PyGrumpyArray)> {
    let range_parsed: Option<(f64, f64)> = if let Some(r) = range {
        let tup = r.downcast::<pyo3::types::PyTuple>().map_err(|_| PyValueError::new_err("range must be a tuple (lo, hi)."))?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err("range must have length 2."));
        }
        Some((tup.get_item(0)?.extract::<f64>()?, tup.get_item(1)?.extract::<f64>()?))
    } else {
        None
    };
    let (h, edges) = hist_ops::histogram(py, &x.inner, bins, range_parsed, density, weights.as_ref().map(|w| &w.inner))?;
    Ok((PyGrumpyArray { inner: h }, PyGrumpyArray { inner: edges }))
}

#[pyfunction]
pub fn nonzero(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = ss_ops::nonzero(py, &x.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (x, v, right=false))]
pub fn search_sorted(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    v: PyRef<'_, PyGrumpyArray>,
    right: bool,
) -> PyResult<PyGrumpyArray> {
    let out = ss_ops::search_sorted(py, &x.inner, &v.inner, right)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (cond, x=None, y=None))]
pub fn where_(
    py: Python<'_>,
    cond: PyRef<'_, PyGrumpyArray>,
    x: Option<PyRef<'_, PyGrumpyArray>>,
    y: Option<PyRef<'_, PyGrumpyArray>>,
) -> PyResult<PyObject> {
    match (x, y) {
        (None, None) => {
            let out = where_ops::where_indices(py, &cond.inner)?;
            Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
        }
        (Some(xx), Some(yy)) => {
            let out = where_ops::where_select(py, &cond.inner, &xx.inner, &yy.inner)?;
            Ok(Py::new(py, PyGrumpyArray { inner: out })?.into_py(py))
        }
        _ => Err(PyValueError::new_err("where(cond, x, y) requires both x and y.")),
    }
}

#[pyfunction]
pub fn argwhere(py: Python<'_>, cond: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = where_ops::argwhere(py, &cond.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn dot(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::dot(py, &a.inner, &b.inner)
}

#[pyfunction]
pub fn inner(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::inner(py, &a.inner, &b.inner)
}

#[pyfunction]
pub fn norm(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::norm(py, &a.inner)
}

#[pyfunction]
pub fn trace(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::trace(py, &a.inner)
}

#[pyfunction]
pub fn outer(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = linalg_ops::outer(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn cross(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = linalg_ops::cross(py, &a.inner, &b.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
pub fn det(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyObject> {
    linalg_ops::det(py, &a.inner)
}

#[pyfunction]
pub fn inv(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>) -> PyResult<PyGrumpyArray> {
    let out = linalg_ops::inv(py, &a.inner)?;
    Ok(PyGrumpyArray { inner: out })
}

#[pyfunction]
#[pyo3(signature = (subscripts, *operands))]
pub fn einsum(py: Python<'_>, subscripts: String, operands: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<PyObject> {
    let mut ops: Vec<GrumpyArray> = Vec::with_capacity(operands.len());
    for i in 0..operands.len() {
        let item = operands.get_item(i)?;
        let arr: PyRef<'_, PyGrumpyArray> = item.extract()?;
        ops.push(arr.inner.clone());
    }
    let result = match einsum_ops::einsum(py, &subscripts, &ops) {
        Ok(r) => r,
        Err(_) => einsum_ops::einsum_numpy_fallback(py, &subscripts, &ops)?,
    };
    match result {
        einsum_ops::TensorOut::Scalar(o) => Ok(o),
        einsum_ops::TensorOut::Array(a) => Ok(Py::new(py, PyGrumpyArray { inner: a })?.into_py(py)),
    }
}

#[pyfunction]
#[pyo3(signature = (a, b, axes=2usize))]
pub fn tensordot(py: Python<'_>, a: PyRef<'_, PyGrumpyArray>, b: PyRef<'_, PyGrumpyArray>, axes: usize) -> PyResult<PyObject> {
    match einsum_ops::tensordot(py, &a.inner, &b.inner, axes)? {
        einsum_ops::TensorOut::Scalar(o) => Ok(o),
        einsum_ops::TensorOut::Array(arr) => Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py)),
    }
}

#[pyfunction]
#[pyo3(signature = (query, data, k=None, radius=None, dim=0, loop_=true, return_distances=false))]
pub fn neighbors(
    py: Python<'_>,
    query: PyRef<'_, PyGrumpyArray>,
    data: PyRef<'_, PyGrumpyArray>,
    k: Option<usize>,
    radius: Option<f64>,
    dim: isize,
    loop_: bool,
    return_distances: bool,
) -> PyResult<PyObject> {
    // Release the GIL: neighbors is a pure Rust compute kernel and can run in parallel threads.
    let q = query.inner.clone();
    let d = data.inner.clone();
    let (edge, dist) = py.allow_threads(move || {
        neigh_ops::neighbors_edge_index_and_distances(&q, &d, k, radius, dim, loop_, return_distances)
    })?;
    let edge_obj = Py::new(py, PyGrumpyArray { inner: edge })?.into_py(py);
    if let Some(dd) = dist {
        let dist_obj = Py::new(py, PyGrumpyArray { inner: dd })?.into_py(py);
        Ok(pyo3::types::PyTuple::new_bound(py, [edge_obj, dist_obj]).into_py(py))
    } else {
        Ok(edge_obj)
    }
}

#[pyfunction]
#[pyo3(signature = (mapping, schema=None))]
pub fn dataframe(py: Python<'_>, mapping: Bound<'_, PyAny>, schema: Option<Bound<'_, PyAny>>) -> PyResult<PyGrumpyDataFrame> {
    let d = mapping
        .downcast::<pyo3::types::PyDict>()
        .map_err(|_| PyValueError::new_err("dataframe(mapping, ...) requires a dict."))?;
    let sch = if let Some(s) = schema {
        Some(df_ops::Schema::parse(py, &s)?)
    } else {
        None
    };
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<GrumpyArray> = Vec::new();
    for (k, v) in d.iter() {
        let name = k.extract::<String>().map_err(|_| PyValueError::new_err("dataframe keys must be strings."))?;
        // If already a GrumpyArray, clone it; else build.
        let arr = if let Ok(g) = v.extract::<PyRef<'_, PyGrumpyArray>>() {
            g.inner.clone()
        } else {
            // dtype inference: use infer_dtype then build_array
            let inferred = crate::dtype::infer_dtype(py, &v)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = crate::dtype::inferclass_to_dtype(inferred);
            crate::layout::build_array(py, &v, dt)?
        };
        names.push(name);
        cols.push(arr);
    }
    let df = df_ops::GrumpyDataFrame::new(names, cols, sch)?;
    Ok(PyGrumpyDataFrame { inner: df })
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyGrumpyArray>()?;
    m.add_class::<PyGrumpyDataFrame>()?;
    m.add_class::<PyDataFrameAccessor>()?;
    m.add_class::<PyCompiledPlan>()?;
    m.add_class::<PyCompiledBatchesIter>()?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
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
    m.add_function(wrap_pyfunction!(neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(stored_len, m)?)?;
    m.add_function(wrap_pyfunction!(load_slice, m)?)?;
    m.add_function(wrap_pyfunction!(compiled_stream_apply, m)?)?;
    Ok(())
}

#[pyfunction]
fn stored_len(path: String) -> PyResult<usize> {
    io_ops::stored_axis0_len(&path)
}

#[pyfunction]
fn load_slice(py: Python<'_>, path: String, start: usize, stop: usize) -> PyResult<PyObject> {
    if let Ok(arr) = io_ops::load_array(py, &path) {
        let sliced = slice_axis0_view(&arr, start, stop)?;
        return Ok(Py::new(py, PyGrumpyArray { inner: sliced })?.into_py(py));
    }
    let df = io_ops::load_dataframe(py, &path)?;
    let sliced = df_slice_axis0_view(&df, start, stop)?;
    Ok(Py::new(py, PyGrumpyDataFrame { inner: sliced })?.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (obj, path, chunk_size=1024usize))]
fn save(py: Python<'_>, obj: Bound<'_, PyAny>, path: String, chunk_size: usize) -> PyResult<()> {
    if let Ok(arr) = obj.extract::<PyRef<'_, PyGrumpyArray>>() {
        return io_ops::save_array(py, &arr.inner, &path, chunk_size);
    }
    if let Ok(df) = obj.extract::<PyRef<'_, PyGrumpyDataFrame>>() {
        return io_ops::save_dataframe(py, &df.inner, &path, chunk_size);
    }
    Err(PyValueError::new_err("gr.save expects a GrumpyArray or GrumpyDataFrame."))
}

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<PyObject> {
    // Try array first (fast); if metadata says dataframe, this errors and we fall back.
    if let Ok(arr) = io_ops::load_array(py, &path) {
        return Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py));
    }
    let df = io_ops::load_dataframe(py, &path)?;
    Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
}


