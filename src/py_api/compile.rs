use crate::dataframe as df_ops;
use crate::layout::{drop_layout_axes, GrumpyArray, Layout, Leaf, LeafBuffer, OffsetView};
use crate::neighbors as neigh_ops;
use crate::ops::{self, BinOp};
use crate::reduce::{self, ReduceOp, ReduceOutput};
use crate::stream::{self, BatchPayload, BatchPlan};
use crate::io as io_ops;
use crate::dtype::DType;
use crate::py_api::types::{PyCompiledBatchesIter, PyCompiledPlan, PyGrumpyArray, PyGrumpyDataFrame, PlanOp};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use half;
use numpy::Element;
use crate::error::{arg_invalid, arg_must_be_positive, dtype_unsupported, internal, schema_violation, unsupported, unknown_column};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use crate::py_api::py_stream::{prepare_stream_plan_with_shuffle_level, spawn_prefetch_loader};

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
    pub(crate) fn new(spec: Bound<'_, PyAny>) -> PyResult<Self> {
        let seq = spec
            .downcast::<pyo3::types::PyList>()
            .map_err(|_| arg_invalid("spec", "CompiledPlan spec must be a list of dicts", "pass a list of op dicts with an op field per entry."))?;
        let mut ops: Vec<PlanOp> = Vec::with_capacity(seq.len());
        for i in 0..seq.len() {
            let item = seq.get_item(i)?;
            let d = item
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| arg_invalid("spec[]", "each CompiledPlan entry must be a dict", "wrap each op in a dict with an op key."))?;
            let op_obj = d
                .get_item("op")?
                .ok_or_else(|| arg_invalid("spec[]", "missing required field 'op'", "each op dict must include op=<name>."))?;
            let op_name: String = op_obj
                .extract()
                .map_err(|_| arg_invalid("op", "must be a string", "use op names like add_scalar, reduce, df_get."))?;
            let is_int = d
                .get_item("is_int")?
                .and_then(|x| x.extract::<bool>().ok())
                .unwrap_or(false);
            let val_f64 = || -> PyResult<f64> {
                let v = d
                    .get_item("value")?
                    .ok_or_else(|| arg_invalid("value", "scalar op missing required field 'value'", "include value=<number> for scalar ops."))?;
                v.extract::<f64>()
                    .map_err(|_| arg_invalid("value", "must be a number", "pass an int or float for scalar op value."))
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
                        .ok_or_else(|| arg_invalid("k", "neighbors op missing required field 'k'", "include k=<int> for neighbors_knn_self."))?
                        .extract()
                        .map_err(|_| arg_invalid("k", "must be an int", "pass a positive integer neighbor count."))?;
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
                        .ok_or_else(|| arg_invalid("reduce", "reduce op missing required field reduce", "include reduce=sum, mean, min, max, or ptp."))?
                        .extract()
                        .map_err(|_| arg_invalid("reduce", "must be a string", "use reduce names sum, mean, min, max, or ptp."))?;
                    let dim: Option<isize> = d
                        .get_item("dim")?
                        .map(|x| x.extract::<isize>())
                        .transpose()
                        .map_err(|_| arg_invalid("dim", "must be an int", "pass dim=<axis> for reductions."))?;
                    let rop = match which.as_str() {
                        "sum" => ReduceOp::Sum,
                        "mean" => ReduceOp::Mean,
                        "min" => ReduceOp::Min,
                        "max" => ReduceOp::Max,
                        "ptp" => ReduceOp::Ptp,
                        _ => return Err(unsupported("reduce op", "unsupported reduction name", "use sum, mean, min, max, or ptp.")),
                    };
                    ops.push(PlanOp::ReduceCur { op: rop, dim });
                }
                "df_get" => {
                    let level0: String = d
                        .get_item("level0")?
                        .ok_or_else(|| arg_invalid("level0", "df_get missing required field 'level0'", "include level0=<schema level>."))?
                        .extract()
                        .map_err(|_| arg_invalid("level0", "must be a string", "pass a schema level name."))?;
                    let col: String = d
                        .get_item("col")?
                        .ok_or_else(|| arg_invalid("col", "df_get missing required field 'col'", "include col=<column name>."))?
                        .extract()
                        .map_err(|_| arg_invalid("col", "must be a string", "pass the target column name."))?;
                    ops.push(PlanOp::DfGetTmp { level0, col });
                }
                "reduce_tmp" => {
                    let which: String = d
                        .get_item("reduce")?
                        .ok_or_else(|| arg_invalid("reduce", "reduce_tmp missing required field reduce", "include reduce=sum, mean, min, max, or ptp."))?
                        .extract()
                        .map_err(|_| arg_invalid("reduce", "must be a string", "use sum, mean, min, max, or ptp."))?;
                    let dim: isize = d
                        .get_item("dim")?
                        .ok_or_else(|| arg_invalid("dim", "reduce_tmp missing required field 'dim'", "include dim=<axis>."))?
                        .extract()
                        .map_err(|_| arg_invalid("dim", "must be an int", "pass the reduction axis."))?;
                    let rop = match which.as_str() {
                        "sum" => ReduceOp::Sum,
                        "mean" => ReduceOp::Mean,
                        "min" => ReduceOp::Min,
                        "max" => ReduceOp::Max,
                        "ptp" => ReduceOp::Ptp,
                        _ => return Err(unsupported("reduce_tmp", "unsupported reduction name", "use sum, mean, min, max, or ptp.")),
                    };
                    ops.push(PlanOp::ReduceTmp { op: rop, dim });
                }
                "df_set" => {
                    let level0: String = d
                        .get_item("level0")?
                        .ok_or_else(|| arg_invalid("level0", "df_set missing required field 'level0'", "include level0=<schema level>."))?
                        .extract()
                        .map_err(|_| arg_invalid("level0", "must be a string", "pass a schema level name."))?;
                    let col: String = d
                        .get_item("col")?
                        .ok_or_else(|| arg_invalid("col", "df_set missing required field 'col'", "include col=<column name>."))?
                        .extract()
                        .map_err(|_| arg_invalid("col", "must be a string", "pass the target column name."))?;
                    ops.push(PlanOp::DfSetTmp { level0, col });
                }
                _ => {
                    return Err(unsupported("CompiledPlan", format!("unknown op '{op_name}'"), "use a supported op name from the CompiledPlan API."))
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
            return Err(arg_invalid("batch", "expects a GrumpyArray or GrumpyDataFrame", "pass a grumpy array or dataframe batch to run()."));
        }
        let mut tmp: Option<GrumpyArray> = None;

        for op in &self.ops {
            match op {
                PlanOp::AddScalar { value, is_int } => {
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "add_scalar requires an array batch", "ensure the pipeline starts with an array input."))?;
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
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "sub_scalar requires an array batch", "ensure the pipeline starts with an array input."))?;
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
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "mul_scalar requires an array batch", "ensure the pipeline starts with an array input."))?;
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
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "div_scalar requires an array batch", "ensure the pipeline starts with an array input."))?;
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
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "mod_scalar requires an array batch", "ensure the pipeline starts with an array input."))?;
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
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "mul_scalar_sum_all requires an array batch", "ensure the pipeline starts with an array input."))?;
                    if !is_int {
                        return Err(arg_invalid("value", "mul_scalar_sum_all requires an integer scalar", "pass is_int=True with an integral value."));
                    }
                    let s = *value as i32;
                    let sum = py.allow_threads(move || ops::mul_scalar_sum_all_i64(&a0, s))?;
                    cur_scalar = Some(sum.to_object(py));
                }
                PlanOp::NeighborsKnnSelf { k, dim, loop_ } => {
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "neighbors requires an array batch", "ensure the pipeline starts with an array input."))?;
                    let kk = *k;
                    let dd = *dim;
                    let lp = *loop_;
                    let a = py.allow_threads(move || neigh_ops::neighbors(&a0, &a0, Some(kk), None, dd, lp))?;
                    cur_arr = Some(a);
                }
                PlanOp::ReduceCur { op, dim } => {
                    let a0 = cur_arr.take().ok_or_else(|| arg_invalid("batch", "reduce requires an array batch", "ensure the pipeline starts with an array input."))?;
                    match reduce::reduce(py, &a0, *dim, *op)? {
                        ReduceOutput::Array(out) => cur_arr = Some(out),
                        ReduceOutput::Scalar(obj) => cur_scalar = Some(obj),
                    }
                }
                PlanOp::DfGetTmp { level0, col } => {
                    let df = cur_df
                        .as_ref()
                        .ok_or_else(|| arg_invalid("batch", "df_get requires a dataframe batch", "ensure the pipeline starts with a dataframe input."))?
                        .clone();
                    tmp = Some(df_get_level0_column(&df, level0, col)?);
                }
                PlanOp::ReduceTmp { op, dim } => {
                    let a = tmp.take().ok_or_else(|| arg_invalid("tmp", "reduce_tmp missing temporary array", "precede reduce_tmp with df_get."))?;
                    match reduce::reduce(py, &a, Some(*dim), *op)? {
                        ReduceOutput::Array(out) => tmp = Some(out),
                        ReduceOutput::Scalar(_) => return Err(unsupported("reduce_tmp", "scalar reduction result cannot be assigned to dataframe column", "use reduce with an explicit dim that returns an array.")),
                    }
                }
                PlanOp::DfSetTmp { level0, col } => {
                    let mut df = cur_df
                        .take()
                        .ok_or_else(|| arg_invalid("batch", "df_set requires a dataframe batch", "ensure the pipeline starts with a dataframe input."))?;
                    let rhs = tmp.take().ok_or_else(|| arg_invalid("tmp", "df_set missing temporary value", "precede df_set with reduce_tmp or df_get."))?;
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
        Err(arg_invalid("run", "CompiledPlan produced no result", "ensure the plan ends with a supported output op."))
    }
}

fn run_plan_array_rust(ops_plan: &[PlanOp], mut cur: GrumpyArray, gpu: crate::gpu::GpuPreference) -> PyResult<GrumpyArray> {
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
                    return Err(arg_invalid("value", "mul_scalar_sum_all requires an integer scalar", "pass is_int=True with an integral value."));
                }
                let s = *value as i32;
                let sum = ops::mul_scalar_sum_all_i64(&cur, s)?;
                return Err(unsupported(
                    "mul_scalar_sum_all",
                    format!("produced scalar {sum}; array pipeline cannot consume scalars"),
                    "use a reduction that returns an array or run outside a compiled array pipeline.",
                ));
            }
            PlanOp::NeighborsKnnSelf { k, dim, loop_ } => {
                cur = neigh_ops::neighbors_with_gpu(&cur, &cur, Some(*k), None, *dim, *loop_, gpu)?;
            }
            PlanOp::ReduceCur { op, dim } => {
                match dim {
                    Some(d) => cur = reduce::reduce_array(&cur, *d, *op)?,
                    None => {
                        return Err(unsupported(
                            "reduce",
                            "reduce without dim in array Rust plan is not supported",
                            "pass dim= explicitly for array reductions in compiled pipelines.",
                        ));
                    }
                }
            }
            _ => {
                return Err(unsupported(
                    "compiled array pipeline",
                    "Rust scheduled pipelines support scalar ops, neighbors, and reductions on arrays",
                    "use supported ops or run the step outside the compiled pipeline.",
                ));
            }
        }
    }
    Ok(cur)
}

fn run_plan_df_rust(ops_plan: &[PlanOp], mut cur: df_ops::GrumpyDataFrame) -> PyResult<df_ops::GrumpyDataFrame> {
    let mut tmp: Option<GrumpyArray> = None;
    for op in ops_plan {
        match op {
            PlanOp::DfGetTmp { level0, col } => {
                tmp = Some(df_get_level0_column(&cur, level0, col)?);
            }
            PlanOp::ReduceTmp { op, dim } => {
                let a = tmp.take().ok_or_else(|| arg_invalid("tmp", "reduce_tmp missing temporary array", "precede reduce_tmp with df_get."))?;
                tmp = Some(reduce::reduce_array(&a, *dim, *op)?);
            }
            PlanOp::DfSetTmp { level0, col } => {
                let rhs = tmp.take().ok_or_else(|| arg_invalid("tmp", "df_set missing temporary value", "precede df_set with reduce_tmp or df_get."))?;
                df_set_level0_column(&mut cur, level0, col, rhs)?;
            }
            _ => {
                return Err(unsupported(
                    "compiled dataframe pipeline",
                    "Rust scheduled pipelines support df_get, reduce_tmp, and df_set",
                    "use supported dot-notation assignment ops in the compiled plan.",
                ));
            }
        }
    }
    Ok(cur)
}

#[pyfunction]
#[pyo3(signature = (path, batch_size, drop_last, cpu, _prefetch, spec, batch_on=None, shuffle=None, seed=None, world_size=1, rank=0, batch_indices=None, gpu="false"))]
pub fn compiled_stream_apply(
    _py: Python<'_>,
    path: String,
    batch_size: usize,
    drop_last: bool,
    cpu: usize,
    _prefetch: usize,
    spec: Bound<'_, PyAny>,
    batch_on: Option<String>,
    shuffle: Option<String>,
    seed: Option<u64>,
    world_size: usize,
    rank: usize,
    batch_indices: Option<Vec<usize>>,
    gpu: &str,
) -> PyResult<PyCompiledBatchesIter> {
    let gpu_pref = crate::gpu::GpuPreference::parse(gpu)?;
    if cpu < 1 {
        return Err(arg_must_be_positive("cpu", cpu));
    }
    if batch_size == 0 {
        return Err(arg_must_be_positive("batch_size", batch_size));
    }
    let plan_ops = PyCompiledPlan::new(spec)?;
    let shuffle_arg = shuffle.as_deref();
    let (handle, plan, is_df, _shuffle_within, _seed) = prepare_stream_plan_with_shuffle_level(
        &path,
        batch_size,
        drop_last,
        batch_on.as_deref(),
        shuffle_arg,
        seed,
        world_size,
        rank,
        batch_indices.as_deref(),
        false,
    )?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(cpu)
        .build()
        .map_err(|e| internal("compiled_stream_apply", format!("failed to build thread pool: {e}")))?;
    if is_df {
        let handle = handle.clone();
        let results: Vec<PyResult<df_ops::GrumpyDataFrame>> = pool.install(|| {
            plan.batches
                .par_iter()
                .map(|batch| {
                    let payload = stream::load_batch(&handle, batch)?;
                    match payload {
                        stream::BatchPayload::DataFrame(df) => run_plan_df_rust(&plan_ops.ops, df),
                        _ => Err(internal("compiled_stream_apply", "array payload for dataframe stream")),
                    }
                })
                .collect()
        });
        let mut outs: Vec<df_ops::GrumpyDataFrame> = Vec::with_capacity(results.len());
        for r in results {
            outs.push(r?);
        }
        return Ok(PyCompiledBatchesIter {
            arr_batches: None,
            df_batches: Some(outs),
            pos: 0,
        });
    }
    let handle = handle.clone();
    let results: Vec<PyResult<GrumpyArray>> = pool.install(|| {
        plan.batches
            .par_iter()
            .map(|batch| {
                let payload = stream::load_batch(&handle, batch)?;
                match payload {
                    stream::BatchPayload::Array(arr) => run_plan_array_rust(&plan_ops.ops, arr, gpu_pref),
                    _ => Err(internal("compiled_stream_apply", "dataframe payload for array stream")),
                }
            })
            .collect()
    });
    let mut outs: Vec<GrumpyArray> = Vec::with_capacity(results.len());
    for r in results {
        outs.push(r?);
    }
    Ok(PyCompiledBatchesIter {
        arr_batches: Some(outs),
        df_batches: None,
        pos: 0,
    })
}

pub(crate) fn df_get_level0_column(df: &df_ops::GrumpyDataFrame, level0: &str, colname: &str) -> PyResult<GrumpyArray> {
    let schema = df.schema.as_ref().ok_or_else(|| schema_violation("dot-notation requires a schema", "dataframe has no schema=", "pass schema= when constructing the dataframe."))?;
    let level = *schema
        .name_to_level
        .get(level0)
        .ok_or_else(|| schema_violation("invalid schema path", "level0 is not a declared schema level", "use a name from schema=."))?;
    // Find column
    let mut col: Option<GrumpyArray> = None;
    for (n, c) in df.names.iter().zip(df.cols.iter()) {
        if n == colname {
            col = Some(c.clone());
            break;
        }
    }
    let col = col.ok_or_else(|| unknown_column(colname))?;
    let layout = drop_layout_axes(&col.layout, level)?;
    Ok(GrumpyArray { dtype: col.dtype, layout })
}

pub(crate) fn df_set_level0_column(df: &mut df_ops::GrumpyDataFrame, level0: &str, colname: &str, rhs: GrumpyArray) -> PyResult<()> {
    let schema = df.schema.as_ref().ok_or_else(|| schema_violation("dot-notation assignment requires a schema", "dataframe has no schema=", "pass schema= when constructing the dataframe."))?;
    let level = *schema
        .name_to_level
        .get(level0)
        .ok_or_else(|| schema_violation("invalid schema path", "level0 is not a declared schema level", "use a name from schema=."))?;
    let col_level = schema.level_for_column(colname)?;

    let rhs2 = if level == col_level {
        df.renest_rhs_for_level(level, level0, rhs)?
    } else {
        rhs
    };
    df.set_column_array(colname.to_string(), rhs2)
}

pub(crate) fn scalar_like(dt: DType, value: f64, is_int: bool) -> PyResult<GrumpyArray> {
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
        DType::Bool => return Err(dtype_unsupported("compiled scalar ops", DType::Bool)),
        DType::Char | DType::String => return Err(dtype_unsupported("compiled scalar ops", dt)),
    };
    if is_int {
        // ok; just a hint today
    }
    Ok(GrumpyArray { dtype: dt, layout: Layout::Leaf(leaf) })
}
