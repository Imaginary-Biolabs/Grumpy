use crate::dataframe as df_ops;
use crate::io as io_ops;
use crate::layout::GrumpyArray;
use crate::random::GrumpyRng;
use crate::reduce::ReduceOp;
use crate::stream::{BatchPayload, BatchPlan};
use std::cell::RefCell;
use std::sync::mpsc::Receiver;
use pyo3::prelude::*;

#[pyclass(name = "GrumpyArray")]
pub struct PyGrumpyArray {
    pub(crate) inner: GrumpyArray,
}

#[pyclass(name = "Generator")]
pub struct PyGenerator {
    pub(crate) inner: RefCell<GrumpyRng>,
}

#[pyclass(name = "GrumpyDataFrame")]
pub struct PyGrumpyDataFrame {
    pub(crate) inner: df_ops::GrumpyDataFrame,
}

#[pyclass(name = "DataFrameAccessor")]
pub struct PyDataFrameAccessor {
    pub(crate) parent: Py<PyGrumpyDataFrame>,
    // Schema levels path, e.g. ["residue"] or ["molecule","residue"].
    pub(crate) path: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) enum PlanOp {
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
    pub(crate) ops: Vec<PlanOp>,
}

#[pyclass(name = "CompiledBatchesIter")]
pub struct PyCompiledBatchesIter {
    pub(crate) arr_batches: Option<Vec<GrumpyArray>>,
    pub(crate) df_batches: Option<Vec<df_ops::GrumpyDataFrame>>,
    pub(crate) pos: usize,
}

#[pyclass(name = "StreamBatchesIter")]
pub struct PyStreamBatchesIter {
    pub(crate) handle: io_ops::DatasetHandle,
    pub(crate) is_dataframe: bool,
    pub(crate) plan: BatchPlan,
    pub(crate) pos: usize,
    pub(crate) shuffle_within: Option<String>,
    pub(crate) seed: Option<u64>,
    pub(crate) prefetch_rx: Option<Receiver<PyResult<BatchPayload>>>,
}
