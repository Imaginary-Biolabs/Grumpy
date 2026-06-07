use crate::dataframe::{CanonShape, Schema};
use crate::dataframe as df_ops;
use crate::io::OpenSession;
use crate::layout::GrumpyArray;
use crate::random::GrumpyRng;
use crate::reduce::ReduceOp;
use std::cell::RefCell;
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
    pub(crate) index_level: usize,
}

/// Lazy on-disk dataframe handle returned by :func:`grumpy.open`.
#[pyclass(name = "OpenDataFrame")]
pub struct PyOpenDataFrame {
    pub(crate) session: OpenSession,
    pub(crate) column_names: Option<Vec<String>>,
    pub(crate) schema: Option<Schema>,
    pub(crate) canon: CanonShape,
    pub(crate) index_depth: usize,
}

/// Lazy column proxy; indexing materializes a :class:`GrumpyArray`.
#[pyclass(name = "OpenColumn")]
pub struct PyOpenColumn {
    pub(crate) session: OpenSession,
    pub(crate) column_name: String,
    pub(crate) column_names: Option<Vec<String>>,
    pub(crate) schema: Option<Schema>,
    pub(crate) canon: CanonShape,
    pub(crate) index_depth: usize,
    /// Schema axes to drop for dot-notation column access.
    pub(crate) drop_axes: usize,
    /// When true, flatten nested column to leaf before indexing (``open.col`` access).
    pub(crate) flatten_to_leaf: bool,
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

