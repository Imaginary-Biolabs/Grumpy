use crate::layout::{build_array, GrumpyArray};
use crate::ops::{self, BinOp};
use crate::dtype::DType;
use crate::py_api::types::PyGrumpyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};

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

pub(crate) fn apply_elementwise_binop(
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
