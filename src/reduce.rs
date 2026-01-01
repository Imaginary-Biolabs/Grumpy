//! Reductions (`sum`, `mean`, `min`, `max`, `ptp`) for `GrumpyArray`.
//!
//! Key design points:
//! - Works over **pure list-chains** (no unions) with a recursive engine for arbitrary depth.
//! - Provides **rectangular 2D fast paths** (single tight loop) and uses them whenever possible.
//! - Provides a **no-GIL reduction-to-layout** (`reduce_array`) for Rust scheduling of compiled pipelines:
//!   it never creates Python scalars; leaf reductions are materialized as 1-element `Leaf` layouts.
//!
//! Adding a new reduction:
//! - Extend `ReduceOp` and wire it in `py_api.rs` + `python/grumpy/__init__.py`.
//! - Implement:
//!   - 2D fast path(s) if it’s performance critical (`reduce_rect2d_fast`)
//!   - leaf-to-scalar logic for both GIL (`reduce_leaf_to_scalar`) and no-GIL (`reduce_leaf_to_scalar_value`)
//! - Add parity tests for deep list-chains and for streamed `OffsetView` batches.

use crate::dtype::DType;
use crate::layout::{offsetview_to_listoffset, GrumpyArray, Layout, Leaf, LeafBuffer};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use half::f16;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use std::string::String;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    Mean,
    Ptp,
}

pub enum ReduceOutput {
    Scalar(PyObject),
    Array(GrumpyArray),
}

/// Reduce and require an array output (used by Rust-scheduled compiled pipelines).
///
/// Returns an error if the reduction would produce a Python scalar (e.g. reducing a 1D leaf).
pub fn reduce_array(arr: &GrumpyArray, dim: isize, op: ReduceOp) -> PyResult<GrumpyArray> {
    // Full, no-GIL reduction-to-layout implementation for pure list-chains.
    //
    // Key property: never needs to produce Python scalars. Any leaf-to-scalar reductions are
    // materialized as a 1-element `Leaf` layout directly in Rust, and then stacked.

    let norm_layout;
    let layout: &Layout = match &arr.layout {
        Layout::OffsetView(v) => {
            norm_layout = Layout::ListOffset(offsetview_to_listoffset(v)?);
            &norm_layout
        }
        _ => &arr.layout,
    };

    let depth = crate::layout::list_chain_depth(layout)
        .ok_or_else(|| PyValueError::new_err("reduce currently only supports pure list-chain arrays."))?;
    let axis = normalize_axis(dim, depth)?;

    // Scalar outputs are not supported in Rust scheduling (returning PyObject would require the GIL).
    if depth == 0 {
        return Err(PyValueError::new_err(
            "Rust scheduled reductions do not support scalar outputs (reduce on 1D leaf).",
        ));
    }

    // 2D list->leaf fast paths first.
    if depth == 1 {
        if let Some(out) = reduce_rect2d_fast(layout, arr.dtype, dim, op)? {
            return Ok(out);
        }
        return match axis {
            0 => Ok(reduce_2d_dim0_to_leaf(layout, arr.dtype, op)?),
            1 => Ok(reduce_2d_dim1_to_leaf(layout, arr.dtype, op)?),
            _ => Err(PyValueError::new_err("Invalid dim for 2D reduction.")),
        };
    }

    let out_dt = reduce_out_dtype(arr.dtype, op)?;
    let out_layout = reduce_list_chain_to_layout_nogil(layout, arr.dtype, out_dt, depth, axis, op)?;
    Ok(GrumpyArray { dtype: out_dt, layout: out_layout })
}

pub fn reduce(py: Python<'_>, arr: &GrumpyArray, dim: isize, op: ReduceOp) -> PyResult<ReduceOutput> {
    let norm_layout;
    let layout: &Layout = match &arr.layout {
        Layout::OffsetView(v) => {
            norm_layout = Layout::ListOffset(offsetview_to_listoffset(v)?);
            &norm_layout
        }
        _ => &arr.layout,
    };

    // dim is interpreted like NumPy axis (0 is outermost).
    // For list-chains: depth == number of list levels; valid axes are 0..=depth (inclusive),
    // where axis==depth reduces the leaf values inside the deepest list level.
    let depth = crate::layout::list_chain_depth(layout)
        .ok_or_else(|| PyValueError::new_err("reduce currently only supports pure list-chain arrays."))?;

    let axis = normalize_axis(dim, depth)?;

    if depth == 0 {
        // 1D leaf: reduce over axis=0 => scalar
        if axis != 0 {
            return Err(PyValueError::new_err("Invalid dim for 1D reduction."));
        }
        return Ok(ReduceOutput::Scalar(reduce_leaf_to_scalar(py, layout, arr.dtype, op)?));
    }

    // Fast path: rectangular 2D list->leaf with all-valid leaf.
    if depth == 1 {
        if let Some(out) = reduce_rect2d_fast(layout, arr.dtype, dim, op)? {
            return Ok(ReduceOutput::Array(out));
        }
        return match axis {
            0 => Ok(ReduceOutput::Array(reduce_2d_dim0_to_leaf(layout, arr.dtype, op)?)),
            1 => Ok(ReduceOutput::Array(reduce_2d_dim1_to_leaf(layout, arr.dtype, op)?)),
            _ => Err(PyValueError::new_err("Invalid dim for 2D reduction.")),
        };
    }

    let out_dt = reduce_out_dtype(arr.dtype, op)?;
    let out_layout = reduce_list_chain_to_layout(py, layout, arr.dtype, out_dt, depth, axis, op)?;
    Ok(ReduceOutput::Array(GrumpyArray { dtype: out_dt, layout: out_layout }))
}

// ---------------- no-GIL deep reduction engine (layout output) ----------------

#[derive(Clone, Copy, Debug)]
enum ScalarValue {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
}

fn scalar_to_leaf_layout(dt: DType, v: Option<ScalarValue>) -> PyResult<Layout> {
    if v.is_none() {
        return Ok(scalar_null_leaf(dt));
    }
    let v = v.unwrap();
    let mut leaf = Leaf::new(dt);
    leaf.len = 1;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; 1]);
    leaf.buffer = match (dt, v) {
        (DType::Int8, ScalarValue::I64(x)) => LeafBuffer::I8(Arc::new(vec![x as i8])),
        (DType::Int16, ScalarValue::I64(x)) => LeafBuffer::I16(Arc::new(vec![x as i16])),
        (DType::Int32, ScalarValue::I64(x)) => LeafBuffer::I32(Arc::new(vec![x as i32])),
        (DType::Int64, ScalarValue::I64(x)) => LeafBuffer::I64(Arc::new(vec![x])),
        (DType::UInt8, ScalarValue::U64(x)) => LeafBuffer::U8(Arc::new(vec![x as u8])),
        (DType::UInt16, ScalarValue::U64(x)) => LeafBuffer::U16(Arc::new(vec![x as u16])),
        (DType::UInt32, ScalarValue::U64(x)) => LeafBuffer::U32(Arc::new(vec![x as u32])),
        (DType::UInt64, ScalarValue::U64(x)) => LeafBuffer::U64(Arc::new(vec![x])),
        (DType::Float16, ScalarValue::F64(x)) => LeafBuffer::F16(Arc::new(vec![f16::from_f64(x).to_bits()])),
        (DType::Float32, ScalarValue::F64(x)) => LeafBuffer::F32(Arc::new(vec![x as f32])),
        (DType::Float64, ScalarValue::F64(x)) => LeafBuffer::F64(Arc::new(vec![x])),
        (DType::Bool, ScalarValue::Bool(x)) => LeafBuffer::Bool(Arc::new(vec![if x { 1u8 } else { 0u8 }])),
        // Allow int->float outputs for Mean/Sum rules.
        (DType::Float64, ScalarValue::I64(x)) => LeafBuffer::F64(Arc::new(vec![x as f64])),
        (DType::Float64, ScalarValue::U64(x)) => LeafBuffer::F64(Arc::new(vec![x as f64])),
        // Sum on bool returns int64 (per reduce_out_dtype)
        (DType::Int64, ScalarValue::Bool(x)) => LeafBuffer::I64(Arc::new(vec![if x { 1 } else { 0 }])),
        (DType::Int64, ScalarValue::U64(x)) => LeafBuffer::I64(Arc::new(vec![x as i64])),
        _ => {
            return Err(PyValueError::new_err(
                "Internal error: unsupported scalar-to-leaf conversion in no-GIL reduction.",
            ))
        }
    };
    Ok(Layout::Leaf(leaf))
}

fn reduce_leaf_to_scalar_value(layout: &Layout, dt: DType, op: ReduceOp) -> PyResult<Option<ScalarValue>> {
    let leaf = match layout {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Internal error: expected leaf layout.")),
    };
    if leaf.len == 0 {
        return Ok(None);
    }
    match (dt, &leaf.buffer) {
        (DType::Bool, LeafBuffer::Bool(v)) => {
            let mut any = false;
            let mut sum: i64 = 0;
            let mut minv: u8 = 1;
            let mut maxv: u8 = 0;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] {
                    continue;
                }
                any = true;
                let x = v[i];
                sum += x as i64;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any {
                return Ok(None);
            }
            Ok(Some(match op {
                ReduceOp::Sum => ScalarValue::I64(sum),
                ReduceOp::Mean => ScalarValue::F64(sum as f64 / (leaf.validity.count_ones() as f64)),
                ReduceOp::Min => ScalarValue::Bool(minv != 0),
                ReduceOp::Max => ScalarValue::Bool(maxv != 0),
                ReduceOp::Ptp => ScalarValue::Bool((maxv - minv) != 0),
            }))
        }
        (DType::Int32, LeafBuffer::I32(v)) => {
            let mut any = false;
            let mut count: i64 = 0;
            let mut sum: i64 = 0;
            let mut minv: i64 = i64::MAX;
            let mut maxv: i64 = i64::MIN;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = v[i] as i64;
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            Ok(Some(match op {
                ReduceOp::Sum => ScalarValue::I64(sum),
                ReduceOp::Mean => ScalarValue::F64(sum as f64 / count as f64),
                ReduceOp::Min => ScalarValue::I64(minv),
                ReduceOp::Max => ScalarValue::I64(maxv),
                ReduceOp::Ptp => ScalarValue::I64(maxv - minv),
            }))
        }
        (DType::Int64, LeafBuffer::I64(v)) => {
            let mut any = false;
            let mut count: i64 = 0;
            let mut sum: i64 = 0;
            let mut minv: i64 = i64::MAX;
            let mut maxv: i64 = i64::MIN;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = v[i];
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            Ok(Some(match op {
                ReduceOp::Sum => ScalarValue::I64(sum),
                ReduceOp::Mean => ScalarValue::F64(sum as f64 / count as f64),
                ReduceOp::Min => ScalarValue::I64(minv),
                ReduceOp::Max => ScalarValue::I64(maxv),
                ReduceOp::Ptp => ScalarValue::I64(maxv - minv),
            }))
        }
        (DType::UInt32, LeafBuffer::U32(v)) => {
            let mut any = false;
            let mut count: u64 = 0;
            let mut sum: u64 = 0;
            let mut minv: u64 = u64::MAX;
            let mut maxv: u64 = 0;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = v[i] as u64;
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            Ok(Some(match op {
                ReduceOp::Sum => ScalarValue::I64(sum as i64),
                ReduceOp::Mean => ScalarValue::F64(sum as f64 / count as f64),
                ReduceOp::Min => ScalarValue::U64(minv),
                ReduceOp::Max => ScalarValue::U64(maxv),
                ReduceOp::Ptp => ScalarValue::U64(maxv - minv),
            }))
        }
        (DType::UInt64, LeafBuffer::U64(v)) => {
            let mut any = false;
            let mut count: u64 = 0;
            let mut sum: u64 = 0;
            let mut minv: u64 = u64::MAX;
            let mut maxv: u64 = 0;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = v[i];
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            Ok(Some(match op {
                ReduceOp::Sum => ScalarValue::I64(sum as i64),
                ReduceOp::Mean => ScalarValue::F64(sum as f64 / count as f64),
                ReduceOp::Min => ScalarValue::U64(minv),
                ReduceOp::Max => ScalarValue::U64(maxv),
                ReduceOp::Ptp => ScalarValue::U64(maxv - minv),
            }))
        }
        (DType::Float32, LeafBuffer::F32(v)) => {
            let mut any = false;
            let mut count: i64 = 0;
            let mut sum: f64 = 0.0;
            let mut minv: f64 = f64::INFINITY;
            let mut maxv: f64 = f64::NEG_INFINITY;
            let mut any_nan = false;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = v[i] as f64;
                if x.is_nan() { any_nan = true; }
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            let out = match op {
                ReduceOp::Sum => ScalarValue::F64(sum),
                ReduceOp::Mean => ScalarValue::F64(sum / (count as f64)),
                ReduceOp::Min => ScalarValue::F64(if any_nan { f64::NAN } else { minv }),
                ReduceOp::Max => ScalarValue::F64(if any_nan { f64::NAN } else { maxv }),
                ReduceOp::Ptp => ScalarValue::F64(if any_nan { f64::NAN } else { maxv - minv }),
            };
            Ok(Some(out))
        }
        (DType::Float64, LeafBuffer::F64(v)) => {
            let mut any = false;
            let mut count: i64 = 0;
            let mut sum: f64 = 0.0;
            let mut minv: f64 = f64::INFINITY;
            let mut maxv: f64 = f64::NEG_INFINITY;
            let mut any_nan = false;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = v[i];
                if x.is_nan() { any_nan = true; }
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            let out = match op {
                ReduceOp::Sum => ScalarValue::F64(sum),
                ReduceOp::Mean => ScalarValue::F64(sum / (count as f64)),
                ReduceOp::Min => ScalarValue::F64(if any_nan { f64::NAN } else { minv }),
                ReduceOp::Max => ScalarValue::F64(if any_nan { f64::NAN } else { maxv }),
                ReduceOp::Ptp => ScalarValue::F64(if any_nan { f64::NAN } else { maxv - minv }),
            };
            Ok(Some(out))
        }
        (DType::Float16, LeafBuffer::F16(v)) => {
            let mut any = false;
            let mut count: i64 = 0;
            let mut sum: f64 = 0.0;
            let mut minv: f64 = f64::INFINITY;
            let mut maxv: f64 = f64::NEG_INFINITY;
            let mut any_nan = false;
            for i in 0..leaf.len {
                if leaf.has_nulls && !leaf.validity[i] { continue; }
                any = true;
                count += 1;
                let x = f16::from_bits(v[i]).to_f64();
                if x.is_nan() { any_nan = true; }
                sum += x;
                if x < minv { minv = x; }
                if x > maxv { maxv = x; }
            }
            if !any { return Ok(None); }
            let out = match op {
                ReduceOp::Sum => ScalarValue::F64(sum),
                ReduceOp::Mean => ScalarValue::F64(sum / (count as f64)),
                ReduceOp::Min => ScalarValue::F64(if any_nan { f64::NAN } else { minv }),
                ReduceOp::Max => ScalarValue::F64(if any_nan { f64::NAN } else { maxv }),
                ReduceOp::Ptp => ScalarValue::F64(if any_nan { f64::NAN } else { maxv - minv }),
            };
            Ok(Some(out))
        }
        _ => Err(PyValueError::new_err("Unsupported dtype for reduction.")),
    }
}

fn reduce_list_chain_to_layout_nogil(
    layout: &Layout,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    axis: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        Layout::OffsetView(v) => {
            return reduce_list_chain_to_layout_nogil(
                &Layout::ListOffset(offsetview_to_listoffset(v)?),
                in_dt,
                out_dt,
                depth,
                axis,
                op,
            );
        }
        _ => return Err(PyValueError::new_err("Internal error: expected ListOffset at top.")),
    };
    if axis == 0 {
        return reduce_axis0_listoffset_nogil(lo, in_dt, out_dt, depth, op);
    }
    reduce_axis_gt0_listoffset_nogil(lo, in_dt, out_dt, depth, axis, op)
}

fn reduce_axis_gt0_listoffset_nogil(
    lo: &crate::layout::ListOffset,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    axis: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    let n = lo.len();
    let mut elems: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let el = crate::layout::drop_axis0_select_element(&Layout::ListOffset(lo.clone()), i)?;
        let el_depth = depth - 1;
        let reduced = if el_depth == 0 {
            if axis - 1 != 0 {
                return Err(PyValueError::new_err("dim out of range."));
            }
            let s = reduce_leaf_to_scalar_value(&el, in_dt, op)?;
            scalar_to_leaf_layout(out_dt, s)?
        } else if el_depth == 1 {
            let arr = GrumpyArray { dtype: in_dt, layout: el };
            match axis - 1 {
                0 => reduce_2d_dim0_to_leaf(&arr.layout, in_dt, op)?.layout,
                1 => reduce_2d_dim1_to_leaf(&arr.layout, in_dt, op)?.layout,
                _ => return Err(PyValueError::new_err("dim out of range.")),
            }
        } else {
            reduce_list_chain_to_layout_nogil(&el, in_dt, out_dt, el_depth, axis - 1, op)?
        };
        elems.push(reduced);
    }
    stack_elements_as_list(out_dt, &elems)
}

fn reduce_axis0_listoffset_nogil(
    lo: &crate::layout::ListOffset,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    if depth == 1 {
        return Ok(reduce_2d_dim0_to_leaf(&Layout::ListOffset(lo.clone()), in_dt, op)?.layout);
    }
    let nrows = lo.len();
    let mut maxlen: usize = 0;
    for i in 0..nrows {
        let len_i = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
        if len_i > maxlen {
            maxlen = len_i;
        }
    }
    let mut reduced_cols: Vec<Layout> = Vec::with_capacity(maxlen);
    for j in 0..maxlen {
        let mut children: Vec<Layout> = Vec::new();
        for i in 0..nrows {
            let start = lo.offsets[i] as usize;
            let end = lo.offsets[i + 1] as usize;
            if start + j < end {
                let idx = start + j;
                let child = crate::layout::drop_axis0_select_element(lo.content.as_ref(), idx)?;
                children.push(child);
            }
        }
        if children.is_empty() {
            reduced_cols.push(scalar_null_leaf(out_dt));
            continue;
        }
        let stacked = build_stack_layout(&children)?;
        let stacked_depth = depth - 1;
        let reduced_j = if stacked_depth == 0 {
            let s = reduce_leaf_to_scalar_value(&stacked, in_dt, op)?;
            scalar_to_leaf_layout(out_dt, s)?
        } else if stacked_depth == 1 {
            let arr = GrumpyArray { dtype: in_dt, layout: stacked };
            reduce_2d_dim0_to_leaf(&arr.layout, in_dt, op)?.layout
        } else {
            reduce_list_chain_to_layout_nogil(&stacked, in_dt, out_dt, stacked_depth, 0, op)?
        };
        reduced_cols.push(reduced_j);
    }
    stack_elements_as_list(out_dt, &reduced_cols)
}

fn normalize_axis(dim: isize, depth: usize) -> PyResult<usize> {
    let ndims = depth as isize + 1;
    let mut d = dim;
    if d < 0 {
        d += ndims;
    }
    if d < 0 || d >= ndims {
        return Err(PyValueError::new_err("dim out of range."));
    }
    Ok(d as usize)
}

fn reduce_out_dtype(in_dt: DType, op: ReduceOp) -> PyResult<DType> {
    match op {
        ReduceOp::Mean => Ok(DType::Float64),
        ReduceOp::Sum => match in_dt {
            DType::Float16 | DType::Float32 | DType::Float64 => Ok(DType::Float64),
            DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64
            | DType::Bool => Ok(DType::Int64),
            DType::Char | DType::String => Err(PyValueError::new_err(
                "sum/mean are only supported for numeric dtypes.",
            )),
        },
        ReduceOp::Min | ReduceOp::Max | ReduceOp::Ptp => match in_dt {
            DType::Char | DType::String => Err(PyValueError::new_err(
                "min/max/ptp are only supported for numeric dtypes.",
            )),
            _ => Ok(in_dt),
        },
    }
}

fn reduce_list_chain_to_layout(
    py: Python<'_>,
    layout: &Layout,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    axis: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    // depth >= 2 here.
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Internal error: expected ListOffset at top.")),
    };

    if axis == 0 {
        return reduce_axis0_listoffset(py, lo, in_dt, out_dt, depth, op);
    }
    reduce_axis_gt0_listoffset(py, lo, in_dt, out_dt, depth, axis, op)
}

fn reduce_axis_gt0_listoffset(
    py: Python<'_>,
    lo: &crate::layout::ListOffset,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    axis: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    // Reduce inside each element independently (axis-1 on the element).
    let n = lo.len();
    let mut elems: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let el = crate::layout::drop_axis0_select_element(&Layout::ListOffset(lo.clone()), i)?;
        let el_depth = depth - 1;
        let reduced = if el_depth == 0 {
            // scalar leaf: only axis=0 is valid
            if axis - 1 != 0 {
                return Err(PyValueError::new_err("dim out of range."));
            }
            let s = reduce_leaf_to_scalar(py, &el, in_dt, op)?;
            scalar_py_to_leaf(out_dt, py, &s)?
        } else if el_depth == 1 {
            // 2D inside element
            let arr = GrumpyArray { dtype: in_dt, layout: el };
            match axis - 1 {
                0 => reduce_2d_dim0_to_leaf(&arr.layout, in_dt, op)?.layout,
                1 => reduce_2d_dim1_to_leaf(&arr.layout, in_dt, op)?.layout,
                _ => return Err(PyValueError::new_err("dim out of range.")),
            }
        } else {
            reduce_list_chain_to_layout(py, &el, in_dt, out_dt, el_depth, axis - 1, op)?
        };
        elems.push(reduced);
    }
    stack_elements_as_list(out_dt, &elems)
}

fn reduce_axis0_listoffset(
    py: Python<'_>,
    lo: &crate::layout::ListOffset,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    // Reduce across axis0 by positional alignment (Awkward-like): missing positions are skipped.
    // For depth==2, use existing 2D dim0 kernel (already Awkward-like after our semantic tweak below).
    if depth == 1 {
        return Ok(reduce_2d_dim0_to_leaf(&Layout::ListOffset(lo.clone()), in_dt, op)?.layout);
    }

    let nrows = lo.len();
    let mut maxlen: usize = 0;
    for i in 0..nrows {
        let len_i = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
        if len_i > maxlen {
            maxlen = len_i;
        }
    }

    let mut reduced_cols: Vec<Layout> = Vec::with_capacity(maxlen);
    for j in 0..maxlen {
        // Collect the j-th child from each row that has it.
        let mut children: Vec<Layout> = Vec::new();
        for i in 0..nrows {
            let start = lo.offsets[i] as usize;
            let end = lo.offsets[i + 1] as usize;
            if start + j < end {
                let idx = start + j;
                let child = crate::layout::drop_axis0_select_element(lo.content.as_ref(), idx)?;
                children.push(child);
            }
        }
        if children.is_empty() {
            // No values in this column: represent as null scalar.
            reduced_cols.push(scalar_null_leaf(out_dt));
            continue;
        }

        // Stack children as a list and reduce its axis0.
        let stacked = build_stack_layout(&children)?;
        let stacked_depth = depth - 1;
        let reduced_j = if stacked_depth == 0 {
            let s = reduce_leaf_to_scalar(py, &stacked, in_dt, op)?;
            scalar_py_to_leaf(out_dt, py, &s)?
        } else if stacked_depth == 1 {
            // 2D case within recursion
            let arr = GrumpyArray { dtype: in_dt, layout: stacked };
            reduce_2d_dim0_to_leaf(&arr.layout, in_dt, op)?.layout
        } else {
            reduce_list_chain_to_layout(py, &stacked, in_dt, out_dt, stacked_depth, 0, op)?
        };
        reduced_cols.push(reduced_j);
    }

    stack_elements_as_list(out_dt, &reduced_cols)
}

fn scalar_null_leaf(dt: DType) -> Layout {
    let mut leaf = Leaf::new(dt);
    leaf.len = 1;
    leaf.has_nulls = true;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 0; 1]);
    leaf.buffer = match dt {
        DType::Int8 => LeafBuffer::I8(Arc::new(vec![0i8])),
        DType::Int16 => LeafBuffer::I16(Arc::new(vec![0i16])),
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64])),
        DType::UInt8 => LeafBuffer::U8(Arc::new(vec![0u8])),
        DType::UInt16 => LeafBuffer::U16(Arc::new(vec![0u16])),
        DType::UInt32 => LeafBuffer::U32(Arc::new(vec![0u32])),
        DType::UInt64 => LeafBuffer::U64(Arc::new(vec![0u64])),
        DType::Float16 => LeafBuffer::F16(Arc::new(vec![0u16])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64])),
        DType::Bool => LeafBuffer::Bool(Arc::new(vec![0u8])),
        DType::Char => LeafBuffer::Char(Arc::new(vec![0u32])),
        DType::String => LeafBuffer::String(Arc::new(vec![String::new()])),
    };
    Layout::Leaf(leaf)
}

fn scalar_py_to_leaf(dt: DType, py: Python<'_>, obj: &PyObject) -> PyResult<Layout> {
    // Convert a Python scalar/None into a single-element leaf layout with dtype=dt.
    let mut leaf = Leaf::new(dt);
    leaf.len = 0;
    if obj.bind(py).is_none() {
        leaf.push_null();
        return Ok(Layout::Leaf(leaf));
    }
    // Reuse encode_scalar for numeric types (string/char handled by push_scalar on build path, but not here).
    match dt {
        DType::String | DType::Char => {
            // Slow but correct: build via python list.
            let out = crate::layout::build_array(py, &pyo3::types::PyList::new_bound(py, [obj.clone_ref(py)]).into_any(), dt)?;
            return Ok(out.layout);
        }
        _ => {
            let (valid, bytes) = Leaf::encode_scalar(py, &obj.bind(py), dt)?;
            leaf.push_value(&bytes)?;
            if !valid {
                leaf.has_nulls = true;
            }
            Ok(Layout::Leaf(leaf))
        }
    }
}

fn stack_elements_as_list(dt: DType, elems: &[Layout]) -> PyResult<Layout> {
    // elems are the axis0 elements of the result (all same depth).
    if elems.is_empty() {
        return Ok(Layout::Leaf(Leaf::new(dt)));
    }
    if matches!(elems[0], Layout::Leaf(_)) {
        // If each element is a scalar leaf, return a 1D leaf of length elems.len().
        // Otherwise, represent as ListOffset of nested layouts.
        let all_scalar = elems.iter().all(|e| matches!(e, Layout::Leaf(l) if l.len == 1));
        if all_scalar {
            let mut out = Leaf::new(dt);
            out.len = elems.len();
            out.validity = Arc::new(bitvec![u8, Lsb0; 1; elems.len()]);
            out.has_nulls = false;
            // Allocate
            out.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(Arc::new(vec![0i8; elems.len()])),
                DType::Int16 => LeafBuffer::I16(Arc::new(vec![0i16; elems.len()])),
                DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; elems.len()])),
                DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; elems.len()])),
                DType::UInt8 => LeafBuffer::U8(Arc::new(vec![0u8; elems.len()])),
                DType::UInt16 => LeafBuffer::U16(Arc::new(vec![0u16; elems.len()])),
                DType::UInt32 => LeafBuffer::U32(Arc::new(vec![0u32; elems.len()])),
                DType::UInt64 => LeafBuffer::U64(Arc::new(vec![0u64; elems.len()])),
                DType::Float16 => LeafBuffer::F16(Arc::new(vec![0u16; elems.len()])),
                DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; elems.len()])),
                DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; elems.len()])),
                DType::Bool => LeafBuffer::Bool(Arc::new(vec![0u8; elems.len()])),
                DType::Char => LeafBuffer::Char(Arc::new(vec![0u32; elems.len()])),
                DType::String => LeafBuffer::String(Arc::new(vec![String::new(); elems.len()])),
            };
            let out_valid = Arc::make_mut(&mut out.validity);
            for (i, e) in elems.iter().enumerate() {
                let l = match e { Layout::Leaf(l) => l, _ => unreachable!() };
                if !l.validity[0] {
                    out_valid.set(i, false);
                    out.has_nulls = true;
                    continue;
                }
                match (&l.buffer, &mut out.buffer) {
                    (LeafBuffer::I8(v), LeafBuffer::I8(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::I16(v), LeafBuffer::I16(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::I32(v), LeafBuffer::I32(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::I64(v), LeafBuffer::I64(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::U8(v), LeafBuffer::U8(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::U16(v), LeafBuffer::U16(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::U32(v), LeafBuffer::U32(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::U64(v), LeafBuffer::U64(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::F16(v), LeafBuffer::F16(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::F32(v), LeafBuffer::F32(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::F64(v), LeafBuffer::F64(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::Bool(v), LeafBuffer::Bool(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::Char(v), LeafBuffer::Char(o)) => Arc::make_mut(o)[i] = v[0],
                    (LeafBuffer::String(v), LeafBuffer::String(o)) => Arc::make_mut(o)[i] = v[0].clone(),
                    _ => return Err(PyValueError::new_err("Internal error: scalar leaf buffer mismatch.")),
                }
            }
            return Ok(Layout::Leaf(out));
        }
    }

    // General case: build ListOffset with concatenated content.
    build_stack_layout(elems)
}

fn build_stack_layout(elems: &[Layout]) -> PyResult<Layout> {
    if elems.is_empty() {
        return Err(PyValueError::new_err("Internal error: cannot stack empty layouts."));
    }
    let mut offsets: Vec<i64> = Vec::with_capacity(elems.len() + 1);
    offsets.push(0);
    let mut acc: i64 = 0;
    for e in elems {
        acc += e.len() as i64;
        offsets.push(acc);
    }
    let content = concat_axis0_layouts(elems)?;
    Ok(Layout::ListOffset(crate::layout::ListOffset { offsets: Arc::new(offsets), content: Box::new(content) }))
}

fn concat_axis0_layouts(layouts: &[Layout]) -> PyResult<Layout> {
    if layouts.is_empty() {
        return Err(PyValueError::new_err("Internal error: cannot concat empty layouts."));
    }
    match &layouts[0] {
        Layout::Leaf(first) => {
            let dt = first.dtype;
            let total: usize = layouts.iter().map(|l| l.len()).sum();
            let mut out = Leaf::new(dt);
            out.len = total;
            out.validity = Arc::new(bitvec![u8, Lsb0; 1; total]);
            out.has_nulls = false;
            out.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(Arc::new(Vec::with_capacity(total))),
                DType::Int16 => LeafBuffer::I16(Arc::new(Vec::with_capacity(total))),
                DType::Int32 => LeafBuffer::I32(Arc::new(Vec::with_capacity(total))),
                DType::Int64 => LeafBuffer::I64(Arc::new(Vec::with_capacity(total))),
                DType::UInt8 => LeafBuffer::U8(Arc::new(Vec::with_capacity(total))),
                DType::UInt16 => LeafBuffer::U16(Arc::new(Vec::with_capacity(total))),
                DType::UInt32 => LeafBuffer::U32(Arc::new(Vec::with_capacity(total))),
                DType::UInt64 => LeafBuffer::U64(Arc::new(Vec::with_capacity(total))),
                DType::Float16 => LeafBuffer::F16(Arc::new(Vec::with_capacity(total))),
                DType::Float32 => LeafBuffer::F32(Arc::new(Vec::with_capacity(total))),
                DType::Float64 => LeafBuffer::F64(Arc::new(Vec::with_capacity(total))),
                DType::Bool => LeafBuffer::Bool(Arc::new(Vec::with_capacity(total))),
                DType::Char => LeafBuffer::Char(Arc::new(Vec::with_capacity(total))),
                DType::String => LeafBuffer::String(Arc::new(Vec::with_capacity(total))),
            };
            let mut pos = 0usize;
            for l in layouts {
                let leaf = match l {
                    Layout::Leaf(x) => x,
                    _ => return Err(PyValueError::new_err("Internal error: concat mixed layout kinds.")),
                };
                if leaf.dtype != dt {
                    return Err(PyValueError::new_err("Internal error: concat leaf dtype mismatch."));
                }
                if leaf.has_nulls {
                    out.has_nulls = true;
                }
                for i in 0..leaf.len {
                    if !leaf.validity[i] {
                        Arc::make_mut(&mut out.validity).set(pos + i, false);
                    }
                }
                match (&leaf.buffer, &mut out.buffer) {
                    (LeafBuffer::I8(v), LeafBuffer::I8(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::I16(v), LeafBuffer::I16(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::I32(v), LeafBuffer::I32(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::I64(v), LeafBuffer::I64(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U8(v), LeafBuffer::U8(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U16(v), LeafBuffer::U16(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U32(v), LeafBuffer::U32(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U64(v), LeafBuffer::U64(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::F16(v), LeafBuffer::F16(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::F32(v), LeafBuffer::F32(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::F64(v), LeafBuffer::F64(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::Bool(v), LeafBuffer::Bool(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::Char(v), LeafBuffer::Char(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::String(v), LeafBuffer::String(o)) => Arc::make_mut(o).extend(v.iter().cloned()),
                    _ => return Err(PyValueError::new_err("Internal error: concat leaf buffer mismatch.")),
                }
                pos += leaf.len;
            }
            Ok(Layout::Leaf(out))
        }
        Layout::ListOffset(_) => {
            let mut all_offsets: Vec<i64> = Vec::new();
            all_offsets.push(0);
            let mut content_segs: Vec<Layout> = Vec::with_capacity(layouts.len());
            let mut acc: i64 = 0;
            for l in layouts {
                let lo = match l {
                    Layout::ListOffset(lo) => lo,
                    _ => return Err(PyValueError::new_err("Internal error: concat mixed layout kinds.")),
                };
                let offs = lo.offsets.as_slice();
                if offs.is_empty() {
                    return Err(PyValueError::new_err("Internal error: invalid offsets."));
                }
                for &o in &offs[1..] {
                    all_offsets.push(acc + o);
                }
                acc += *offs.last().unwrap();
                content_segs.push(lo.content.as_ref().clone());
            }
            let content = concat_axis0_layouts(&content_segs)?;
            Ok(Layout::ListOffset(crate::layout::ListOffset { offsets: Arc::new(all_offsets), content: Box::new(content) }))
        }
        _ => Err(PyValueError::new_err("concat_axis0_layouts: unsupported layout kind.")),
    }
}

fn rect2d_shape(layout: &Layout) -> Option<(usize, usize, &[i64], &Leaf)> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return None,
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return None,
    };
    let nrows = lo.len();
    if nrows == 0 {
        return Some((0, 0, lo.offsets.as_slice(), leaf));
    }
    let first = (lo.offsets[1] - lo.offsets[0]) as usize;
    for i in 0..nrows {
        let len_i = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
        if len_i != first {
            return None;
        }
    }
    if leaf.len != nrows * first {
        return None;
    }
    Some((nrows, first, lo.offsets.as_slice(), leaf))
}

fn reduce_rect2d_fast(layout: &Layout, dt: DType, dim: isize, op: ReduceOp) -> PyResult<Option<GrumpyArray>> {
    let (nrows, ncols, _off, leaf) = match rect2d_shape(layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if leaf.has_nulls {
        return Ok(None);
    }
    if nrows == 0 {
        // Empty outer dimension; for now return empty 1D leaf for dim=1, and empty 1D leaf for dim=0.
        // (Matches shape expectations; exact dtype rules handled below.)
    }
    let dim_u = if dim < 0 { (1isize + dim + 1) as usize } else { dim as usize };
    match dim_u {
        1 => reduce_rect2d_dim1_fast(nrows, ncols, leaf, dt, op),
        0 => reduce_rect2d_dim0_fast(nrows, ncols, leaf, dt, op),
        _ => Ok(None),
    }
}

fn reduce_rect2d_dim1_fast(
    nrows: usize,
    ncols: usize,
    leaf: &Leaf,
    dt: DType,
    op: ReduceOp,
) -> PyResult<Option<GrumpyArray>> {
    let out_dt = match (dt, op) {
        (DType::Int32 | DType::Int64, ReduceOp::Sum) => DType::Int64,
        (DType::Float32 | DType::Float64, ReduceOp::Sum) => DType::Float64,
        (_, ReduceOp::Mean) => DType::Float64,
        _ => dt,
    };
    let mut out = out_leaf_for_2d(nrows, out_dt)?;
    // all-valid because rectangular fastpath requires all-valid input and dim=1 always produces a value per row (even if ncols==0 -> None)
    let out_valid = Arc::make_mut(&mut out.validity);
    if ncols == 0 {
        for i in 0..nrows {
            out_valid.set(i, false);
        }
        out.has_nulls = true;
        return Ok(Some(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) }));
    }
    match (dt, op) {
        (DType::Int32, ReduceOp::Sum) => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let o = match &mut out.buffer { LeafBuffer::I64(o) => Arc::make_mut(o), _ => unreachable!() };
            for i in 0..nrows {
                let base = i * ncols;
                let mut acc: i64 = 0;
                for j in 0..ncols { acc += v[base + j] as i64; }
                o[i] = acc;
            }
        }
        (DType::Int32, ReduceOp::Mean) => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let o = match &mut out.buffer { LeafBuffer::F64(o) => Arc::make_mut(o), _ => unreachable!() };
            let denom = ncols as f64;
            for i in 0..nrows {
                let base = i * ncols;
                let mut acc: f64 = 0.0;
                for j in 0..ncols { acc += v[base + j] as f64; }
                o[i] = acc / denom;
            }
        }
        (DType::Float64, ReduceOp::Mean) | (DType::Float64, ReduceOp::Sum) => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let o = match &mut out.buffer { LeafBuffer::F64(o) => Arc::make_mut(o), _ => unreachable!() };
            let denom = ncols as f64;
            for i in 0..nrows {
                let base = i * ncols;
                let mut acc: f64 = 0.0;
                for j in 0..ncols { acc += v[base + j]; }
                o[i] = if op == ReduceOp::Mean { acc / denom } else { acc };
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) }))
}

fn reduce_rect2d_dim0_fast(
    nrows: usize,
    ncols: usize,
    leaf: &Leaf,
    dt: DType,
    op: ReduceOp,
) -> PyResult<Option<GrumpyArray>> {
    // For rectangular all-valid input, dim=0 always has values for all positions if nrows>0.
    // Output is a 1D leaf of length ncols.
    let out_dt = match (dt, op) {
        (DType::Int32 | DType::Int64, ReduceOp::Sum) => DType::Int64,
        (DType::Float32 | DType::Float64, ReduceOp::Sum) => DType::Float64,
        (_, ReduceOp::Mean) => DType::Float64,
        _ => dt,
    };
    let mut out = out_leaf_for_2d(ncols, out_dt)?;
    let out_valid = Arc::make_mut(&mut out.validity);
    if nrows == 0 {
        for j in 0..ncols { out_valid.set(j, false); }
        out.has_nulls = true;
        return Ok(Some(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) }));
    }
    match (dt, op) {
        (DType::Int32, ReduceOp::Sum) => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let o = match &mut out.buffer { LeafBuffer::I64(o) => Arc::make_mut(o), _ => unreachable!() };
            for j in 0..ncols {
                let mut acc: i64 = 0;
                for i in 0..nrows { acc += v[i * ncols + j] as i64; }
                o[j] = acc;
            }
        }
        (DType::Int32, ReduceOp::Mean) => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let o = match &mut out.buffer { LeafBuffer::F64(o) => Arc::make_mut(o), _ => unreachable!() };
            let denom = nrows as f64;
            for j in 0..ncols {
                let mut acc: f64 = 0.0;
                for i in 0..nrows { acc += v[i * ncols + j] as f64; }
                o[j] = acc / denom;
            }
        }
        _ => return Ok(None),
    }
    // all valid
    Ok(Some(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) }))
}

fn reduce_leaf_to_scalar(py: Python<'_>, layout: &Layout, dt: DType, op: ReduceOp) -> PyResult<PyObject> {
    let leaf = match layout {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf layout.")),
    };
    match (dt, op) {
        (DType::Int32, ReduceOp::Sum) => {
            let mut acc: i64 = 0;
            let mut any = false;
            if let LeafBuffer::I32(v) = &leaf.buffer {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        acc += v[i] as i64;
                        any = true;
                    }
                }
            }
            return Ok(if any { acc.into_py(py) } else { py.None() });
        }
        (DType::Int64, ReduceOp::Sum) => {
            let mut acc: i64 = 0;
            let mut any = false;
            if let LeafBuffer::I64(v) = &leaf.buffer {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        acc = acc.wrapping_add(v[i]);
                        any = true;
                    }
                }
            }
            return Ok(if any { acc.into_py(py) } else { py.None() });
        }
        (DType::Float32, ReduceOp::Sum) | (DType::Float32, ReduceOp::Mean) => {
            let mut acc: f64 = 0.0;
            let mut cnt: i64 = 0;
            if let LeafBuffer::F32(v) = &leaf.buffer {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        acc += v[i] as f64;
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                return Ok(py.None());
            }
            if op == ReduceOp::Mean {
                acc /= cnt as f64;
            }
            return Ok(acc.into_py(py));
        }
        (DType::Float64, ReduceOp::Sum) | (DType::Float64, ReduceOp::Mean) => {
            let mut acc: f64 = 0.0;
            let mut cnt: i64 = 0;
            if let LeafBuffer::F64(v) = &leaf.buffer {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        acc += v[i];
                        cnt += 1;
                    }
                }
            }
            if cnt == 0 {
                return Ok(py.None());
            }
            if op == ReduceOp::Mean {
                acc /= cnt as f64;
            }
            return Ok(acc.into_py(py));
        }
        _ => Err(PyValueError::new_err("Reduction not implemented for this dtype/op yet.")),
    }
}

fn out_leaf_for_2d(n: usize, out_dt: DType) -> PyResult<Leaf> {
    let mut out = Leaf::new(out_dt);
    out.len = n;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out.has_nulls = false;
    out.buffer = match out_dt {
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        _ => return Err(PyValueError::new_err("Unsupported reduction output dtype.")),
    };
    Ok(out)
}

fn reduce_2d_dim1_to_leaf(layout: &Layout, dt: DType, op: ReduceOp) -> PyResult<GrumpyArray> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Expected list layout.")),
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf at depth 2.")),
    };
    let nrows = lo.len();

    let out_dt = match (dt, op) {
        (DType::Int32 | DType::Int64, ReduceOp::Sum) => DType::Int64,
        (DType::Float32 | DType::Float64, ReduceOp::Sum) => DType::Float64,
        (DType::Float32 | DType::Float64, ReduceOp::Mean) => DType::Float64,
        (_, ReduceOp::Mean) => DType::Float64,
        _ => dt,
    };
    let mut out = out_leaf_for_2d(nrows, out_dt)?;
    let out_valid = Arc::make_mut(&mut out.validity);

    for i in 0..nrows {
        let start = lo.offsets[i] as usize;
        let end = lo.offsets[i + 1] as usize;
        let mut any = false;
        match (dt, op) {
            (DType::Int32, ReduceOp::Sum) => {
                let mut acc: i64 = 0;
                if let LeafBuffer::I32(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            acc += v[j] as i64;
                            any = true;
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    if let LeafBuffer::I64(o) = &mut out.buffer { Arc::make_mut(o)[i] = acc; }
                }
            }
            (DType::Int32, ReduceOp::Min) | (DType::Int32, ReduceOp::Max) | (DType::Int32, ReduceOp::Ptp) => {
                let mut mn: i32 = 0;
                let mut mx: i32 = 0;
                if let LeafBuffer::I32(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            let x = v[j];
                            if !any { mn = x; mx = x; any = true; } else {
                                if x < mn { mn = x; }
                                if x > mx { mx = x; }
                            }
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    let val = match op { ReduceOp::Min => mn, ReduceOp::Max => mx, ReduceOp::Ptp => mx.wrapping_sub(mn), _ => 0 };
                    if let LeafBuffer::I32(o) = &mut out.buffer { Arc::make_mut(o)[i] = val; }
                }
            }
            (DType::Int64, ReduceOp::Sum) => {
                let mut acc: i64 = 0;
                if let LeafBuffer::I64(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            acc = acc.wrapping_add(v[j]);
                            any = true;
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    if let LeafBuffer::I64(o) = &mut out.buffer { Arc::make_mut(o)[i] = acc; }
                }
            }
            (DType::Int64, ReduceOp::Min) | (DType::Int64, ReduceOp::Max) | (DType::Int64, ReduceOp::Ptp) => {
                let mut mn: i64 = 0;
                let mut mx: i64 = 0;
                if let LeafBuffer::I64(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            let x = v[j];
                            if !any { mn = x; mx = x; any = true; } else {
                                if x < mn { mn = x; }
                                if x > mx { mx = x; }
                            }
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    let val = match op { ReduceOp::Min => mn, ReduceOp::Max => mx, ReduceOp::Ptp => mx.wrapping_sub(mn), _ => 0 };
                    if let LeafBuffer::I64(o) = &mut out.buffer { Arc::make_mut(o)[i] = val; }
                }
            }
            (DType::Int32 | DType::Int64, ReduceOp::Mean) => {
                let mut acc: f64 = 0.0;
                let mut cnt: i64 = 0;
                match (&leaf.buffer, dt) {
                    (LeafBuffer::I32(v), DType::Int32) => {
                        for j in start..end {
                            if leaf.validity[j] {
                                acc += v[j] as f64;
                                cnt += 1;
                                any = true;
                            }
                        }
                    }
                    (LeafBuffer::I64(v), DType::Int64) => {
                        for j in start..end {
                            if leaf.validity[j] {
                                acc += v[j] as f64;
                                cnt += 1;
                                any = true;
                            }
                        }
                    }
                    _ => {}
                }
                if !any { out_valid.set(i, false); } else {
                    acc /= cnt as f64;
                    if let LeafBuffer::F64(o) = &mut out.buffer { Arc::make_mut(o)[i] = acc; }
                }
            }
            (DType::Float32, ReduceOp::Sum) | (DType::Float32, ReduceOp::Mean) => {
                let mut acc: f64 = 0.0;
                let mut cnt: i64 = 0;
                if let LeafBuffer::F32(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            acc += v[j] as f64;
                            cnt += 1;
                            any = true;
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    if op == ReduceOp::Mean { acc /= cnt as f64; }
                    if let LeafBuffer::F64(o) = &mut out.buffer { Arc::make_mut(o)[i] = acc; }
                }
            }
            (DType::Float32, ReduceOp::Min) | (DType::Float32, ReduceOp::Max) | (DType::Float32, ReduceOp::Ptp) => {
                let mut mn: f32 = 0.0;
                let mut mx: f32 = 0.0;
                let mut seen_nan = false;
                if let LeafBuffer::F32(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            let x = v[j];
                            if x.is_nan() { seen_nan = true; }
                            if !any { mn = x; mx = x; any = true; } else {
                                if x < mn { mn = x; }
                                if x > mx { mx = x; }
                            }
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    let val = if seen_nan { f32::NAN } else {
                        match op { ReduceOp::Min => mn, ReduceOp::Max => mx, ReduceOp::Ptp => mx - mn, _ => 0.0 }
                    };
                    if let LeafBuffer::F32(o) = &mut out.buffer { Arc::make_mut(o)[i] = val; }
                }
            }
            (DType::Float64, ReduceOp::Sum) | (DType::Float64, ReduceOp::Mean) => {
                let mut acc: f64 = 0.0;
                let mut cnt: i64 = 0;
                if let LeafBuffer::F64(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            acc += v[j];
                            cnt += 1;
                            any = true;
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    if op == ReduceOp::Mean { acc /= cnt as f64; }
                    if let LeafBuffer::F64(o) = &mut out.buffer { Arc::make_mut(o)[i] = acc; }
                }
            }
            (DType::Float64, ReduceOp::Min) | (DType::Float64, ReduceOp::Max) | (DType::Float64, ReduceOp::Ptp) => {
                let mut mn: f64 = 0.0;
                let mut mx: f64 = 0.0;
                let mut seen_nan = false;
                if let LeafBuffer::F64(v) = &leaf.buffer {
                    for j in start..end {
                        if leaf.validity[j] {
                            let x = v[j];
                            if x.is_nan() { seen_nan = true; }
                            if !any { mn = x; mx = x; any = true; } else {
                                if x < mn { mn = x; }
                                if x > mx { mx = x; }
                            }
                        }
                    }
                }
                if !any { out_valid.set(i, false); } else {
                    let val = if seen_nan { f64::NAN } else {
                        match op { ReduceOp::Min => mn, ReduceOp::Max => mx, ReduceOp::Ptp => mx - mn, _ => 0.0 }
                    };
                    if let LeafBuffer::F64(o) = &mut out.buffer { Arc::make_mut(o)[i] = val; }
                }
            }
            _ => return Err(PyValueError::new_err("Reduction not implemented for this dtype/op.")),
        }
    }

    Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) })
}

fn reduce_2d_dim0_to_leaf(layout: &Layout, dt: DType, op: ReduceOp) -> PyResult<GrumpyArray> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Expected list layout.")),
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf at depth 2.")),
    };
    let nrows = lo.len();
    let mut maxlen: usize = 0;
    for i in 0..nrows {
        let len = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
        if len > maxlen { maxlen = len; }
    }

    let out_dt = match (dt, op) {
        (DType::Int32 | DType::Int64, ReduceOp::Sum) => DType::Int64,
        (DType::Float32 | DType::Float64, ReduceOp::Sum) => DType::Float64,
        (DType::Float32 | DType::Float64, ReduceOp::Mean) => DType::Float64,
        (_, ReduceOp::Mean) => DType::Float64,
        _ => dt,
    };
    let mut out = out_leaf_for_2d(maxlen, out_dt)?;
    let out_valid = Arc::make_mut(&mut out.validity);

    match (dt, op) {
        (DType::Int32, ReduceOp::Sum) => {
            let mut sums = vec![0i64; maxlen];
            let mut counts = vec![0i64; maxlen];
            if let LeafBuffer::I32(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            sums[j] += v[ix] as i64;
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::I64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = sums[j];
                    }
                }
            }
        }
        (DType::Int64, ReduceOp::Sum) => {
            let mut sums = vec![0i64; maxlen];
            let mut counts = vec![0i64; maxlen];
            if let LeafBuffer::I64(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            sums[j] = sums[j].wrapping_add(v[ix]);
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::I64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = sums[j];
                    }
                }
            }
        }
        (DType::Int32 | DType::Int64, ReduceOp::Mean) => {
            let mut sums = vec![0f64; maxlen];
            let mut counts = vec![0i64; maxlen];
            match (&leaf.buffer, dt) {
                (LeafBuffer::I32(v), DType::Int32) => {
                    for i in 0..nrows {
                        let start = lo.offsets[i] as usize;
                        let end = lo.offsets[i + 1] as usize;
                        let len = end - start;
                        for j in 0..len {
                            let ix = start + j;
                            if leaf.validity[ix] {
                                sums[j] += v[ix] as f64;
                                counts[j] += 1;
                            }
                        }
                    }
                }
                (LeafBuffer::I64(v), DType::Int64) => {
                    for i in 0..nrows {
                        let start = lo.offsets[i] as usize;
                        let end = lo.offsets[i + 1] as usize;
                        let len = end - start;
                        for j in 0..len {
                            let ix = start + j;
                            if leaf.validity[ix] {
                                sums[j] += v[ix] as f64;
                                counts[j] += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
            if let LeafBuffer::F64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = sums[j] / (counts[j] as f64);
                    }
                }
            }
        }
        (DType::Int32, ReduceOp::Min) | (DType::Int32, ReduceOp::Max) | (DType::Int32, ReduceOp::Ptp) => {
            let mut counts = vec![0i64; maxlen];
            let mut mn = vec![0i32; maxlen];
            let mut mx = vec![0i32; maxlen];
            if let LeafBuffer::I32(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            let x = v[ix];
                            if counts[j] == 0 { mn[j] = x; mx[j] = x; } else {
                                if x < mn[j] { mn[j] = x; }
                                if x > mx[j] { mx[j] = x; }
                            }
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::I32(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = match op {
                            ReduceOp::Min => mn[j],
                            ReduceOp::Max => mx[j],
                            ReduceOp::Ptp => mx[j].wrapping_sub(mn[j]),
                            _ => 0,
                        };
                    }
                }
            }
        }
        (DType::Int64, ReduceOp::Min) | (DType::Int64, ReduceOp::Max) | (DType::Int64, ReduceOp::Ptp) => {
            let mut counts = vec![0i64; maxlen];
            let mut mn = vec![0i64; maxlen];
            let mut mx = vec![0i64; maxlen];
            if let LeafBuffer::I64(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            let x = v[ix];
                            if counts[j] == 0 { mn[j] = x; mx[j] = x; } else {
                                if x < mn[j] { mn[j] = x; }
                                if x > mx[j] { mx[j] = x; }
                            }
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::I64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = match op {
                            ReduceOp::Min => mn[j],
                            ReduceOp::Max => mx[j],
                            ReduceOp::Ptp => mx[j].wrapping_sub(mn[j]),
                            _ => 0,
                        };
                    }
                }
            }
        }
        (DType::Float32, ReduceOp::Sum) | (DType::Float32, ReduceOp::Mean) => {
            let mut sums = vec![0f64; maxlen];
            let mut counts = vec![0i64; maxlen];
            if let LeafBuffer::F32(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            sums[j] += v[ix] as f64;
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::F64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = if op == ReduceOp::Mean {
                            sums[j] / (counts[j] as f64)
                        } else {
                            sums[j]
                        };
                    }
                }
            }
        }
        (DType::Float32, ReduceOp::Min) | (DType::Float32, ReduceOp::Max) | (DType::Float32, ReduceOp::Ptp) => {
            let mut counts = vec![0i64; maxlen];
            let mut mn = vec![0f32; maxlen];
            let mut mx = vec![0f32; maxlen];
            let mut seen_nan = vec![false; maxlen];
            if let LeafBuffer::F32(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            let x = v[ix];
                            if x.is_nan() { seen_nan[j] = true; }
                            if counts[j] == 0 { mn[j] = x; mx[j] = x; } else {
                                if x < mn[j] { mn[j] = x; }
                                if x > mx[j] { mx[j] = x; }
                            }
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::F32(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = if seen_nan[j] {
                            f32::NAN
                        } else {
                            match op {
                                ReduceOp::Min => mn[j],
                                ReduceOp::Max => mx[j],
                                ReduceOp::Ptp => mx[j] - mn[j],
                                _ => 0.0,
                            }
                        };
                    }
                }
            }
        }
        (DType::Float64, ReduceOp::Sum) | (DType::Float64, ReduceOp::Mean) => {
            let mut sums = vec![0f64; maxlen];
            let mut counts = vec![0i64; maxlen];
            if let LeafBuffer::F64(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            sums[j] += v[ix];
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::F64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = if op == ReduceOp::Mean {
                            sums[j] / (counts[j] as f64)
                        } else {
                            sums[j]
                        };
                    }
                }
            }
        }
        (DType::Float64, ReduceOp::Min) | (DType::Float64, ReduceOp::Max) | (DType::Float64, ReduceOp::Ptp) => {
            let mut counts = vec![0i64; maxlen];
            let mut mn = vec![0f64; maxlen];
            let mut mx = vec![0f64; maxlen];
            let mut seen_nan = vec![false; maxlen];
            if let LeafBuffer::F64(v) = &leaf.buffer {
                for i in 0..nrows {
                    let start = lo.offsets[i] as usize;
                    let end = lo.offsets[i + 1] as usize;
                    let len = end - start;
                    for j in 0..len {
                        let ix = start + j;
                        if leaf.validity[ix] {
                            let x = v[ix];
                            if x.is_nan() { seen_nan[j] = true; }
                            if counts[j] == 0 { mn[j] = x; mx[j] = x; } else {
                                if x < mn[j] { mn[j] = x; }
                                if x > mx[j] { mx[j] = x; }
                            }
                            counts[j] += 1;
                        }
                    }
                }
            }
            if let LeafBuffer::F64(o) = &mut out.buffer {
                let o = Arc::make_mut(o);
                for j in 0..maxlen {
                    if counts[j] != nrows as i64 {
                        out_valid.set(j, false);
                        out.has_nulls = true;
                    } else {
                        o[j] = if seen_nan[j] {
                            f64::NAN
                        } else {
                            match op {
                                ReduceOp::Min => mn[j],
                                ReduceOp::Max => mx[j],
                                ReduceOp::Ptp => mx[j] - mn[j],
                                _ => 0.0,
                            }
                        };
                    }
                }
            }
        }
        _ => return Err(PyValueError::new_err("Reduction not implemented for this dtype/op.")),
    }

    Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) })
}


