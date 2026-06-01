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
use crate::error::{
    dim_out_of_range, internal, reduction_empty, reduction_scalar_unsupported, union_op_dim_unsupported,
    unsupported,
};
use crate::layout::{layout_ndim, offsetview_to_listoffset, drop_axis0_select_element, GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use crate::layout_ops::{concat_axis0_layouts, map_last_axis, map_union_axis0, LastAxisLeafMode};
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

    if layout.has_union() {
        let ndim = layout_ndim(layout)?;
        let axis = normalize_axis(dim, ndim)?;
        if axis == 0 {
            return reduce_union_axis0_nogil(arr, op);
        }
        if axis == ndim - 1 {
            let out_dt = reduce_out_dtype(arr.dtype, op)?;
            let out_layout = if layout.has_union() && matches!(layout, Layout::UnionScalarList(_)) {
                map_union_axis0(layout, out_dt, |sub| {
                    reduce_element_last_axis(&sub, arr.dtype, out_dt, op)
                })?
            } else {
                reduce_last_layout(layout, arr.dtype, out_dt, op)?
            };
            return Ok(GrumpyArray {
                dtype: out_dt,
                layout: out_layout,
            });
        }
        return Err(union_op_dim_unsupported(
            "reduce",
            dim,
            "dim=0 and the innermost axis",
        ));
    }

    let depth = crate::layout::list_chain_depth(layout)
        .ok_or_else(|| {
            unsupported(
                "reduce",
                "currently only supports pure list-chain arrays.",
                "use union-aware reductions (dim=0 or innermost) for UnionScalarList arrays.",
            )
        })?;
    let axis = normalize_axis(dim, depth)?;

    // Scalar outputs are not supported in Rust scheduling (returning PyObject would require the GIL).
    if depth == 0 {
        return Err(reduction_scalar_unsupported("reduce"));
    }

    // 2D list->leaf fast paths first.
    if depth == 1 {
        if let Some(out) = reduce_rect2d_fast(layout, arr.dtype, dim, op)? {
            return Ok(out);
        }
        return match axis {
            0 => Ok(reduce_2d_dim0_to_leaf(layout, arr.dtype, op)?),
            1 => Ok(reduce_2d_dim1_to_leaf(layout, arr.dtype, op)?),
            _ => Err(dim_out_of_range(dim, 2)),
        };
    }

    let out_dt = reduce_out_dtype(arr.dtype, op)?;
    let out_layout = reduce_list_chain_to_layout_nogil(layout, arr.dtype, out_dt, depth, axis, op)?;
    Ok(GrumpyArray { dtype: out_dt, layout: out_layout })
}

pub fn reduce(py: Python<'_>, arr: &GrumpyArray, dim: Option<isize>, op: ReduceOp) -> PyResult<ReduceOutput> {
    let norm_layout;
    let layout: &Layout = match &arr.layout {
        Layout::OffsetView(v) => {
            norm_layout = Layout::ListOffset(offsetview_to_listoffset(v)?);
            &norm_layout
        }
        _ => &arr.layout,
    };

    if layout.has_union() {
        return reduce_union(py, arr, dim, op);
    }

    let depth = crate::layout::list_chain_depth(layout)
        .ok_or_else(|| {
            unsupported(
                "reduce",
                "currently only supports pure list-chain arrays.",
                "use union-aware reductions (dim=0 or innermost) for UnionScalarList arrays.",
            )
        })?;

    // Sum/mean over all leaves (no dim): flatten reduction for 2D list->leaf.
    if dim.is_none() {
        if depth == 1 {
            if let Some(scalar) = reduce_listoffset2d_all_fast(py, layout, arr.dtype, op)? {
                return Ok(ReduceOutput::Scalar(scalar));
            }
        }
        if depth == 0 && op == ReduceOp::Sum {
            return Ok(ReduceOutput::Scalar(reduce_leaf_to_scalar(py, layout, arr.dtype, op)?));
        }
        return Err(PyValueError::new_err(
            "Reduction over all axes (no dim) is only supported for sum on list->leaf arrays.",
        ));
    }
    let dim = dim.unwrap();

    let axis = normalize_axis(dim, depth)?;

    if depth == 0 {
        if axis != 0 {
            return Err(dim_out_of_range(dim, 1));
        }
        return Ok(ReduceOutput::Scalar(reduce_leaf_to_scalar(py, layout, arr.dtype, op)?));
    }

    if depth == 1 {
        if let Some(out) = reduce_rect2d_fast(layout, arr.dtype, dim, op)? {
            return Ok(ReduceOutput::Array(out));
        }
        if let Some(out) = reduce_ragged2d_dim1_fast(layout, arr.dtype, dim, op)? {
            return Ok(ReduceOutput::Array(out));
        }
        return match axis {
            0 => Ok(ReduceOutput::Array(reduce_2d_dim0_to_leaf(layout, arr.dtype, op)?)),
            1 => Ok(ReduceOutput::Array(reduce_2d_dim1_to_leaf(layout, arr.dtype, op)?)),
            _ => Err(dim_out_of_range(dim, 2)),
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
            return Err(internal(
                "reduce",
                "unsupported scalar-to-leaf conversion in no-GIL reduction",
            ))
        }
    };
    Ok(Layout::Leaf(leaf))
}

fn reduce_leaf_to_scalar_value(layout: &Layout, dt: DType, op: ReduceOp) -> PyResult<Option<ScalarValue>> {
    let leaf = match layout {
        Layout::Leaf(l) => l,
        _ => return Err(internal("reduce", "expected leaf layout")),
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
        _ => return Err(internal("reduce", "expected ListOffset at top")),
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
                return Err(dim_out_of_range(axis as isize, depth));
            }
            let s = reduce_leaf_to_scalar_value(&el, in_dt, op)?;
            scalar_to_leaf_layout(out_dt, s)?
        } else if el_depth == 1 {
            let arr = GrumpyArray { dtype: in_dt, layout: el };
            match axis - 1 {
                0 => reduce_2d_dim0_to_leaf(&arr.layout, in_dt, op)?.layout,
                1 => reduce_2d_dim1_to_leaf(&arr.layout, in_dt, op)?.layout,
                _ => return Err(dim_out_of_range(axis as isize, depth)),
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

fn reduce_op_name(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::Mean => "mean",
        ReduceOp::Ptp => "ptp",
    }
}

fn normalize_axis(dim: isize, depth: usize) -> PyResult<usize> {
    let ndims = depth as isize + 1;
    let mut d = dim;
    if d < 0 {
        d += ndims;
    }
    if d < 0 || d >= ndims {
        return Err(dim_out_of_range(dim, depth + 1));
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
            DType::Char | DType::String => Err(unsupported(
                "reduce",
                "sum/mean are only supported for numeric dtypes",
                "cast to a numeric dtype before reducing.",
            )),
        },
        ReduceOp::Min | ReduceOp::Max | ReduceOp::Ptp => match in_dt {
            DType::Char | DType::String => Err(unsupported(
                "reduce",
                "min/max/ptp are only supported for numeric dtypes",
                "cast to a numeric dtype before reducing.",
            )),
            _ => Ok(in_dt),
        },
    }
}

fn reduce_list_chain_to_layout(
    _py: Python<'_>,
    layout: &Layout,
    in_dt: DType,
    out_dt: DType,
    depth: usize,
    axis: usize,
    op: ReduceOp,
) -> PyResult<Layout> {
    // Deep reductions share the no-GIL engine (numeric leaf stacking via ScalarValue).
    reduce_list_chain_to_layout_nogil(layout, in_dt, out_dt, depth, axis, op)
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
                    _ => return Err(internal("reduce", "scalar leaf buffer mismatch")),
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
        return Err(internal("reduce", "cannot stack empty layouts"));
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

fn listoffset2d_leaf_view(layout: &Layout) -> Option<(&[i64], &Leaf)> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return None,
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return None,
    };
    Some((lo.offsets.as_slice(), leaf))
}

fn reduce_listoffset2d_all_fast(
    py: Python<'_>,
    layout: &Layout,
    dt: DType,
    op: ReduceOp,
) -> PyResult<Option<PyObject>> {
    if op != ReduceOp::Sum {
        return Ok(None);
    }
    let (_off, leaf) = match listoffset2d_leaf_view(layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if leaf.has_nulls || dt != DType::Int32 {
        return Ok(None);
    }
    let v = match &leaf.buffer {
        LeafBuffer::I32(buf) => buf.as_slice(),
        _ => return Ok(None),
    };
    let sum = crate::kernels::sum_i32_to_i64(v);
    Ok(Some(sum.to_object(py)))
}

fn reduce_ragged2d_dim1_fast(
    layout: &Layout,
    dt: DType,
    dim: isize,
    op: ReduceOp,
) -> PyResult<Option<GrumpyArray>> {
    let dim_u = if dim < 0 { (1isize + dim + 1) as usize } else { dim as usize };
    if dim_u != 1 || op != ReduceOp::Sum || dt != DType::Int32 {
        return Ok(None);
    }
    let (offsets, leaf) = match listoffset2d_leaf_view(layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if leaf.has_nulls {
        return Ok(None);
    }
    let v = match &leaf.buffer {
        LeafBuffer::I32(buf) => buf.as_slice(),
        _ => return Ok(None),
    };
    let nrows = offsets.len().saturating_sub(1);
    let mut out = out_leaf_for_2d(nrows, DType::Int64)?;
    let o = match &mut out.buffer {
        LeafBuffer::I64(x) => Arc::make_mut(x),
        _ => unreachable!(),
    };
    crate::kernels::sum_i32_row_sums_to_i64(v, offsets, o);
    Ok(Some(GrumpyArray {
        dtype: DType::Int64,
        layout: Layout::Leaf(out),
    }))
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

fn reduce_last_layout(
    layout: &Layout,
    in_dt: DType,
    _out_dt: DType,
    op: ReduceOp,
) -> PyResult<Layout> {
    map_last_axis(
        layout,
        LastAxisLeafMode::PromoteShortLeaf,
        &|lo, leaf| {
            let tmp = Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(Layout::Leaf(leaf.clone())),
            });
            Ok(reduce_2d_dim1_to_leaf(&tmp, in_dt, op)?.layout)
        },
    )
}

fn reduce_union_last_axis(_py: Python<'_>, arr: &GrumpyArray, op: ReduceOp) -> PyResult<ReduceOutput> {
    let out_dt = reduce_out_dtype(arr.dtype, op)?;
    let layout = map_union_axis0(&arr.layout, out_dt, |sub| {
        reduce_element_last_axis(&sub, arr.dtype, out_dt, op)
    })?;
    Ok(ReduceOutput::Array(GrumpyArray {
        dtype: out_dt,
        layout,
    }))
}

fn reduce_element_last_axis(
    sub: &Layout,
    in_dt: DType,
    out_dt: DType,
    op: ReduceOp,
) -> PyResult<Layout> {
    if let Layout::Leaf(l) = sub {
        if l.len <= 1 {
            return Ok(sub.clone());
        }
        let val = reduce_leaf_to_scalar_value(sub, in_dt, op)?;
        return scalar_to_leaf_layout(out_dt, val);
    }
    if crate::layout::list_chain_depth(sub) == Some(1) {
        return Ok(reduce_2d_dim1_to_leaf(sub, in_dt, op)?.layout);
    }
    reduce_last_layout(sub, in_dt, out_dt, op)
}

fn reduce_union_axis0_nogil(arr: &GrumpyArray, op: ReduceOp) -> PyResult<GrumpyArray> {
    let out_dt = reduce_out_dtype(arr.dtype, op)?;
    let n = arr.len();
    let mut tags = Vec::with_capacity(n);
    let mut index = Vec::with_capacity(n);
    let mut scalars = Leaf::new(out_dt);
    scalars.len = n;
    scalars.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    scalars.has_nulls = false;
    scalars.buffer = LeafBuffer::new(out_dt);

    for i in 0..n {
        let sub = drop_axis0_select_element(&arr.layout, i)?;
        let val = fold_layout_values(&sub, arr.dtype, op)?;
        push_scalar_to_leaf(&mut scalars, out_dt, val)?;
        tags.push(0);
        index.push(i as i64);
    }

    let lists = ListOffset {
        offsets: Arc::new(vec![0i64]),
        content: Box::new(Layout::Leaf(Leaf::new(out_dt))),
    };
    Ok(GrumpyArray {
        dtype: out_dt,
        layout: Layout::UnionScalarList(UnionScalarList {
            tags,
            index,
            scalars,
            lists,
        }),
    })
}

fn reduce_union(py: Python<'_>, arr: &GrumpyArray, dim: Option<isize>, op: ReduceOp) -> PyResult<ReduceOutput> {
    match dim {
        None => match op {
            ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Min | ReduceOp::Max | ReduceOp::Ptp => {
                let val = fold_layout_values(&arr.layout, arr.dtype, op)?;
                Ok(ReduceOutput::Scalar(wrap_reduce_scalar(py, arr.dtype, val)?))
            }
        },
        Some(d) => {
            let ndim = crate::layout::layout_ndim(&arr.layout)?;
            let mut axis = d;
            if axis < 0 {
                axis += ndim as isize;
            }
            if axis < 0 || axis as usize >= ndim {
                return Err(dim_out_of_range(d, ndim));
            }
            if axis == 0 {
                return reduce_union_axis0(py, arr, op);
            }
            if axis as usize == ndim - 1 {
                return reduce_union_last_axis(py, arr, op);
            }
            return Err(union_op_dim_unsupported(
                "reduce",
                d,
                "dim=0 and the innermost axis",
            ));
        }
    }
}

fn reduce_union_axis0(_py: Python<'_>, arr: &GrumpyArray, op: ReduceOp) -> PyResult<ReduceOutput> {
    Ok(ReduceOutput::Array(reduce_union_axis0_nogil(arr, op)?))
}

fn fold_layout_values(layout: &Layout, dt: DType, op: ReduceOp) -> PyResult<f64> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut minv = f64::INFINITY;
    let mut maxv = f64::NEG_INFINITY;
    walk_layout_fold(layout, dt, &mut |v| {
        sum += v;
        count += 1;
        if v < minv {
            minv = v;
        }
        if v > maxv {
            maxv = v;
        }
        Ok(())
    })?;
    if count == 0 {
        return Err(reduction_empty(reduce_op_name(op)));
    }
    Ok(match op {
        ReduceOp::Sum => sum,
        ReduceOp::Mean => sum / count as f64,
        ReduceOp::Min => minv,
        ReduceOp::Max => maxv,
        ReduceOp::Ptp => maxv - minv,
    })
}

fn walk_layout_fold(
    layout: &Layout,
    dt: DType,
    f: &mut dyn FnMut(f64) -> PyResult<()>,
) -> PyResult<()> {
    match layout {
        Layout::Leaf(l) => {
            for i in 0..l.len {
                if l.validity[i] {
                    f(read_leaf_as_f64(dt, &l.buffer, i)?)?;
                }
            }
        }
        Layout::ListOffset(lo) => {
            for i in 0..lo.len() {
                let s = lo.offsets[i] as usize;
                let e = lo.offsets[i + 1] as usize;
                walk_layout_fold(
                    &crate::layout::take_range(lo.content.as_ref(), s, e)?,
                    dt,
                    f,
                )?;
            }
        }
        Layout::OffsetView(v) => {
            for i in 0..v.len() {
                walk_layout_fold(&drop_axis0_select_element(layout, i)?, dt, f)?;
            }
        }
        Layout::Indexed(ix) => {
            for i in 0..ix.len() {
                walk_layout_fold(&drop_axis0_select_element(layout, i)?, dt, f)?;
            }
        }
        Layout::UnionScalarList(u) => {
            for i in 0..u.len() {
                walk_layout_fold(&drop_axis0_select_element(layout, i)?, dt, f)?;
            }
        }
    }
    Ok(())
}

fn read_leaf_as_f64(dt: DType, buf: &LeafBuffer, i: usize) -> PyResult<f64> {
    Ok(match (dt, buf) {
        (DType::Int32, LeafBuffer::I32(v)) => v[i] as f64,
        (DType::Int64, LeafBuffer::I64(v)) => v[i] as f64,
        (DType::Float32, LeafBuffer::F32(v)) => v[i] as f64,
        (DType::Float64, LeafBuffer::F64(v)) => v[i],
        _ => {
            return Err(PyValueError::new_err(
                "Union sum currently supports int32/int64/float32/float64.",
            ))
        }
    })
}

fn push_scalar_to_leaf(leaf: &mut Leaf, dt: DType, val: f64) -> PyResult<()> {
    match (&mut leaf.buffer, dt) {
        (LeafBuffer::I32(v), DType::Int32) => Arc::make_mut(v).push(val as i32),
        (LeafBuffer::I64(v), DType::Int64) => Arc::make_mut(v).push(val as i64),
        (LeafBuffer::F32(v), DType::Float32) => Arc::make_mut(v).push(val as f32),
        (LeafBuffer::F64(v), DType::Float64) => Arc::make_mut(v).push(val),
        _ => {
            return Err(PyValueError::new_err(
                "Union sum currently supports int32/int64/float32/float64.",
            ))
        }
    }
    Ok(())
}

fn wrap_reduce_scalar(py: Python<'_>, dt: DType, val: f64) -> PyResult<PyObject> {
    Ok(match dt {
        DType::Int32 => (val as i32).into_py(py),
        DType::Int64 => (val as i64).into_py(py),
        DType::Float32 => (val as f32).into_py(py),
        DType::Float64 => val.into_py(py),
        _ => {
            return Err(PyValueError::new_err(
                "Union sum currently supports int32/int64/float32/float64.",
            ))
        }
    })
}

