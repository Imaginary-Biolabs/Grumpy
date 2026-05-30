//! Elementwise binary operations (add/sub/mul/div/mod) for `GrumpyArray`.
//!
//! This module is performance-critical. The general pattern is:
//! - Prefer **views over copies** (`OffsetView`, `Indexed`) whenever possible.
//! - Prefer **typed tight loops** over recursion for hot cases (rectangular 2D and ragged 2D list->leaf).
//! - For compiled pipelines, prefer **in-place scalar ops** via copy-on-write (`Arc::make_mut`) to avoid intermediates.
//!
//! Adding a new elementwise op:
//! - **1) Add an enum variant** to `BinOp` and implement dtype rules in `elementwise_out_dtype`.
//! - **2) Fast paths** (in order):
//!   - `elementwise_rect2d_fast` (rectangular 2D list->leaf, all-valid) with one tight loop.
//!   - `elementwise_ragged2d_fast` (ragged 2D list->leaf) with per-row broadcast handling.
//! - **3) Generic engine**:
//!   - `elementwise_layout` for same-structure (including unions).
//!   - `elementwise_layout_broadcast` for broadcasting on pure list-chains.
//! - **4) Compiled pipeline support**:
//!   - If scalar form is common (e.g. `x + 1`), extend `elementwise_scalar_inplace`.
//! - **5) Bench + test**:
//!   - Add kernel-only timings (avoid allocating outputs in the timed region).
//!   - Add parity tests including `OffsetView` batches and null propagation.

use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyResult;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,       // numpy-style remainder (Python %)
    Remainder, // alias of Mod for now
}

/// Apply a scalar binary op to an array **in-place** using copy-on-write.
///
/// This is intended for compiled pipelines to avoid allocating intermediate GrumpyArrays.
/// It is range-aware for outer-axis views (`OffsetView`) so we don't accidentally mutate
/// elements outside the visible slice.
///
/// Returns:
/// - Ok(true) if applied in-place.
/// - Ok(false) if not supported in-place (caller should fall back to allocating path).
pub fn elementwise_scalar_inplace(
    a: &mut GrumpyArray,
    op: BinOp,
    value: f64,
    is_int: bool,
) -> PyResult<bool> {
    // If operation would change dtype (e.g. int true_divide => float64), we cannot do this in-place.
    if op == BinOp::Div {
        match a.dtype {
            DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64 => return Ok(false),
            _ => {}
        }
    }
    // Mod by 0 should raise (match elementwise behavior).
    if (op == BinOp::Mod || op == BinOp::Remainder) && value == 0.0 {
        return Err(PyValueError::new_err("Modulo by zero."));
    }

    fn apply_leaf_range(
        leaf: &mut Leaf,
        dt: DType,
        op: BinOp,
        value: f64,
        is_int: bool,
        start: usize,
        end: usize,
    ) -> PyResult<bool> {
        // For nulls: output validity matches input validity (scalar is all-valid).
        let has_nulls = leaf.has_nulls;
        match (&mut leaf.buffer, dt) {
            (LeafBuffer::I32(v), DType::Int32) => {
                if !is_int {
                    return Err(PyValueError::new_err(
                        "Scalar value must be an int for dtype=int32.",
                    ));
                }
                let s = value as i32;
                if (op == BinOp::Mod || op == BinOp::Remainder) && s == 0 {
                    return Err(PyValueError::new_err("Modulo by zero."));
                }
                let vv = Arc::make_mut(v);
                if has_nulls {
                    for i in start..end {
                        if leaf.validity[i] {
                            vv[i] = match op {
                                BinOp::Add => vv[i].wrapping_add(s),
                                BinOp::Sub => vv[i].wrapping_sub(s),
                                BinOp::Mul => vv[i].wrapping_mul(s),
                                BinOp::Div => unreachable!("int div handled earlier"),
                                BinOp::Mod | BinOp::Remainder => vv[i].wrapping_rem(s),
                            };
                        }
                    }
                } else {
                    for i in start..end {
                        vv[i] = match op {
                            BinOp::Add => vv[i].wrapping_add(s),
                            BinOp::Sub => vv[i].wrapping_sub(s),
                            BinOp::Mul => vv[i].wrapping_mul(s),
                            BinOp::Div => unreachable!("int div handled earlier"),
                            BinOp::Mod | BinOp::Remainder => vv[i].wrapping_rem(s),
                        };
                    }
                }
            }
            (LeafBuffer::I64(v), DType::Int64) => {
                if !is_int {
                    return Err(PyValueError::new_err(
                        "Scalar value must be an int for dtype=int64.",
                    ));
                }
                let s = value as i64;
                if (op == BinOp::Mod || op == BinOp::Remainder) && s == 0 {
                    return Err(PyValueError::new_err("Modulo by zero."));
                }
                let vv = Arc::make_mut(v);
                if has_nulls {
                    for i in start..end {
                        if leaf.validity[i] {
                            vv[i] = match op {
                                BinOp::Add => vv[i].wrapping_add(s),
                                BinOp::Sub => vv[i].wrapping_sub(s),
                                BinOp::Mul => vv[i].wrapping_mul(s),
                                BinOp::Div => unreachable!("int div handled earlier"),
                                BinOp::Mod | BinOp::Remainder => vv[i].wrapping_rem(s),
                            };
                        }
                    }
                } else {
                    for i in start..end {
                        vv[i] = match op {
                            BinOp::Add => vv[i].wrapping_add(s),
                            BinOp::Sub => vv[i].wrapping_sub(s),
                            BinOp::Mul => vv[i].wrapping_mul(s),
                            BinOp::Div => unreachable!("int div handled earlier"),
                            BinOp::Mod | BinOp::Remainder => vv[i].wrapping_rem(s),
                        };
                    }
                }
            }
            (LeafBuffer::F32(v), DType::Float32) => {
                let s = value as f32;
                let vv = Arc::make_mut(v);
                if has_nulls {
                    for i in start..end {
                        if leaf.validity[i] {
                            vv[i] = match op {
                                BinOp::Add => vv[i] + s,
                                BinOp::Sub => vv[i] - s,
                                BinOp::Mul => vv[i] * s,
                                BinOp::Div => vv[i] / s,
                                BinOp::Mod | BinOp::Remainder => vv[i] % s,
                            };
                        }
                    }
                } else {
                    for i in start..end {
                        vv[i] = match op {
                            BinOp::Add => vv[i] + s,
                            BinOp::Sub => vv[i] - s,
                            BinOp::Mul => vv[i] * s,
                            BinOp::Div => vv[i] / s,
                            BinOp::Mod | BinOp::Remainder => vv[i] % s,
                        };
                    }
                }
            }
            (LeafBuffer::F64(v), DType::Float64) => {
                let s = value as f64;
                let vv = Arc::make_mut(v);
                if has_nulls {
                    for i in start..end {
                        if leaf.validity[i] {
                            vv[i] = match op {
                                BinOp::Add => vv[i] + s,
                                BinOp::Sub => vv[i] - s,
                                BinOp::Mul => vv[i] * s,
                                BinOp::Div => vv[i] / s,
                                BinOp::Mod | BinOp::Remainder => vv[i] % s,
                            };
                        }
                    }
                } else {
                    for i in start..end {
                        vv[i] = match op {
                            BinOp::Add => vv[i] + s,
                            BinOp::Sub => vv[i] - s,
                            BinOp::Mul => vv[i] * s,
                            BinOp::Div => vv[i] / s,
                            BinOp::Mod | BinOp::Remainder => vv[i] % s,
                        };
                    }
                }
            }
            _ => {
                // Not supported in-place (yet).
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn apply_layout_visible_range(
        layout: &mut Layout,
        dt: DType,
        op: BinOp,
        value: f64,
        is_int: bool,
        outer_start: usize,
        outer_stop: usize,
    ) -> PyResult<bool> {
        match layout {
            Layout::Leaf(leaf) => {
                // For a leaf at this level, the visible range is [outer_start, outer_stop).
                apply_leaf_range(leaf, dt, op, value, is_int, outer_start, outer_stop)
            }
            Layout::ListOffset(lo) => {
                // Map outer visible range to inner content range using offsets.
                let s = lo.offsets[outer_start] as usize;
                let e = lo.offsets[outer_stop] as usize;
                apply_layout_visible_range(&mut lo.content, dt, op, value, is_int, s, e)
            }
            Layout::OffsetView(v) => {
                // OffsetView is a view on the outer axis of its offsets.
                let start = v.start + outer_start;
                let stop = v.start + outer_stop;
                if stop > v.stop {
                    return Err(PyValueError::new_err("OffsetView range out of bounds."));
                }
                let s = v.offsets[start] as usize;
                let e = v.offsets[stop] as usize;
                apply_layout_visible_range(&mut v.content, dt, op, value, is_int, s, e)
            }
            Layout::Indexed(_) => Ok(false), // cannot easily map visible range
            Layout::UnionScalarList(_) => Ok(false), // scalar ops on union not supported in compiled in-place path
        }
    }

    // Start with the full visible outer range of the array.
    let outer_len = a.layout.len();
    let ok = apply_layout_visible_range(&mut a.layout, a.dtype, op, value, is_int, 0, outer_len)?;
    Ok(ok)
}

pub fn elementwise_with_scalar(
    a: &GrumpyArray,
    op: BinOp,
    value: f64,
    is_int: bool,
) -> PyResult<GrumpyArray> {
    if let Some(out) = elementwise_rect2d_scalar_fast(a, op, value, is_int)? {
        return Ok(out);
    }
    if let Some(out) = elementwise_listoffset2d_scalar_fast(a, op, value, is_int)? {
        return Ok(out);
    }
    let mut out = a.clone();
    if !elementwise_scalar_inplace(&mut out, op, value, is_int)? {
        return Err(PyValueError::new_err(
            "Scalar elementwise operation is not supported for this layout/dtype.",
        ));
    }
    Ok(out)
}

pub fn elementwise(a: &GrumpyArray, b: &GrumpyArray, op: BinOp) -> PyResult<GrumpyArray> {
    let out_dtype = elementwise_out_dtype(a.dtype, b.dtype, op)?;
    let (a2, b2) = if a.dtype != b.dtype {
        crate::cast::cast_array_pair(a, b)?
    } else {
        (a.clone(), b.clone())
    };
    // Rectangular 2D fast path (ListOffset -> Leaf) for common dtypes, no nulls.
    if let Some(out) = elementwise_rect2d_fast(&a2, &b2, op)? {
        return Ok(out);
    }
    // Same-offset 2D list->leaf (ragged or rectangular): one flat leaf pass.
    if let Some(out) = elementwise_same_listoffset2d_fast(&a2, &b2, op)? {
        return Ok(out);
    }
    // Ragged 2D fast path (ListOffset -> Leaf) including per-row broadcast (len==1).
    if let Some(out) = elementwise_ragged2d_fast(&a2, &b2, op)? {
        return Ok(out);
    }
    // If the two layouts match exactly (including unions), we can run the generic recursive kernel
    // (no broadcasting). This is the Awkward-like "same structure" elementwise case.
    if layouts_compatible(&a2.layout, &b2.layout) {
        let layout = elementwise_layout(&a2.layout, &b2.layout, a2.dtype, b2.dtype, out_dtype, op)?;
        return Ok(GrumpyArray { dtype: out_dtype, layout });
    }

    // Broadcasting path: currently only supported for pure list-chains (no unions).
    if a2.layout.has_union() || b2.layout.has_union() {
        return Err(PyValueError::new_err(
            "Broadcasting on union layouts is not supported. If both operands have the same union structure, it is supported.",
        ));
    }

    let layout = elementwise_layout_broadcast(&a2.layout, &b2.layout, a2.dtype, b2.dtype, out_dtype, op)?;
    Ok(GrumpyArray { dtype: out_dtype, layout })
}

/// Write ``a op b`` into ``out`` when layouts match (same offsets, list->leaf, all-valid int32).
pub fn elementwise_into(out: &mut GrumpyArray, a: &GrumpyArray, b: &GrumpyArray, op: BinOp) -> PyResult<()> {
    if !layouts_compatible(&a.layout, &b.layout) || !layouts_compatible(&a.layout, &out.layout) {
        return Err(PyValueError::new_err(
            "elementwise out= requires identical list structure for a, b, and out.",
        ));
    }
    if a.dtype != DType::Int32 || b.dtype != DType::Int32 || out.dtype != DType::Int32 {
        return Err(PyValueError::new_err(
            "elementwise out= currently requires dtype=int32 for a, b, and out.",
        ));
    }
    let (off, leaf_a) = listoffset2d_leaf_view(&a.layout).ok_or_else(|| {
        PyValueError::new_err("elementwise out= requires 2D list->leaf layout.")
    })?;
    let leaf_b = listoffset2d_leaf_view(&b.layout).unwrap().1;
    let leaf_o = match &mut out.layout {
        Layout::ListOffset(lo) => match lo.content.as_mut() {
            Layout::Leaf(l) => l,
            _ => return Err(PyValueError::new_err("elementwise out= requires leaf content.")),
        },
        _ => return Err(PyValueError::new_err("elementwise out= requires list layout.")),
    };
    if leaf_a.has_nulls || leaf_b.has_nulls || leaf_o.has_nulls {
        return Err(PyValueError::new_err("elementwise out= requires all-valid arrays."));
    }
    let aa = match &leaf_a.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Err(PyValueError::new_err("elementwise out= requires int32 leaf.")),
    };
    let bb = match &leaf_b.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Err(PyValueError::new_err("elementwise out= requires int32 leaf.")),
    };
    let oo = match &mut leaf_o.buffer {
        LeafBuffer::I32(v) => Arc::make_mut(v),
        _ => return Err(PyValueError::new_err("elementwise out= requires int32 leaf.")),
    };
    let _ = off;
    match op {
        BinOp::Mul => crate::kernels::mul_i32_slices(aa, bb, oo),
        BinOp::Add => crate::kernels::add_i32_slices(aa, bb, oo),
        BinOp::Sub => crate::kernels::sub_i32_slices(aa, bb, oo),
        _ => {
            return Err(PyValueError::new_err(
                "elementwise out= supports add/sub/mul for int32 only.",
            ))
        }
    }
    Ok(())
}

/// Fused ``(a * scalar).sum()`` over all leaves for 2D list->leaf int32 arrays.
pub fn mul_scalar_sum_all_i64(a: &GrumpyArray, scalar: i32) -> PyResult<i64> {
    if a.dtype != DType::Int32 {
        return Err(PyValueError::new_err("mul_scalar_sum_all requires int32 array."));
    }
    let (_off, leaf) = listoffset2d_leaf_view(&a.layout)
        .ok_or_else(|| PyValueError::new_err("mul_scalar_sum_all requires 2D list->leaf layout."))?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("mul_scalar_sum_all requires all-valid array."));
    }
    let v = match &leaf.buffer {
        LeafBuffer::I32(buf) => buf.as_slice(),
        _ => return Err(PyValueError::new_err("mul_scalar_sum_all requires int32 leaf.")),
    };
    Ok(crate::kernels::sum_i32_mul_scalar_to_i64(v, scalar))
}

/// Fused ``(a * b).sum()`` over all leaves for matching 2D list->leaf int32 arrays.
pub fn mul_sum_all_i64(a: &GrumpyArray, b: &GrumpyArray) -> PyResult<i64> {
    if a.dtype != DType::Int32 || b.dtype != DType::Int32 {
        return Err(PyValueError::new_err("mul_sum_all requires int32 arrays."));
    }
    let (off_a, leaf_a) = listoffset2d_leaf_view(&a.layout)
        .ok_or_else(|| PyValueError::new_err("mul_sum_all requires 2D list->leaf layout."))?;
    let (off_b, leaf_b) = listoffset2d_leaf_view(&b.layout)
        .ok_or_else(|| PyValueError::new_err("mul_sum_all requires 2D list->leaf layout."))?;
    if off_a != off_b {
        return Err(PyValueError::new_err("mul_sum_all requires matching offsets."));
    }
    if leaf_a.has_nulls || leaf_b.has_nulls {
        return Err(PyValueError::new_err("mul_sum_all requires all-valid arrays."));
    }
    let aa = match &leaf_a.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Err(PyValueError::new_err("mul_sum_all requires int32 leaf.")),
    };
    let bb = match &leaf_b.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Err(PyValueError::new_err("mul_sum_all requires int32 leaf.")),
    };
    Ok(crate::kernels::sum_i32_mul_to_i64(aa, bb))
}

fn elementwise_out_dtype(a: DType, b: DType, op: BinOp) -> PyResult<DType> {
    match op {
        BinOp::Div => {
            let promoted = if a == b {
                a
            } else {
                crate::cast::promote_binary(a, b)?
            };
            match promoted {
                DType::Float16 | DType::Float32 | DType::Float64 => Ok(promoted),
                DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64 => Ok(DType::Float64),
                DType::Bool => Err(PyValueError::new_err(
                    "Arithmetic ops are not supported for dtype=bool (cast to an integer dtype first).",
                )),
                DType::Char | DType::String => Err(PyValueError::new_err(
                    "Division is only supported for numeric dtypes.",
                )),
            }
        }
        BinOp::Add => {
            if a == DType::String && b == DType::String {
                return Ok(DType::String);
            }
            let promoted = crate::cast::promote_binary(a, b)?;
            match promoted {
                DType::Bool => Err(PyValueError::new_err(
                    "Arithmetic ops are not supported for dtype=bool (cast to an integer dtype first).",
                )),
                DType::Char => Err(PyValueError::new_err(
                    "Operation not supported for char dtype (use string).",
                )),
                DType::String => Err(PyValueError::new_err(
                    "Only add is supported for string dtype.",
                )),
                _ => Ok(promoted),
            }
        }
        BinOp::Sub | BinOp::Mul | BinOp::Mod | BinOp::Remainder => {
            let promoted = crate::cast::promote_binary(a, b)?;
            match promoted {
                DType::Bool => Err(PyValueError::new_err(
                    "Arithmetic ops are not supported for dtype=bool (cast to an integer dtype first).",
                )),
                DType::Char | DType::String => Err(PyValueError::new_err(
                    "Operation not supported for non-numeric dtypes.",
                )),
                _ => Ok(promoted),
            }
        }
    }
}

fn elementwise_ragged2d_fast(a: &GrumpyArray, b: &GrumpyArray, op: BinOp) -> PyResult<Option<GrumpyArray>> {
    // Only supports depth=2 pure list chain: ListOffset -> Leaf
    let la = match &a.layout {
        Layout::ListOffset(lo) => lo,
        _ => return Ok(None),
    };
    let lb = match &b.layout {
        Layout::ListOffset(lo) => lo,
        _ => return Ok(None),
    };
    let leaf_a = match la.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Ok(None),
    };
    let leaf_b = match lb.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Ok(None),
    };
    // Outer axis must match for now (we handle axis0 broadcast elsewhere).
    let nrows = la.len();
    if nrows != lb.len() {
        return Ok(None);
    }

    // Determine output dtype rules (subset matching the existing fast paths).
    let out_dt = match op {
        BinOp::Div => {
            if a.dtype != b.dtype {
                return Ok(None);
            }
            match a.dtype {
                DType::Float32 | DType::Float64 => a.dtype,
                DType::Int32 | DType::Int64 => DType::Float64,
                _ => return Ok(None),
            }
        }
        _ => {
            if a.dtype != b.dtype {
                return Ok(None);
            }
            match a.dtype {
                DType::Int32 | DType::Int64 | DType::Float32 | DType::Float64 => a.dtype,
                _ => return Ok(None),
            }
        }
    };

    // Precompute per-row output lengths and offsets.
    let mut out_offsets: Vec<i64> = Vec::with_capacity(nrows + 1);
    out_offsets.push(0);
    let mut total: i64 = 0;
    for i in 0..nrows {
        let lena = la.offsets[i + 1] - la.offsets[i];
        let lenb = lb.offsets[i + 1] - lb.offsets[i];
        let out_len = if lena == lenb {
            lena
        } else if lena == 1 {
            lenb
        } else if lenb == 1 {
            lena
        } else {
            return Err(PyValueError::new_err(
                "Broadcasting failed for ragged2d: per-row lengths incompatible.",
            ));
        };
        total += out_len;
        out_offsets.push(total);
    }
    let n = total as usize;

    let mut out_leaf = Leaf::new(out_dt);
    out_leaf.len = n;
    let all_valid = !(leaf_a.has_nulls || leaf_b.has_nulls);
    out_leaf.has_nulls = !all_valid;
    out_leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out_leaf.buffer = match out_dt {
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        _ => return Ok(None),
    };

    // Write output values row-by-row.
    let mut out_pos = 0usize;
    if all_valid {
        match (a.dtype, &leaf_a.buffer, &leaf_b.buffer, &mut out_leaf.buffer, op) {
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai].wrapping_add(bb[bi]);
                        out_pos += 1;
                    }
                }
            }
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai].wrapping_sub(bb[bi]);
                        out_pos += 1;
                    }
                }
            }
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai].wrapping_mul(bb[bi]);
                        out_pos += 1;
                    }
                }
            }
            // int -> float64 div
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::F64(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = (aa[ai] as f64) / (bb[bi] as f64);
                        out_pos += 1;
                    }
                }
            }

            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai].wrapping_add(bb[bi]);
                        out_pos += 1;
                    }
                }
            }
            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai].wrapping_sub(bb[bi]);
                        out_pos += 1;
                    }
                }
            }
            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai].wrapping_mul(bb[bi]);
                        out_pos += 1;
                    }
                }
            }
            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::F64(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = (aa[ai] as f64) / (bb[bi] as f64);
                        out_pos += 1;
                    }
                }
            }

            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] + bb[bi];
                        out_pos += 1;
                    }
                }
            }
            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] - bb[bi];
                        out_pos += 1;
                    }
                }
            }
            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] * bb[bi];
                        out_pos += 1;
                    }
                }
            }
            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] / bb[bi];
                        out_pos += 1;
                    }
                }
            }

            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] + bb[bi];
                        out_pos += 1;
                    }
                }
            }
            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] - bb[bi];
                        out_pos += 1;
                    }
                }
            }
            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] * bb[bi];
                        out_pos += 1;
                    }
                }
            }
            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        oo[out_pos] = aa[ai] / bb[bi];
                        out_pos += 1;
                    }
                }
            }
            _ => return Ok(None),
        }
    } else {
        let out_valid = Arc::make_mut(&mut out_leaf.validity);
        match (a.dtype, &leaf_a.buffer, &leaf_b.buffer, &mut out_leaf.buffer, op) {
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0;
                        } else {
                            oo[out_pos] = aa[ai].wrapping_add(bb[bi]);
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0;
                        } else {
                            oo[out_pos] = aa[ai].wrapping_sub(bb[bi]);
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0;
                        } else {
                            oo[out_pos] = aa[ai].wrapping_mul(bb[bi]);
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::F64(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = (aa[ai] as f64) / (bb[bi] as f64);
                        }
                        out_pos += 1;
                    }
                }
            }

            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0;
                        } else {
                            oo[out_pos] = aa[ai].wrapping_add(bb[bi]);
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0;
                        } else {
                            oo[out_pos] = aa[ai].wrapping_sub(bb[bi]);
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0;
                        } else {
                            oo[out_pos] = aa[ai].wrapping_mul(bb[bi]);
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::F64(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = (aa[ai] as f64) / (bb[bi] as f64);
                        }
                        out_pos += 1;
                    }
                }
            }

            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] + bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] - bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] * bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] / bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }

            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Add) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] + bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Sub) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] - bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Mul) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] * bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Div) => {
                let oo = Arc::make_mut(oo);
                for i in 0..nrows {
                    let sa = la.offsets[i] as usize;
                    let ea = la.offsets[i + 1] as usize;
                    let sb = lb.offsets[i] as usize;
                    let eb = lb.offsets[i + 1] as usize;
                    let lena = ea - sa;
                    let lenb = eb - sb;
                    let out_len = if lena == lenb { lena } else if lena == 1 { lenb } else { lena };
                    for k in 0..out_len {
                        let ai = if lena == 1 { sa } else { sa + k };
                        let bi = if lenb == 1 { sb } else { sb + k };
                        let valid = leaf_a.validity[ai] && leaf_b.validity[bi];
                        if !valid {
                            out_leaf.has_nulls = true;
                            out_valid.set(out_pos, false);
                            oo[out_pos] = 0.0;
                        } else {
                            oo[out_pos] = aa[ai] / bb[bi];
                        }
                        out_pos += 1;
                    }
                }
            }
            _ => return Ok(None),
        }
    }

    let out_layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(out_offsets),
        content: Box::new(Layout::Leaf(out_leaf)),
    });
    Ok(Some(GrumpyArray { dtype: out_dt, layout: out_layout }))
}

fn elementwise_layout_broadcast(
    a: &Layout,
    b: &Layout,
    a_dt: DType,
    b_dt: DType,
    out_dt: DType,
    op: BinOp,
) -> PyResult<Layout> {
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => Ok(Layout::Leaf(elementwise_leaf_broadcast(la, lb, a_dt, b_dt, out_dt, op)?)),

        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => {
            let na = oa.len();
            let nb = ob.len();
            if na == nb && oa.offsets == ob.offsets {
                let content = elementwise_layout_broadcast(oa.content.as_ref(), ob.content.as_ref(), a_dt, b_dt, out_dt, op)?;
                return Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }));
            }
            // Axis-0 broadcast if one side has len==1
            if na == 1 && nb > 1 {
                return broadcast_listoffset_axis0(oa, ob, a_dt, b_dt, out_dt, op, true);
            }
            if nb == 1 && na > 1 {
                return broadcast_listoffset_axis0(oa, ob, a_dt, b_dt, out_dt, op, false);
            }
            // Same outer length but different offsets: broadcast per list element by computing each element separately
            // and concatenating along axis 0.
            if na == nb {
                let mut segs: Vec<Layout> = Vec::with_capacity(na);
                for i in 0..na {
                    let a_el = crate::layout::drop_axis0_select_element(&Layout::ListOffset(oa.clone()), i)?;
                    let b_el = crate::layout::drop_axis0_select_element(&Layout::ListOffset(ob.clone()), i)?;
                    segs.push(elementwise_layout_broadcast(&a_el, &b_el, a_dt, b_dt, out_dt, op)?);
                }
                let content = concat_axis0_layouts(&segs)?;
                // Build outer offsets from each segment length.
                let mut offsets: Vec<i64> = Vec::with_capacity(na + 1);
                offsets.push(0);
                let mut acc: i64 = 0;
                for s in &segs {
                    acc += s.len() as i64;
                    offsets.push(acc);
                }
                return Ok(Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(content) }));
            }
            Err(PyValueError::new_err("Broadcasting failed: incompatible outer lengths."))
        }

        // Scalar broadcast: leaf(len=1) over list
        (Layout::Leaf(la), Layout::ListOffset(ob)) if la.len == 1 => {
            let content = elementwise_layout_broadcast(a, ob.content.as_ref(), a_dt, b_dt, out_dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: ob.offsets.clone(), content: Box::new(content) }))
        }
        (Layout::ListOffset(oa), Layout::Leaf(lb)) if lb.len == 1 => {
            let content = elementwise_layout_broadcast(oa.content.as_ref(), b, a_dt, b_dt, out_dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }))
        }

        (Layout::OffsetView(v), other) => {
            // Treat OffsetView as listoffset-like by materializing a lightweight ListOffset over the view range.
            let mut offs: Vec<i64> = Vec::with_capacity(v.len() + 1);
            let base = v.offsets[v.start];
            for i in v.start..=v.stop {
                offs.push(v.offsets[i] - base);
            }
            let child_start = v.offsets[v.start] as usize;
            let child_end = v.offsets[v.stop] as usize;
            let content = crate::layout::take_range(v.content.as_ref(), child_start, child_end)?;
            let as_lo = Layout::ListOffset(ListOffset { offsets: Arc::new(offs), content: Box::new(content) });
            elementwise_layout_broadcast(&as_lo, other, a_dt, b_dt, out_dt, op)
        }
        (other, Layout::OffsetView(v)) => {
            let mut offs: Vec<i64> = Vec::with_capacity(v.len() + 1);
            let base = v.offsets[v.start];
            for i in v.start..=v.stop {
                offs.push(v.offsets[i] - base);
            }
            let child_start = v.offsets[v.start] as usize;
            let child_end = v.offsets[v.stop] as usize;
            let content = crate::layout::take_range(v.content.as_ref(), child_start, child_end)?;
            let as_lo = Layout::ListOffset(ListOffset { offsets: Arc::new(offs), content: Box::new(content) });
            elementwise_layout_broadcast(other, &as_lo, a_dt, b_dt, out_dt, op)
        }

        _ => Err(PyValueError::new_err("Broadcasting failed: incompatible layouts.")),
    }
}

fn broadcast_listoffset_axis0(
    a: &ListOffset,
    b: &ListOffset,
    a_dt: DType,
    b_dt: DType,
    out_dt: DType,
    op: BinOp,
    a_is_scalar_axis0: bool,
) -> PyResult<Layout> {
    // Broadcast the single list element from the len==1 side across the other side's axis0.
    let (small, big, small_dt, big_dt, small_first) = if a_is_scalar_axis0 {
        (a, b, a_dt, b_dt, true)
    } else {
        (b, a, b_dt, a_dt, false)
    };
    let big_n = big.len();
    let small_el = crate::layout::drop_axis0_select_element(&Layout::ListOffset(small.clone()), 0)?;
    let mut segs: Vec<Layout> = Vec::with_capacity(big_n);
    for i in 0..big_n {
        let big_el = crate::layout::drop_axis0_select_element(&Layout::ListOffset(big.clone()), i)?;
        let seg = if small_first {
            elementwise_layout_broadcast(&small_el, &big_el, small_dt, big_dt, out_dt, op)?
        } else {
            elementwise_layout_broadcast(&big_el, &small_el, big_dt, small_dt, out_dt, op)?
        };
        segs.push(seg);
    }
    let content = concat_axis0_layouts(&segs)?;
    let mut offsets: Vec<i64> = Vec::with_capacity(big_n + 1);
    offsets.push(0);
    let mut acc: i64 = 0;
    for s in &segs {
        acc += s.len() as i64;
        offsets.push(acc);
    }
    Ok(Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(content) }))
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
            // allocate buffer with capacity and then extend
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
            // Concatenate listoffset arrays along axis0 by concatenating offsets (with shift)
            // and recursively concatenating the flattened contents.
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
                // append offsets[1..] shifted by acc
                for &o in &offs[1..] {
                    all_offsets.push(acc + o);
                }
                acc += *offs.last().unwrap();
                content_segs.push(lo.content.as_ref().clone());
            }
            let content = concat_axis0_layouts(&content_segs)?;
            Ok(Layout::ListOffset(ListOffset { offsets: Arc::new(all_offsets), content: Box::new(content) }))
        }
        _ => Err(PyValueError::new_err("concat_axis0_layouts: unsupported layout kind.")),
    }
}

fn elementwise_leaf_broadcast(
    a: &Leaf,
    b: &Leaf,
    a_dt: DType,
    b_dt: DType,
    out_dt: DType,
    op: BinOp,
) -> PyResult<Leaf> {
    if a_dt != b_dt && !(op == BinOp::Div && out_dt == DType::Float64) {
        return Err(PyValueError::new_err("Broadcasting still requires matching dtypes (casting not implemented)."));
    }
    let n = if a.len == b.len { a.len } else if a.len == 1 { b.len } else if b.len == 1 { a.len } else {
        return Err(PyValueError::new_err("Broadcasting failed: leaf lengths incompatible."));
    };
    let mut out = Leaf::new(out_dt);
    out.len = n;
    let a_all = !a.has_nulls;
    let b_all = !b.has_nulls;
    out.has_nulls = !(a_all && b_all);
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    if out.has_nulls {
        let vv = Arc::make_mut(&mut out.validity);
        for i in 0..n {
            let ai = if a.len == 1 { 0 } else { i };
            let bi = if b.len == 1 { 0 } else { i };
            if !(a.validity[ai] && b.validity[bi]) {
                vv.set(i, false);
            }
        }
    }
    // Allocate output buffer.
    out.buffer = match out_dt {
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        _ => return Err(PyValueError::new_err("Broadcasting leaf only implemented for int32/int64/float32/float64.")),
    };

    // Division of ints/uints produces float64 (already handled in read_as_f64 path); broadcast indices apply.
    if op == BinOp::Div && out_dt == DType::Float64 {
        let outv = match &mut out.buffer {
            LeafBuffer::F64(v) => Arc::make_mut(v),
            _ => unreachable!(),
        };
        for i in 0..n {
            if out.has_nulls && !out.validity[i] { continue; }
            let ai = if a.len == 1 { 0 } else { i };
            let bi = if b.len == 1 { 0 } else { i };
            let av = read_as_f64(a_dt, &a.buffer, ai)?;
            let bv = read_as_f64(b_dt, &b.buffer, bi)?;
            outv[i] = av / bv;
        }
        return Ok(out);
    }

    match (a_dt, &a.buffer, &b.buffer, &mut out.buffer) {
        (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if out.has_nulls && !out.validity[i] { continue; }
                let ai = if a.len == 1 { 0 } else { i };
                let bi = if b.len == 1 { 0 } else { i };
                oo[i] = match op {
                    BinOp::Add => aa[ai].wrapping_add(bb[bi]),
                    BinOp::Sub => aa[ai].wrapping_sub(bb[bi]),
                    BinOp::Mul => aa[ai].wrapping_mul(bb[bi]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected int32 div path.")),
                    BinOp::Mod | BinOp::Remainder => i32_modlike(aa[ai], bb[bi])?,
                };
            }
        }
        (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if out.has_nulls && !out.validity[i] { continue; }
                let ai = if a.len == 1 { 0 } else { i };
                let bi = if b.len == 1 { 0 } else { i };
                oo[i] = match op {
                    BinOp::Add => aa[ai].wrapping_add(bb[bi]),
                    BinOp::Sub => aa[ai].wrapping_sub(bb[bi]),
                    BinOp::Mul => aa[ai].wrapping_mul(bb[bi]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected int64 div path.")),
                    BinOp::Mod | BinOp::Remainder => i64_modlike(aa[ai], bb[bi])?,
                };
            }
        }
        (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if out.has_nulls && !out.validity[i] { continue; }
                let ai = if a.len == 1 { 0 } else { i };
                let bi = if b.len == 1 { 0 } else { i };
                oo[i] = match op {
                    BinOp::Add => aa[ai] + bb[bi],
                    BinOp::Sub => aa[ai] - bb[bi],
                    BinOp::Mul => aa[ai] * bb[bi],
                    BinOp::Div => aa[ai] / bb[bi],
                    BinOp::Mod | BinOp::Remainder => aa[ai] - bb[bi] * (aa[ai] / bb[bi]).floor(),
                };
            }
        }
        (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if out.has_nulls && !out.validity[i] { continue; }
                let ai = if a.len == 1 { 0 } else { i };
                let bi = if b.len == 1 { 0 } else { i };
                oo[i] = match op {
                    BinOp::Add => aa[ai] + bb[bi],
                    BinOp::Sub => aa[ai] - bb[bi],
                    BinOp::Mul => aa[ai] * bb[bi],
                    BinOp::Div => aa[ai] / bb[bi],
                    BinOp::Mod | BinOp::Remainder => aa[ai] - bb[bi] * (aa[ai] / bb[bi]).floor(),
                };
            }
        }
        _ => return Err(PyValueError::new_err("Broadcasting leaf op not implemented for this dtype.")),
    }

    Ok(out)
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

fn build_listoffset2d_from_leaf(offsets: &[i64], leaf: Leaf, dtype: DType) -> GrumpyArray {
    GrumpyArray {
        dtype,
        layout: Layout::ListOffset(ListOffset {
            offsets: Arc::new(offsets.to_vec()),
            content: Box::new(Layout::Leaf(leaf)),
        }),
    }
}

fn elementwise_same_listoffset2d_fast(
    a: &GrumpyArray,
    b: &GrumpyArray,
    op: BinOp,
) -> PyResult<Option<GrumpyArray>> {
    let (off_a, leaf_a) = match listoffset2d_leaf_view(&a.layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    let (off_b, leaf_b) = match listoffset2d_leaf_view(&b.layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if off_a != off_b || a.dtype != b.dtype {
        return Ok(None);
    }
    if leaf_a.has_nulls || leaf_b.has_nulls {
        return Ok(None);
    }
    if a.dtype != DType::Int32 {
        return Ok(None);
    }
    let n = leaf_a.len;
    if n != leaf_b.len {
        return Ok(None);
    }
    let aa = match &leaf_a.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Ok(None),
    };
    let bb = match &leaf_b.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Ok(None),
    };
    let mut out_vec = vec![0i32; n];
    match op {
        BinOp::Mul => crate::kernels::mul_i32_slices(aa, bb, &mut out_vec),
        BinOp::Add => crate::kernels::add_i32_slices(aa, bb, &mut out_vec),
        BinOp::Sub => crate::kernels::sub_i32_slices(aa, bb, &mut out_vec),
        _ => return Ok(None),
    }
    let mut out_leaf = Leaf::new(DType::Int32);
    out_leaf.len = n;
    out_leaf.has_nulls = false;
    out_leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out_leaf.buffer = LeafBuffer::I32(Arc::new(out_vec));
    Ok(Some(build_listoffset2d_from_leaf(off_a, out_leaf, DType::Int32)))
}

fn elementwise_listoffset2d_scalar_fast(
    a: &GrumpyArray,
    op: BinOp,
    value: f64,
    is_int: bool,
) -> PyResult<Option<GrumpyArray>> {
    if !is_int || a.dtype != DType::Int32 {
        return Ok(None);
    }
    let (offsets, leaf_a) = match listoffset2d_leaf_view(&a.layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if leaf_a.has_nulls {
        return Ok(None);
    }
    let s = value as i32;
    let n = leaf_a.len;
    let aa = match &leaf_a.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Ok(None),
    };
    let mut out_vec = vec![0i32; n];
    match op {
        BinOp::Mul => crate::kernels::mul_i32_scalar_slice(aa, s, &mut out_vec),
        BinOp::Add => {
            for i in 0..n {
                out_vec[i] = aa[i].wrapping_add(s);
            }
        }
        BinOp::Sub => {
            for i in 0..n {
                out_vec[i] = aa[i].wrapping_sub(s);
            }
        }
        _ => return Ok(None),
    }
    let mut out_leaf = Leaf::new(DType::Int32);
    out_leaf.len = n;
    out_leaf.has_nulls = false;
    out_leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out_leaf.buffer = LeafBuffer::I32(Arc::new(out_vec));
    Ok(Some(build_listoffset2d_from_leaf(offsets, out_leaf, DType::Int32)))
}

fn rect2d_shape(layout: &Layout) -> Option<(usize, usize, &Vec<i64>)> {
    // Only recognizes a pure 2D list-chain: ListOffset -> Leaf, with constant row length.
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
        return Some((0, 0, &lo.offsets));
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
    Some((nrows, first, &lo.offsets))
}

fn elementwise_rect2d_scalar_fast(
    a: &GrumpyArray,
    op: BinOp,
    value: f64,
    is_int: bool,
) -> PyResult<Option<GrumpyArray>> {
    if !is_int {
        return Ok(None);
    }
    let (nrows, ncols, _offsets) = match rect2d_shape(&a.layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    let leaf_a = match &a.layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(l) => l,
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    if leaf_a.has_nulls || a.dtype != DType::Int32 {
        return Ok(None);
    }
    let s = value as i32;
    let n = nrows * ncols;
    let aa = match &leaf_a.buffer {
        LeafBuffer::I32(v) => v.as_slice(),
        _ => return Ok(None),
    };
    let out_vec: Vec<i32> = match op {
        BinOp::Add => aa.iter().map(|&x| x.wrapping_add(s)).collect(),
        BinOp::Sub => aa.iter().map(|&x| x.wrapping_sub(s)).collect(),
        BinOp::Mul => aa.iter().map(|&x| x.wrapping_mul(s)).collect(),
        _ => return Ok(None),
    };
    debug_assert_eq!(out_vec.len(), n);
    let mut out_leaf = Leaf::new(DType::Int32);
    out_leaf.len = n;
    out_leaf.has_nulls = false;
    out_leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out_leaf.buffer = LeafBuffer::I32(Arc::new(out_vec));
    let out_layout = Layout::ListOffset(ListOffset {
        offsets: match &a.layout {
            Layout::ListOffset(lo) => lo.offsets.clone(),
            _ => unreachable!(),
        },
        content: Box::new(Layout::Leaf(out_leaf)),
    });
    Ok(Some(GrumpyArray {
        dtype: DType::Int32,
        layout: out_layout,
    }))
}

fn elementwise_rect2d_fast(a: &GrumpyArray, b: &GrumpyArray, op: BinOp) -> PyResult<Option<GrumpyArray>> {
    let (nrows, ncols, offsets_a) = match rect2d_shape(&a.layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    let (_nrows2, _ncols2, offsets_b) = match rect2d_shape(&b.layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if offsets_a != offsets_b {
        return Ok(None);
    }
    // Only when no unions and no nulls (all-valid).
    let leaf_a = match &a.layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() { Layout::Leaf(l) => l, _ => return Ok(None) },
        _ => return Ok(None),
    };
    let leaf_b = match &b.layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() { Layout::Leaf(l) => l, _ => return Ok(None) },
        _ => return Ok(None),
    };
    if leaf_a.has_nulls || leaf_b.has_nulls {
        return Ok(None);
    }

    // Dtype rules: require same dtype for add/sub/mul/mod; div: float stays float, int -> float64.
    let out_dt = match op {
        BinOp::Div => {
            if a.dtype != b.dtype {
                return Ok(None);
            }
            match a.dtype {
                DType::Float32 | DType::Float64 => a.dtype,
                DType::Int32 | DType::Int64 | DType::UInt32 | DType::UInt64 => DType::Float64,
                _ => return Ok(None),
            }
        }
        _ => {
            if a.dtype != b.dtype {
                return Ok(None);
            }
            match a.dtype {
                DType::Int32 | DType::Int64 | DType::Float32 | DType::Float64 => a.dtype,
                _ => return Ok(None),
            }
        }
    };

    let n = nrows * ncols;
    let mut out_leaf = Leaf::new(out_dt);
    out_leaf.len = n;
    out_leaf.has_nulls = false;
    out_leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out_leaf.buffer = match out_dt {
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        _ => return Ok(None),
    };

    // Tight contiguous loops (LLVM auto-vectorizes these).
    match (a.dtype, &leaf_a.buffer, &leaf_b.buffer, &mut out_leaf.buffer, op) {
        (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Mul) => {
            let out_vec: Vec<i32> = aa
                .iter()
                .zip(bb.iter())
                .map(|(&a, &b)| a.wrapping_mul(b))
                .collect();
            out_leaf.buffer = LeafBuffer::I32(Arc::new(out_vec));
        }
        (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Add) => {
            let out_vec: Vec<i32> = aa
                .iter()
                .zip(bb.iter())
                .map(|(&a, &b)| a.wrapping_add(b))
                .collect();
            out_leaf.buffer = LeafBuffer::I32(Arc::new(out_vec));
        }
        (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo), BinOp::Sub) => {
            let out_vec: Vec<i32> = aa
                .iter()
                .zip(bb.iter())
                .map(|(&a, &b)| a.wrapping_sub(b))
                .collect();
            out_leaf.buffer = LeafBuffer::I32(Arc::new(out_vec));
        }
        (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Add) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i].wrapping_add(bb[i]); }
        }
        (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Sub) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i].wrapping_sub(bb[i]); }
        }
        (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo), BinOp::Mul) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i].wrapping_mul(bb[i]); }
        }
        (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Add) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] + bb[i]; }
        }
        (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Sub) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] - bb[i]; }
        }
        (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Mul) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] * bb[i]; }
        }
        (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo), BinOp::Div) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] / bb[i]; }
        }
        (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Add) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] + bb[i]; }
        }
        (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Sub) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] - bb[i]; }
        }
        (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Mul) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] * bb[i]; }
        }
        (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo), BinOp::Div) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = aa[i] / bb[i]; }
        }
        // int -> float64 division
        (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::F64(oo), BinOp::Div) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = (aa[i] as f64) / (bb[i] as f64); }
        }
        (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::F64(oo), BinOp::Div) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n { oo[i] = (aa[i] as f64) / (bb[i] as f64); }
        }
        _ => return Ok(None),
    }

    let out_layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(offsets_a.clone()),
        content: Box::new(Layout::Leaf(out_leaf)),
    });
    Ok(Some(GrumpyArray { dtype: out_dt, layout: out_layout }))
}

fn layouts_compatible(a: &Layout, b: &Layout) -> bool {
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => la.len == lb.len,
        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => oa.offsets == ob.offsets && layouts_compatible(oa.content.as_ref(), ob.content.as_ref()),
        (Layout::Indexed(ia), Layout::Indexed(ib)) => ia.index == ib.index && layouts_compatible(ia.content.as_ref(), ib.content.as_ref()),
        (Layout::OffsetView(va), Layout::OffsetView(vb)) => {
            va.start == vb.start
                && va.stop == vb.stop
                && va.offsets == vb.offsets
                && layouts_compatible(va.content.as_ref(), vb.content.as_ref())
        }
        (Layout::UnionScalarList(ua), Layout::UnionScalarList(ub)) => {
            ua.tags == ub.tags
                && ua.index == ub.index
                && ua.lists.offsets == ub.lists.offsets
                && layouts_compatible(&Layout::Leaf(ua.scalars.clone()), &Layout::Leaf(ub.scalars.clone()))
                && layouts_compatible(ua.lists.content.as_ref(), ub.lists.content.as_ref())
        }
        _ => false,
    }
}

fn elementwise_layout(
    a: &Layout,
    b: &Layout,
    a_dt: DType,
    b_dt: DType,
    out_dt: DType,
    op: BinOp,
) -> PyResult<Layout> {
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => Ok(Layout::Leaf(elementwise_leaf(la, lb, a_dt, b_dt, out_dt, op)?)),
        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => {
            let content = elementwise_layout(oa.content.as_ref(), ob.content.as_ref(), a_dt, b_dt, out_dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }))
        }
        (Layout::Indexed(ia), Layout::Indexed(ib)) => {
            let content = elementwise_layout(ia.content.as_ref(), ib.content.as_ref(), a_dt, b_dt, out_dt, op)?;
            Ok(Layout::Indexed(crate::layout::Indexed { index: ia.index.clone(), content: Box::new(content) }))
        }
        (Layout::OffsetView(va), Layout::OffsetView(vb)) => {
            let content = elementwise_layout(va.content.as_ref(), vb.content.as_ref(), a_dt, b_dt, out_dt, op)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: va.offsets.clone(),
                start: va.start,
                stop: va.stop,
                content: Box::new(content),
            }))
        }
        (Layout::UnionScalarList(ua), Layout::UnionScalarList(ub)) => {
            // scalar branch
            let scalars = elementwise_leaf(&ua.scalars, &ub.scalars, a_dt, b_dt, out_dt, op)?;
            // list branch
            let list_content = elementwise_layout(ua.lists.content.as_ref(), ub.lists.content.as_ref(), a_dt, b_dt, out_dt, op)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: ua.tags.clone(),
                index: ua.index.clone(),
                scalars,
                lists: ListOffset { offsets: ua.lists.offsets.clone(), content: Box::new(list_content) },
            }))
        }
        _ => Err(PyValueError::new_err("Internal error: incompatible layouts.")),
    }
}

fn elementwise_leaf(
    a: &Leaf,
    b: &Leaf,
    a_dt: DType,
    b_dt: DType,
    out_dt: DType,
    op: BinOp,
) -> PyResult<Leaf> {
    if a.len != b.len {
        return Err(PyValueError::new_err("Internal error: leaf length mismatch."));
    }
    let n = a.len;
    let mut out = Leaf::new(out_dt);
    out.len = n;
    let all_valid = !(a.has_nulls || b.has_nulls);
    if all_valid {
        out.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
        out.has_nulls = false;
    } else {
        let mut vv = (*a.validity).clone();
        vv &= b.validity.as_bitslice();
        out.validity = Arc::new(vv);
        out.has_nulls = true;
    }

    // Allocate typed output buffer.
    out.buffer = match out_dt {
        DType::Int8 => LeafBuffer::I8(Arc::new(vec![0i8; n])),
        DType::Int16 => LeafBuffer::I16(Arc::new(vec![0i16; n])),
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::UInt8 => LeafBuffer::U8(Arc::new(vec![0u8; n])),
        DType::UInt16 => LeafBuffer::U16(Arc::new(vec![0u16; n])),
        DType::UInt32 => LeafBuffer::U32(Arc::new(vec![0u32; n])),
        DType::UInt64 => LeafBuffer::U64(Arc::new(vec![0u64; n])),
        DType::Float16 => LeafBuffer::F16(Arc::new(vec![0u16; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        DType::Bool => LeafBuffer::Bool(Arc::new(vec![0u8; n])),
        DType::Char => LeafBuffer::Char(Arc::new(vec![0u32; n])),
        DType::String => {
            if op != BinOp::Add {
                return Err(PyValueError::new_err(
                    "Only add (concatenation) is supported for dtype=string.",
                ));
            }
            let aa = match &a.buffer {
                LeafBuffer::String(v) => v.as_slice(),
                _ => unreachable!(),
            };
            let bb = match &b.buffer {
                LeafBuffer::String(v) => v.as_slice(),
                _ => unreachable!(),
            };
            let mut out_s: Vec<String> = Vec::with_capacity(n);
            for i in 0..n {
                if !all_valid && !out.validity[i] {
                    out_s.push(String::new());
                    continue;
                }
                out_s.push(format!("{}{}", aa[i], bb[i]));
            }
            out.buffer = LeafBuffer::String(Arc::new(out_s));
            return Ok(out);
        }
    };

    // Division of ints/uints produces float64
    if op == BinOp::Div && out_dt == DType::Float64 {
        let outv = match &mut out.buffer {
            LeafBuffer::F64(v) => Arc::make_mut(v),
            _ => unreachable!(),
        };
        for i in 0..n {
            if !all_valid && !out.validity[i] {
                continue;
            }
            let av = read_as_f64(a_dt, &a.buffer, i)?;
            let bv = read_as_f64(b_dt, &b.buffer, i)?;
            outv[i] = av / bv;
        }
        return Ok(out);
    }

    // Same dtype operations
    if a_dt != b_dt {
        return Err(PyValueError::new_err("Internal error: dtype mismatch in elementwise leaf."));
    }

    match (a_dt, &a.buffer, &b.buffer, &mut out.buffer) {
        (DType::Int8, LeafBuffer::I8(aa), LeafBuffer::I8(bb), LeafBuffer::I8(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected int8 div path.")),
                    BinOp::Mod | BinOp::Remainder => i8_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::Int16, LeafBuffer::I16(aa), LeafBuffer::I16(bb), LeafBuffer::I16(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected int16 div path.")),
                    BinOp::Mod | BinOp::Remainder => i16_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::Int32, LeafBuffer::I32(aa), LeafBuffer::I32(bb), LeafBuffer::I32(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected int32 div path.")),
                    BinOp::Mod | BinOp::Remainder => i32_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::Int64, LeafBuffer::I64(aa), LeafBuffer::I64(bb), LeafBuffer::I64(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected int64 div path.")),
                    BinOp::Mod | BinOp::Remainder => i64_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::UInt8, LeafBuffer::U8(aa), LeafBuffer::U8(bb), LeafBuffer::U8(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected uint8 div path.")),
                    BinOp::Mod | BinOp::Remainder => u8_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::UInt16, LeafBuffer::U16(aa), LeafBuffer::U16(bb), LeafBuffer::U16(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected uint16 div path.")),
                    BinOp::Mod | BinOp::Remainder => u16_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::UInt32, LeafBuffer::U32(aa), LeafBuffer::U32(bb), LeafBuffer::U32(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected uint32 div path.")),
                    BinOp::Mod | BinOp::Remainder => u32_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::UInt64, LeafBuffer::U64(aa), LeafBuffer::U64(bb), LeafBuffer::U64(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i].wrapping_add(bb[i]),
                    BinOp::Sub => aa[i].wrapping_sub(bb[i]),
                    BinOp::Mul => aa[i].wrapping_mul(bb[i]),
                    BinOp::Div => return Err(PyValueError::new_err("Unexpected uint64 div path.")),
                    BinOp::Mod | BinOp::Remainder => u64_modlike(aa[i], bb[i])?,
                };
            }
        }
        (DType::Float16, LeafBuffer::F16(aa), LeafBuffer::F16(bb), LeafBuffer::F16(oo)) => {
            use half::f16;
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                let a = f16::from_bits(aa[i]).to_f32();
                let b = f16::from_bits(bb[i]).to_f32();
                let v = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Mod | BinOp::Remainder => a - b * (a / b).floor(),
                };
                oo[i] = f16::from_f32(v).to_bits();
            }
        }
        (DType::Float32, LeafBuffer::F32(aa), LeafBuffer::F32(bb), LeafBuffer::F32(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i] + bb[i],
                    BinOp::Sub => aa[i] - bb[i],
                    BinOp::Mul => aa[i] * bb[i],
                    BinOp::Div => aa[i] / bb[i],
                    BinOp::Mod | BinOp::Remainder => aa[i] - bb[i] * (aa[i] / bb[i]).floor(),
                };
            }
        }
        (DType::Float64, LeafBuffer::F64(aa), LeafBuffer::F64(bb), LeafBuffer::F64(oo)) => {
            let oo = Arc::make_mut(oo);
            for i in 0..n {
                if !all_valid && !out.validity[i] { continue; }
                oo[i] = match op {
                    BinOp::Add => aa[i] + bb[i],
                    BinOp::Sub => aa[i] - bb[i],
                    BinOp::Mul => aa[i] * bb[i],
                    BinOp::Div => aa[i] / bb[i],
                    BinOp::Mod | BinOp::Remainder => aa[i] - bb[i] * (aa[i] / bb[i]).floor(),
                };
            }
        }
        _ => return Err(PyValueError::new_err("Internal error: dtype buffer mismatch in elementwise.")),
    }

    Ok(out)
}

fn read_as_f64(dt: DType, buf: &LeafBuffer, i: usize) -> PyResult<f64> {
    match dt {
        DType::Int32 => match buf { LeafBuffer::I32(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::Int64 => match buf { LeafBuffer::I64(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::UInt32 => match buf { LeafBuffer::U32(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::UInt64 => match buf { LeafBuffer::U64(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::Int8 => match buf { LeafBuffer::I8(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::Int16 => match buf { LeafBuffer::I16(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::UInt8 => match buf { LeafBuffer::U8(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::UInt16 => match buf { LeafBuffer::U16(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::Float32 => match buf { LeafBuffer::F32(v) => Ok(v[i] as f64), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::Float64 => match buf { LeafBuffer::F64(v) => Ok(v[i]), _ => Err(PyValueError::new_err("dtype mismatch")) },
        DType::Float16 => {
            use half::f16;
            match buf {
                LeafBuffer::F16(v) => Ok(f16::from_bits(v[i]).to_f32() as f64),
                _ => Err(PyValueError::new_err("dtype mismatch")),
            }
        }
        _ => Err(PyValueError::new_err("Non-numeric dtype in division.")),
    }
}

// Python/numpy-style remainder for signed ints: result has sign of divisor.
fn i8_modlike(a: i8, b: i8) -> PyResult<i8> {
    if b == 0 {
        return Err(PyValueError::new_err("Modulo by zero."));
    }
    let r = a % b;
    if r == 0 {
        return Ok(0);
    }
    if (r < 0) != (b < 0) {
        Ok(r + b)
    } else {
        Ok(r)
    }
}
fn i16_modlike(a: i16, b: i16) -> PyResult<i16> {
    if b == 0 {
        return Err(PyValueError::new_err("Modulo by zero."));
    }
    let r = a % b;
    if r == 0 {
        return Ok(0);
    }
    if (r < 0) != (b < 0) {
        Ok(r + b)
    } else {
        Ok(r)
    }
}
fn i32_modlike(a: i32, b: i32) -> PyResult<i32> {
    if b == 0 {
        return Err(PyValueError::new_err("Modulo by zero."));
    }
    let r = a % b;
    if r == 0 {
        return Ok(0);
    }
    if (r < 0) != (b < 0) {
        Ok(r + b)
    } else {
        Ok(r)
    }
}
fn i64_modlike(a: i64, b: i64) -> PyResult<i64> {
    if b == 0 {
        return Err(PyValueError::new_err("Modulo by zero."));
    }
    let r = a % b;
    if r == 0 {
        return Ok(0);
    }
    if (r < 0) != (b < 0) {
        Ok(r + b)
    } else {
        Ok(r)
    }
}

fn u8_modlike(a: u8, b: u8) -> PyResult<u8> {
    if b == 0 {
        Err(PyValueError::new_err("Modulo by zero."))
    } else {
        Ok(a % b)
    }
}
fn u16_modlike(a: u16, b: u16) -> PyResult<u16> {
    if b == 0 {
        Err(PyValueError::new_err("Modulo by zero."))
    } else {
        Ok(a % b)
    }
}
fn u32_modlike(a: u32, b: u32) -> PyResult<u32> {
    if b == 0 {
        Err(PyValueError::new_err("Modulo by zero."))
    } else {
        Ok(a % b)
    }
}
fn u64_modlike(a: u64, b: u64) -> PyResult<u64> {
    if b == 0 {
        Err(PyValueError::new_err("Modulo by zero."))
    } else {
        Ok(a % b)
    }
}


