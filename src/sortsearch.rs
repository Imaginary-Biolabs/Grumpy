use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

pub fn sort(_py: Python<'_>, x: &GrumpyArray) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("sort does not support nulls yet."));
    }
    match x.dtype {
        DType::Int32 => sort_i32(leaf),
        DType::Int64 => sort_i64(leaf),
        DType::UInt32 => sort_u32(leaf),
        DType::UInt64 => sort_u64(leaf),
        DType::Float32 => sort_f32(leaf),
        DType::Float64 => sort_f64(leaf),
        DType::Bool => sort_bool(leaf),
        DType::Char => sort_char(leaf),
        _ => Err(PyValueError::new_err("sort not implemented for this dtype.")),
    }
}

pub fn sort_axis(py: Python<'_>, x: &GrumpyArray, dim: isize) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Err(PyValueError::new_err("sort on union layouts not implemented."));
    }
    let depth = crate::layout::list_chain_depth(&x.layout)
        .ok_or_else(|| PyValueError::new_err("sort requires a pure list-chain array."))?;
    let axis = normalize_axis(dim, depth)?;
    if depth == 0 {
        if axis != 0 {
            return Err(PyValueError::new_err("sort: dim out of range."));
        }
        return sort(py, x);
    }
    if axis != depth {
        // Only support sorting of scalar values within lists (last axis). Sorting lists-of-lists is undefined here.
        return Err(PyValueError::new_err(
            "sort is only supported on the innermost axis for nested ragged arrays (dim=-1).",
        ));
    }
    Ok(GrumpyArray { dtype: x.dtype, layout: sort_last_layout(&x.layout, x.dtype)? })
}

pub fn argsort(_py: Python<'_>, x: &GrumpyArray) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("argsort does not support nulls yet."));
    }
    match x.dtype {
        DType::Int32 => argsort_i32(leaf),
        DType::Int64 => argsort_i64(leaf),
        DType::UInt32 => argsort_u32(leaf),
        DType::UInt64 => argsort_u64(leaf),
        DType::Float32 => argsort_f32(leaf),
        DType::Float64 => argsort_f64(leaf),
        DType::Bool => argsort_bool(leaf),
        DType::Char => argsort_char(leaf),
        _ => Err(PyValueError::new_err("argsort not implemented for this dtype.")),
    }
}

pub fn argsort_axis(py: Python<'_>, x: &GrumpyArray, dim: isize) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Err(PyValueError::new_err("argsort on union layouts not implemented."));
    }
    let depth = crate::layout::list_chain_depth(&x.layout)
        .ok_or_else(|| PyValueError::new_err("argsort requires a pure list-chain array."))?;
    let axis = normalize_axis(dim, depth)?;
    if depth == 0 {
        if axis != 0 {
            return Err(PyValueError::new_err("argsort: dim out of range."));
        }
        return argsort(py, x);
    }
    if axis != depth {
        return Err(PyValueError::new_err(
            "argsort is only supported on the innermost axis for nested ragged arrays (dim=-1).",
        ));
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: argsort_last_layout(&x.layout, x.dtype)? })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgOp {
    ArgMax,
    ArgMin,
    NanArgMax,
    NanArgMin,
}

pub enum ArgOut {
    Scalar(PyObject),
}

pub fn argreduce(py: Python<'_>, x: &GrumpyArray, op: ArgOp) -> PyResult<ArgOut> {
    let leaf = leaf_1d(&x.layout)?;
    // Skip nulls always. For nan* also skip NaNs.
    match x.dtype {
        DType::Float64 => argreduce_f64(py, leaf, op),
        DType::Float32 => argreduce_f32(py, leaf, op),
        DType::Int32 => argreduce_i32(py, leaf, op),
        DType::Int64 => argreduce_i64(py, leaf, op),
        DType::UInt32 => argreduce_u32(py, leaf, op),
        DType::UInt64 => argreduce_u64(py, leaf, op),
        _ => Err(PyValueError::new_err("argmin/argmax not implemented for this dtype.")),
    }
}

#[allow(dead_code)]
pub fn argreduce_dim1(_py: Python<'_>, x: &GrumpyArray, op: ArgOp) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Err(PyValueError::new_err("argmax/argmin(dim=1) on union layouts not implemented."));
    }
    let (lo, leaf) = listoffset_leaf2d(&x.layout)?;
    let nrows = lo.len();

    let mut out = Leaf::new(DType::Int64);
    out.len = nrows;
    out.has_nulls = true;
    out.validity = Arc::new(bitvec![u8, Lsb0; 0; nrows]);
    out.buffer = LeafBuffer::I64(Arc::new(vec![0i64; nrows]));
    let outv = match &mut out.buffer { LeafBuffer::I64(v) => Arc::make_mut(v), _ => unreachable!() };
    let out_valid = Arc::make_mut(&mut out.validity);

    match x.dtype {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: i32 = 0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                                ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: i64 = 0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                                ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::UInt32 => {
            let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: u32 = 0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                                ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::UInt64 => {
            let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: u64 = 0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                                ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: f64 = 0.0;
                let mut seen_non_nan = false;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) && x.is_nan() {
                        continue;
                    }
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; seen_non_nan = !x.is_nan(); }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax => cmp_f64_argmax(x, best_val),
                                ArgOp::ArgMin => cmp_f64_argmin(x, best_val),
                                ArgOp::NanArgMax => x > best_val,
                                ArgOp::NanArgMin => x < best_val,
                            };
                            if better {
                                best_ix = Some(i - s);
                                best_val = x;
                                seen_non_nan = seen_non_nan || !x.is_nan();
                            }
                        }
                    }
                }
                if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) && !seen_non_nan {
                    continue;
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: f32 = 0.0;
                let mut seen_non_nan = false;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) && x.is_nan() {
                        continue;
                    }
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; seen_non_nan = !x.is_nan(); }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax => cmp_f32_argmax(x, best_val),
                                ArgOp::ArgMin => cmp_f32_argmin(x, best_val),
                                ArgOp::NanArgMax => x > best_val,
                                ArgOp::NanArgMin => x < best_val,
                            };
                            if better {
                                best_ix = Some(i - s);
                                best_val = x;
                                seen_non_nan = seen_non_nan || !x.is_nan();
                            }
                        }
                    }
                }
                if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) && !seen_non_nan {
                    continue;
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        _ => return Err(PyValueError::new_err("argmin/argmax(dim=1) not implemented for this dtype.")),
    }

    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(out) })
}

pub fn argreduce_axis_array(_py: Python<'_>, x: &GrumpyArray, dim: isize, op: ArgOp) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Err(PyValueError::new_err("argmax/argmin on union layouts not implemented."));
    }
    let depth = crate::layout::list_chain_depth(&x.layout)
        .ok_or_else(|| PyValueError::new_err("argmax/argmin requires a pure list-chain array."))?;
    let axis = normalize_axis(dim, depth)?;
    if depth == 0 {
        return Err(PyValueError::new_err("argmax/argmin(dim=...) on 1D arrays returns a scalar; call without dim."));
    }
    if axis != depth {
        return Err(PyValueError::new_err(
            "argmax/argmin are only supported on the innermost axis for nested ragged arrays (dim=-1).",
        ));
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: argreduce_last_layout(&x.layout, x.dtype, op)? })
}

fn normalize_axis(dim: isize, depth: usize) -> PyResult<usize> {
    let nd = depth as isize + 1;
    let mut d = dim;
    if d < 0 {
        d += nd;
    }
    if d < 0 || d >= nd {
        return Err(PyValueError::new_err("dim out of range."));
    }
    Ok(d as usize)
}

fn sort_last_layout(layout: &Layout, dt: DType) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(_) => Err(PyValueError::new_err("Internal error: sort_last_layout expected list.")),
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(leaf) => Ok(sort_listoffset_leaf(lo, leaf, dt)?.layout),
            _ => Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(sort_last_layout(lo.content.as_ref(), dt)?),
            })),
        },
        _ => Err(PyValueError::new_err("sort_last_layout requires list-chains only.")),
    }
}

fn argsort_last_layout(layout: &Layout, dt: DType) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(_) => Err(PyValueError::new_err("Internal error: argsort_last_layout expected list.")),
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(leaf) => Ok(argsort_listoffset_leaf(lo, leaf, dt)?.layout),
            _ => Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(argsort_last_layout(lo.content.as_ref(), dt)?),
            })),
        },
        _ => Err(PyValueError::new_err("argsort_last_layout requires list-chains only.")),
    }
}

fn argreduce_last_layout(layout: &Layout, dt: DType, op: ArgOp) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(_) => Err(PyValueError::new_err("Internal error: argreduce_last_layout expected list.")),
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(leaf) => Ok(argreduce_listoffset_leaf(lo, leaf, dt, op)?.layout),
            _ => Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(argreduce_last_layout(lo.content.as_ref(), dt, op)?),
            })),
        },
        _ => Err(PyValueError::new_err("argreduce_last_layout requires list-chains only.")),
    }
}

fn sort_listoffset_leaf(lo: &ListOffset, leaf: &Leaf, dt: DType) -> PyResult<GrumpyArray> {
    // Equivalent to sort(dim=1) for 2D list->leaf arrays, but without Python dependency.
    if leaf.has_nulls {
        return Err(PyValueError::new_err("sort(dim=-1) does not support nulls yet."));
    }
    let nrows = lo.len();
    let out = match dt {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0i32; leaf.len];
            let mut scratch: Vec<i32> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.sort();
                outv[s..e].copy_from_slice(&scratch);
            }
            GrumpyArray {
                dtype: DType::Int32,
                layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i32_from(outv))) }),
            }
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0i64; leaf.len];
            let mut scratch: Vec<i64> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.sort();
                outv[s..e].copy_from_slice(&scratch);
            }
            GrumpyArray {
                dtype: DType::Int64,
                layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))) }),
            }
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0f32; leaf.len];
            let mut scratch: Vec<f32> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.sort_by(|a, b| cmp_f32_numpy(*a, *b));
                outv[s..e].copy_from_slice(&scratch);
            }
            GrumpyArray {
                dtype: DType::Float32,
                layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_f32_from(outv))) }),
            }
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0f64; leaf.len];
            let mut scratch: Vec<f64> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.sort_by(|a, b| cmp_f64_numpy(*a, *b));
                outv[s..e].copy_from_slice(&scratch);
            }
            GrumpyArray {
                dtype: DType::Float64,
                layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_f64_from(outv))) }),
            }
        }
        _ => return Err(PyValueError::new_err("sort(dim=-1) not implemented for this dtype.")),
    };
    Ok(out)
}

fn argsort_listoffset_leaf(lo: &ListOffset, leaf: &Leaf, dt: DType) -> PyResult<GrumpyArray> {
    if leaf.has_nulls {
        return Err(PyValueError::new_err("argsort(dim=-1) does not support nulls yet."));
    }
    let nrows = lo.len();
    let mut outv: Vec<i64> = vec![0; leaf.len];
    match dt {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let mut idx: Vec<usize> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let m = e - s;
                idx.clear();
                idx.extend(0..m);
                idx.sort_unstable_by(|&i, &j| v[s + i].cmp(&v[s + j]));
                for (k, &ix) in idx.iter().enumerate() {
                    outv[s + k] = ix as i64;
                }
            }
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            let mut idx: Vec<usize> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let m = e - s;
                idx.clear();
                idx.extend(0..m);
                idx.sort_unstable_by(|&i, &j| v[s + i].cmp(&v[s + j]));
                for (k, &ix) in idx.iter().enumerate() {
                    outv[s + k] = ix as i64;
                }
            }
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            let mut idx: Vec<usize> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let m = e - s;
                idx.clear();
                idx.extend(0..m);
                idx.sort_unstable_by(|&i, &j| cmp_f32_numpy(v[s + i], v[s + j]));
                for (k, &ix) in idx.iter().enumerate() {
                    outv[s + k] = ix as i64;
                }
            }
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let mut idx: Vec<usize> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let m = e - s;
                idx.clear();
                idx.extend(0..m);
                idx.sort_unstable_by(|&i, &j| cmp_f64_numpy(v[s + i], v[s + j]));
                for (k, &ix) in idx.iter().enumerate() {
                    outv[s + k] = ix as i64;
                }
            }
        }
        _ => return Err(PyValueError::new_err("argsort(dim=-1) not implemented for this dtype.")),
    }
    Ok(GrumpyArray {
        dtype: DType::Int64,
        layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))) }),
    })
}

fn argreduce_listoffset_leaf(lo: &ListOffset, leaf: &Leaf, dt: DType, op: ArgOp) -> PyResult<GrumpyArray> {
    // Per-row argmax/argmin, producing a 1D int64 leaf of length nrows.
    let nrows = lo.len();
    let mut out = Leaf::new(DType::Int64);
    out.len = nrows;
    out.has_nulls = true;
    out.validity = Arc::new(bitvec![u8, Lsb0; 0; nrows]);
    out.buffer = LeafBuffer::I64(Arc::new(vec![0i64; nrows]));
    let outv = match &mut out.buffer { LeafBuffer::I64(v) => Arc::make_mut(v), _ => unreachable!() };
    let out_valid = Arc::make_mut(&mut out.validity);

    match dt {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: i32 = 0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                                ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: i64 = 0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                                ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: f32 = 0.0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    if (op == ArgOp::NanArgMax || op == ArgOp::NanArgMin) && x.is_nan() {
                        continue;
                    }
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => cmp_f32_numpy(x, best_val) == Ordering::Greater,
                                ArgOp::ArgMin | ArgOp::NanArgMin => cmp_f32_numpy(x, best_val) == Ordering::Less,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let mut best_ix: Option<usize> = None;
                let mut best_val: f64 = 0.0;
                for i in s..e {
                    if !leaf.validity[i] { continue; }
                    let x = v[i];
                    if (op == ArgOp::NanArgMax || op == ArgOp::NanArgMin) && x.is_nan() {
                        continue;
                    }
                    match best_ix {
                        None => { best_ix = Some(i - s); best_val = x; }
                        Some(_) => {
                            let better = match op {
                                ArgOp::ArgMax | ArgOp::NanArgMax => cmp_f64_numpy(x, best_val) == Ordering::Greater,
                                ArgOp::ArgMin | ArgOp::NanArgMin => cmp_f64_numpy(x, best_val) == Ordering::Less,
                            };
                            if better { best_ix = Some(i - s); best_val = x; }
                        }
                    }
                }
                if let Some(ix) = best_ix {
                    out_valid.set(r, true);
                    outv[r] = ix as i64;
                }
            }
        }
        _ => return Err(PyValueError::new_err("argmax/argmin(dim=-1) not implemented for this dtype.")),
    }

    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(out) })
}

pub fn nonzero(_py: Python<'_>, x: &GrumpyArray) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&x.layout)?;
    if x.dtype != DType::Bool {
        return Err(PyValueError::new_err("nonzero currently only supports bool arrays."));
    }
    let v = match &leaf.buffer {
        LeafBuffer::Bool(v) => v.as_slice(),
        _ => unreachable!(),
    };
    let mut idx: Vec<i64> = Vec::new();
    for i in 0..leaf.len {
        if !leaf.validity[i] {
            continue;
        }
        if v[i] != 0 {
            idx.push(i as i64);
        }
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx)) })
}

pub fn search_sorted(_py: Python<'_>, x: &GrumpyArray, v: &GrumpyArray, right: bool) -> PyResult<GrumpyArray> {
    let xl = leaf_1d(&x.layout)?;
    let vl = leaf_1d(&v.layout)?;
    if xl.has_nulls || vl.has_nulls {
        return Err(PyValueError::new_err("search_sorted does not support nulls yet."));
    }
    if x.dtype != v.dtype {
        return Err(PyValueError::new_err("search_sorted requires matching dtypes for now."));
    }
    let mut out = Leaf::new(DType::Int64);
    out.len = vl.len;
    out.has_nulls = false;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; vl.len]);
    out.buffer = LeafBuffer::I64(Arc::new(vec![0i64; vl.len]));
    let oo = match &mut out.buffer {
        LeafBuffer::I64(v) => Arc::make_mut(v),
        _ => unreachable!(),
    };

    match x.dtype {
        DType::Int32 => {
            let xs = match &xl.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let vs = match &vl.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..vl.len {
                oo[i] = if right {
                    upper_bound_i32(xs, xl.len, vs[i]) as i64
                } else {
                    lower_bound_i32(xs, xl.len, vs[i]) as i64
                };
            }
        }
        DType::Int64 => {
            let xs = match &xl.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            let vs = match &vl.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..vl.len {
                oo[i] = if right {
                    upper_bound_i64(xs, xl.len, vs[i]) as i64
                } else {
                    lower_bound_i64(xs, xl.len, vs[i]) as i64
                };
            }
        }
        DType::Float64 => {
            let xs = match &xl.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let vs = match &vl.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..vl.len {
                oo[i] = if right {
                    upper_bound_f64(xs, xl.len, vs[i]) as i64
                } else {
                    lower_bound_f64(xs, xl.len, vs[i]) as i64
                };
            }
        }
        DType::Float32 => {
            let xs = match &xl.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            let vs = match &vl.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..vl.len {
                oo[i] = if right {
                    upper_bound_f32(xs, xl.len, vs[i]) as i64
                } else {
                    lower_bound_f32(xs, xl.len, vs[i]) as i64
                };
            }
        }
        _ => return Err(PyValueError::new_err("search_sorted not implemented for this dtype.")),
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(out) })
}

pub fn partition(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("partition does not support nulls yet."));
    }
    if kth >= leaf.len {
        return Err(PyValueError::new_err("kth out of bounds."));
    }
    match x.dtype {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable(kth);
            Ok(GrumpyArray { dtype: DType::Int32, layout: Layout::Leaf(new_leaf_i32_from(out)) })
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable(kth);
            Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(out)) })
        }
        DType::UInt32 => {
            let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable(kth);
            Ok(GrumpyArray { dtype: DType::UInt32, layout: Layout::Leaf(new_leaf_u32_from(out)) })
        }
        DType::UInt64 => {
            let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable(kth);
            Ok(GrumpyArray { dtype: DType::UInt64, layout: Layout::Leaf(new_leaf_u64_from(out)) })
        }
        DType::Bool => {
            let v = match &leaf.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => unreachable!() };
            let mut out: Vec<u8> = v[..leaf.len].iter().map(|&x| (x != 0) as u8).collect();
            out.select_nth_unstable(kth);
            Ok(GrumpyArray { dtype: DType::Bool, layout: Layout::Leaf(new_leaf_bool_from(out)) })
        }
        DType::Char => {
            let v = match &leaf.buffer { LeafBuffer::Char(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable(kth);
            Ok(GrumpyArray { dtype: DType::Char, layout: Layout::Leaf(new_leaf_char_from(out)) })
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable_by(kth, |a, b| cmp_f64_numpy(*a, *b));
            Ok(GrumpyArray { dtype: DType::Float64, layout: Layout::Leaf(new_leaf_f64_from(out)) })
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            let mut out = v[..leaf.len].to_vec();
            out.select_nth_unstable_by(kth, |a, b| cmp_f32_numpy(*a, *b));
            Ok(GrumpyArray { dtype: DType::Float32, layout: Layout::Leaf(new_leaf_f32_from(out)) })
        }
        _ => Err(PyValueError::new_err("partition not implemented for this dtype.")),
    }
}

pub fn partition_dim1(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Err(PyValueError::new_err("partition(dim=1) on union layouts not implemented."));
    }
    let (lo, leaf) = listoffset_leaf2d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("partition(dim=1) does not support nulls yet."));
    }
    let nrows = lo.len();
    // validate kth for all rows
    for r in 0..nrows {
        let s = lo.offsets[r] as usize;
        let e = lo.offsets[r + 1] as usize;
        if kth >= (e - s) {
            return Err(PyValueError::new_err("kth out of bounds for at least one row."));
        }
    }
    match x.dtype {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0i32; leaf.len];
            let mut scratch: Vec<i32> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.select_nth_unstable(kth);
                outv[s..e].copy_from_slice(&scratch);
            }
            Ok(GrumpyArray { dtype: DType::Int32, layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i32_from(outv))) }) })
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0i64; leaf.len];
            let mut scratch: Vec<i64> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.select_nth_unstable(kth);
                outv[s..e].copy_from_slice(&scratch);
            }
            Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))) }) })
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0f64; leaf.len];
            let mut scratch: Vec<f64> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.select_nth_unstable_by(kth, |a, b| cmp_f64_numpy(*a, *b));
                outv[s..e].copy_from_slice(&scratch);
            }
            Ok(GrumpyArray { dtype: DType::Float64, layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_f64_from(outv))) }) })
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            let mut outv = vec![0f32; leaf.len];
            let mut scratch: Vec<f32> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                scratch.clear();
                scratch.extend_from_slice(&v[s..e]);
                scratch.select_nth_unstable_by(kth, |a, b| cmp_f32_numpy(*a, *b));
                outv[s..e].copy_from_slice(&scratch);
            }
            Ok(GrumpyArray { dtype: DType::Float32, layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_f32_from(outv))) }) })
        }
        _ => Err(PyValueError::new_err("partition(dim=1) not implemented for this dtype.")),
    }
}

pub fn argpartition(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("argpartition does not support nulls yet."));
    }
    if kth >= leaf.len {
        return Err(PyValueError::new_err("kth out of bounds."));
    }
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    match x.dtype {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| v[i].cmp(&v[j]));
        }
        DType::Int64 => {
            let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| v[i].cmp(&v[j]));
        }
        DType::UInt32 => {
            let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| v[i].cmp(&v[j]));
        }
        DType::UInt64 => {
            let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| v[i].cmp(&v[j]));
        }
        DType::Bool => {
            let v = match &leaf.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| ((v[i] != 0) as u8).cmp(&((v[j] != 0) as u8)));
        }
        DType::Char => {
            let v = match &leaf.buffer { LeafBuffer::Char(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| v[i].cmp(&v[j]));
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| cmp_f64_numpy(v[i], v[j]));
        }
        DType::Float32 => {
            let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            idx.select_nth_unstable_by(kth, |&i, &j| cmp_f32_numpy(v[i], v[j]));
        }
        _ => return Err(PyValueError::new_err("argpartition not implemented for this dtype.")),
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}

pub fn argpartition_dim1(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Err(PyValueError::new_err("argpartition(dim=1) on union layouts not implemented."));
    }
    let (lo, leaf) = listoffset_leaf2d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("argpartition(dim=1) does not support nulls yet."));
    }
    let nrows = lo.len();
    for r in 0..nrows {
        let s = lo.offsets[r] as usize;
        let e = lo.offsets[r + 1] as usize;
        if kth >= (e - s) {
            return Err(PyValueError::new_err("kth out of bounds for at least one row."));
        }
    }
    let mut outv: Vec<i64> = vec![0; leaf.len];
    match x.dtype {
        DType::Int32 => {
            let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let mut idx: Vec<usize> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let m = e - s;
                idx.clear();
                idx.extend(0..m);
                idx.select_nth_unstable_by(kth, |&i, &j| v[s + i].cmp(&v[s + j]));
                for (k, &ix) in idx.iter().enumerate() {
                    outv[s + k] = ix as i64;
                }
            }
        }
        DType::Float64 => {
            let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let mut idx: Vec<usize> = Vec::new();
            for r in 0..nrows {
                let s = lo.offsets[r] as usize;
                let e = lo.offsets[r + 1] as usize;
                let m = e - s;
                idx.clear();
                idx.extend(0..m);
                idx.select_nth_unstable_by(kth, |&i, &j| cmp_f64_numpy(v[s + i], v[s + j]));
                for (k, &ix) in idx.iter().enumerate() {
                    outv[s + k] = ix as i64;
                }
            }
        }
        _ => return Err(PyValueError::new_err("argpartition(dim=1) not implemented for this dtype.")),
    }
    Ok(GrumpyArray {
        dtype: DType::Int64,
        layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))) }),
    })
}

// -------- leaf helpers --------

fn leaf_1d<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::OffsetView(v) => leaf_1d(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_1d(ix.content.as_ref()),
        Layout::ListOffset(_) => Err(PyValueError::new_err("Expected 1D leaf array.")),
        Layout::UnionScalarList(_) => Err(PyValueError::new_err("Union not supported.")),
    }
}

fn listoffset_leaf2d<'a>(layout: &'a Layout) -> PyResult<(&'a ListOffset, &'a Leaf)> {
    match layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(l) => Ok((lo, l)),
            _ => Err(PyValueError::new_err("Expected 2D list->leaf array.")),
        },
        Layout::OffsetView(v) => listoffset_leaf2d(v.content.as_ref()),
        Layout::Indexed(ix) => listoffset_leaf2d(ix.content.as_ref()),
        _ => Err(PyValueError::new_err("Expected 2D list->leaf array.")),
    }
}

fn new_leaf_i64_from(v: Vec<i64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I64(Arc::new(v));
    leaf
}

fn new_leaf_i32_from(v: Vec<i32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(v));
    leaf
}

fn new_leaf_u32_from(v: Vec<u32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U32(Arc::new(v));
    leaf
}

fn new_leaf_u64_from(v: Vec<u64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U64(Arc::new(v));
    leaf
}

fn new_leaf_f32_from(v: Vec<f32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F32(Arc::new(v));
    leaf
}

fn new_leaf_f64_from(v: Vec<f64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F64(Arc::new(v));
    leaf
}

fn new_leaf_bool_from(v: Vec<u8>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Bool);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::Bool(Arc::new(v));
    leaf
}

fn new_leaf_char_from(v: Vec<u32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Char);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::Char(Arc::new(v));
    leaf
}

// -------- sort implementations --------

fn sort_i32(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort();
    Ok(GrumpyArray { dtype: DType::Int32, layout: Layout::Leaf(new_leaf_i32_from(out)) })
}
fn sort_i64(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort();
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(out)) })
}
fn sort_u32(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort();
    Ok(GrumpyArray { dtype: DType::UInt32, layout: Layout::Leaf(new_leaf_u32_from(out)) })
}
fn sort_u64(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort();
    Ok(GrumpyArray { dtype: DType::UInt64, layout: Layout::Leaf(new_leaf_u64_from(out)) })
}
fn sort_bool(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => unreachable!() };
    let mut out: Vec<u8> = v[..leaf.len].iter().map(|&x| (x != 0) as u8).collect();
    out.sort();
    Ok(GrumpyArray { dtype: DType::Bool, layout: Layout::Leaf(new_leaf_bool_from(out)) })
}
fn sort_char(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::Char(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort();
    Ok(GrumpyArray { dtype: DType::Char, layout: Layout::Leaf(new_leaf_char_from(out)) })
}
fn sort_f64(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort_by(|a, b| cmp_f64_numpy(*a, *b));
    Ok(GrumpyArray { dtype: DType::Float64, layout: Layout::Leaf(new_leaf_f64_from(out)) })
}
fn sort_f32(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
    let mut out = v[..leaf.len].to_vec();
    out.sort_by(|a, b| cmp_f32_numpy(*a, *b));
    Ok(GrumpyArray { dtype: DType::Float32, layout: Layout::Leaf(new_leaf_f32_from(out)) })
}

// -------- argsort implementations --------

fn argsort_i32(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_i64(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_u32(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_u64(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_bool(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| ((v[i] != 0) as u8).cmp(&((v[j] != 0) as u8)));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_char(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::Char(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_f64(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| cmp_f64_numpy(v[i], v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}
fn argsort_f32(leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
    let mut idx: Vec<usize> = (0..leaf.len).collect();
    idx.sort_unstable_by(|&i, &j| cmp_f32_numpy(v[i], v[j]));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}

// -------- argmin/argmax --------

fn argreduce_i32(py: Python<'_>, leaf: &Leaf, op: ArgOp) -> PyResult<ArgOut> {
    let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
    let mut best_ix: Option<usize> = None;
    let mut best_val: i32 = 0;
    for i in 0..leaf.len {
        if !leaf.validity[i] { continue; }
        let x = v[i];
        match best_ix {
            None => { best_ix = Some(i); best_val = x; }
            Some(_) => {
                let better = match op {
                    ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                    ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                };
                if better { best_ix = Some(i); best_val = x; }
            }
        }
    }
    match best_ix {
        Some(ix) => Ok(ArgOut::Scalar((ix as i64).into_py(py))),
        None => Ok(ArgOut::Scalar(py.None())),
    }
}
fn argreduce_i64(py: Python<'_>, leaf: &Leaf, op: ArgOp) -> PyResult<ArgOut> {
    let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
    let mut best_ix: Option<usize> = None;
    let mut best_val: i64 = 0;
    for i in 0..leaf.len {
        if !leaf.validity[i] { continue; }
        let x = v[i];
        match best_ix {
            None => { best_ix = Some(i); best_val = x; }
            Some(_) => {
                let better = match op {
                    ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                    ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                };
                if better { best_ix = Some(i); best_val = x; }
            }
        }
    }
    match best_ix {
        Some(ix) => Ok(ArgOut::Scalar((ix as i64).into_py(py))),
        None => Ok(ArgOut::Scalar(py.None())),
    }
}
fn argreduce_u32(py: Python<'_>, leaf: &Leaf, op: ArgOp) -> PyResult<ArgOut> {
    let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => unreachable!() };
    let mut best_ix: Option<usize> = None;
    let mut best_val: u32 = 0;
    for i in 0..leaf.len {
        if !leaf.validity[i] { continue; }
        let x = v[i];
        match best_ix {
            None => { best_ix = Some(i); best_val = x; }
            Some(_) => {
                let better = match op {
                    ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                    ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                };
                if better { best_ix = Some(i); best_val = x; }
            }
        }
    }
    match best_ix {
        Some(ix) => Ok(ArgOut::Scalar((ix as i64).into_py(py))),
        None => Ok(ArgOut::Scalar(py.None())),
    }
}
fn argreduce_u64(py: Python<'_>, leaf: &Leaf, op: ArgOp) -> PyResult<ArgOut> {
    let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => unreachable!() };
    let mut best_ix: Option<usize> = None;
    let mut best_val: u64 = 0;
    for i in 0..leaf.len {
        if !leaf.validity[i] { continue; }
        let x = v[i];
        match best_ix {
            None => { best_ix = Some(i); best_val = x; }
            Some(_) => {
                let better = match op {
                    ArgOp::ArgMax | ArgOp::NanArgMax => x > best_val,
                    ArgOp::ArgMin | ArgOp::NanArgMin => x < best_val,
                };
                if better { best_ix = Some(i); best_val = x; }
            }
        }
    }
    match best_ix {
        Some(ix) => Ok(ArgOut::Scalar((ix as i64).into_py(py))),
        None => Ok(ArgOut::Scalar(py.None())),
    }
}

fn argreduce_f64(py: Python<'_>, leaf: &Leaf, op: ArgOp) -> PyResult<ArgOut> {
    let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
    let mut best_ix: Option<usize> = None;
    let mut best_val: f64 = 0.0;
    let mut seen_non_nan = false;
    for i in 0..leaf.len {
        if !leaf.validity[i] { continue; }
        let x = v[i];
        if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) {
            if x.is_nan() { continue; }
        }
        match best_ix {
            None => { best_ix = Some(i); best_val = x; seen_non_nan = !x.is_nan(); }
            Some(_) => {
                // np.argmax with NaN tends to return first NaN (since initial best may be NaN).
                // We approximate NumPy ordering: treat NaN as greater than any number for argmax,
                // and as less than any number for argmin.
                let better = match op {
                    ArgOp::ArgMax => cmp_f64_argmax(x, best_val),
                    ArgOp::ArgMin => cmp_f64_argmin(x, best_val),
                    ArgOp::NanArgMax => x > best_val,
                    ArgOp::NanArgMin => x < best_val,
                };
                if better {
                    best_ix = Some(i);
                    best_val = x;
                    seen_non_nan = seen_non_nan || !x.is_nan();
                }
            }
        }
    }
    if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) && !seen_non_nan {
        // NumPy raises ValueError; we return None for now (API consistency with null-handling).
        return Ok(ArgOut::Scalar(py.None()));
    }
    match best_ix {
        Some(ix) => Ok(ArgOut::Scalar((ix as i64).into_py(py))),
        None => Ok(ArgOut::Scalar(py.None())),
    }
}

fn argreduce_f32(py: Python<'_>, leaf: &Leaf, op: ArgOp) -> PyResult<ArgOut> {
    let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
    let mut best_ix: Option<usize> = None;
    let mut best_val: f32 = 0.0;
    let mut seen_non_nan = false;
    for i in 0..leaf.len {
        if !leaf.validity[i] { continue; }
        let x = v[i];
        if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) {
            if x.is_nan() { continue; }
        }
        match best_ix {
            None => { best_ix = Some(i); best_val = x; seen_non_nan = !x.is_nan(); }
            Some(_) => {
                let better = match op {
                    ArgOp::ArgMax => cmp_f32_argmax(x, best_val),
                    ArgOp::ArgMin => cmp_f32_argmin(x, best_val),
                    ArgOp::NanArgMax => x > best_val,
                    ArgOp::NanArgMin => x < best_val,
                };
                if better {
                    best_ix = Some(i);
                    best_val = x;
                    seen_non_nan = seen_non_nan || !x.is_nan();
                }
            }
        }
    }
    if matches!(op, ArgOp::NanArgMax | ArgOp::NanArgMin) && !seen_non_nan {
        return Ok(ArgOut::Scalar(py.None()));
    }
    match best_ix {
        Some(ix) => Ok(ArgOut::Scalar((ix as i64).into_py(py))),
        None => Ok(ArgOut::Scalar(py.None())),
    }
}

#[inline]
fn cmp_f64_argmax(x: f64, best: f64) -> bool {
    if best.is_nan() { return false; } // keep first NaN
    if x.is_nan() { return true; }
    x > best
}
#[inline]
fn cmp_f64_argmin(x: f64, best: f64) -> bool {
    if best.is_nan() { return false; }
    if x.is_nan() { return true; } // NaN treated as minimal
    x < best
}
#[inline]
fn cmp_f32_argmax(x: f32, best: f32) -> bool {
    if best.is_nan() { return false; }
    if x.is_nan() { return true; }
    x > best
}
#[inline]
fn cmp_f32_argmin(x: f32, best: f32) -> bool {
    if best.is_nan() { return false; }
    if x.is_nan() { return true; }
    x < best
}

// -------- float sorting compare (NumPy-ish: NaNs last) --------

#[inline]
fn cmp_f64_numpy(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}
#[inline]
fn cmp_f32_numpy(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

// -------- searchsorted bounds --------

fn lower_bound_i32(xs: &[i32], n: usize, x: i32) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] < x { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn upper_bound_i32(xs: &[i32], n: usize, x: i32) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn lower_bound_i64(xs: &[i64], n: usize, x: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] < x { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn upper_bound_i64(xs: &[i64], n: usize, x: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn lower_bound_f64(xs: &[f64], n: usize, x: f64) -> usize {
    // Treat NaN as +inf for insertion (NumPy behavior yields end for NaN on sorted ascending).
    if x.is_nan() { return n; }
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cmp_f64_numpy(xs[mid], x) == Ordering::Less { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn upper_bound_f64(xs: &[f64], n: usize, x: f64) -> usize {
    if x.is_nan() { return n; }
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cmp_f64_numpy(xs[mid], x) != Ordering::Greater { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn lower_bound_f32(xs: &[f32], n: usize, x: f32) -> usize {
    if x.is_nan() { return n; }
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cmp_f32_numpy(xs[mid], x) == Ordering::Less { lo = mid + 1; } else { hi = mid; }
    }
    lo
}
fn upper_bound_f32(xs: &[f32], n: usize, x: f32) -> usize {
    if x.is_nan() { return n; }
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cmp_f32_numpy(xs[mid], x) != Ordering::Greater { lo = mid + 1; } else { hi = mid; }
    }
    lo
}


