use crate::dtype::DType;
use crate::error::{dim_out_of_range, dtype_unsupported, internal, unsupported};
use crate::layout::{drop_axis0_select_element, layout_ndim, GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use crate::layout_ops::{
    array_as_leaf_1d, leaf_1d, listoffset_leaf2d, map_last_axis, map_union_axis0, LastAxisLeafMode,
    new_leaf_bool_from, new_leaf_char_from, new_leaf_f32_from, new_leaf_f64_from, new_leaf_i32_from,
    new_leaf_i64_from, new_leaf_u32_from, new_leaf_u64_from,
};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

pub fn sort(_py: Python<'_>, x: &GrumpyArray) -> PyResult<GrumpyArray> {
    let leaf = array_as_leaf_1d(x)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("sort does not support nulls yet."));
    }
    match x.dtype {
        DType::Int32 => sort_i32(&leaf),
        DType::Int64 => sort_i64(&leaf),
        DType::UInt32 => sort_u32(&leaf),
        DType::UInt64 => sort_u64(&leaf),
        DType::Float32 => sort_f32(&leaf),
        DType::Float64 => sort_f64(&leaf),
        DType::Bool => sort_bool(&leaf),
        DType::Char => sort_char(&leaf),
        _ => Err(dtype_unsupported("sort", x.dtype)),
    }
}

pub fn sort_axis(py: Python<'_>, x: &GrumpyArray, dim: isize) -> PyResult<GrumpyArray> {
    let ndim = layout_ndim_for_sort(&x.layout)?;
    let axis = normalize_axis(dim, ndim)?;
    if ndim == 1 {
        if axis != 0 {
            return Err(dim_out_of_range(dim, 1));
        }
        return sort(py, x);
    }
    if axis != ndim - 1 {
        return Err(PyValueError::new_err(
            "sort is only supported on the innermost axis for nested ragged arrays (dim=-1).",
        ));
    }
    if x.layout.has_union() {
        return map_union_last_axis(py, x, |sub| sort(py, sub));
    }
    Ok(GrumpyArray { dtype: x.dtype, layout: sort_last_layout(&x.layout, x.dtype)? })
}

pub fn argsort(_py: Python<'_>, x: &GrumpyArray) -> PyResult<GrumpyArray> {
    let leaf = array_as_leaf_1d(x)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("argsort does not support nulls yet."));
    }
    match x.dtype {
        DType::Int32 => argsort_i32(&leaf),
        DType::Int64 => argsort_i64(&leaf),
        DType::UInt32 => argsort_u32(&leaf),
        DType::UInt64 => argsort_u64(&leaf),
        DType::Float32 => argsort_f32(&leaf),
        DType::Float64 => argsort_f64(&leaf),
        DType::Bool => argsort_bool(&leaf),
        DType::Char => argsort_char(&leaf),
        _ => Err(dtype_unsupported("argsort", x.dtype)),
    }
}

pub fn argsort_axis(py: Python<'_>, x: &GrumpyArray, dim: isize) -> PyResult<GrumpyArray> {
    let ndim = layout_ndim_for_sort(&x.layout)?;
    let axis = normalize_axis(dim, ndim)?;
    if ndim == 1 {
        if axis != 0 {
            return Err(dim_out_of_range(dim, 1));
        }
        return argsort(py, x);
    }
    if axis != ndim - 1 {
        return Err(PyValueError::new_err(
            "argsort is only supported on the innermost axis for nested ragged arrays (dim=-1).",
        ));
    }
    if x.layout.has_union() {
        return map_union_last_axis(py, x, |sub| argsort(py, sub));
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
    let leaf = array_as_leaf_1d(x)?;
    // Skip nulls always. For nan* also skip NaNs.
    match x.dtype {
        DType::Float64 => argreduce_f64(py, &leaf, op),
        DType::Float32 => argreduce_f32(py, &leaf, op),
        DType::Int32 => argreduce_i32(py, &leaf, op),
        DType::Int64 => argreduce_i64(py, &leaf, op),
        DType::UInt32 => argreduce_u32(py, &leaf, op),
        DType::UInt64 => argreduce_u64(py, &leaf, op),
        _ => Err(dtype_unsupported("argmin/argmax", x.dtype)),
    }
}

pub fn argreduce_axis_array(_py: Python<'_>, x: &GrumpyArray, dim: isize, op: ArgOp) -> PyResult<GrumpyArray> {
    let ndim = layout_ndim_for_sort(&x.layout)?;
    let axis = normalize_axis(dim, ndim)?;
    if ndim == 1 {
        return Err(PyValueError::new_err("argmax/argmin(dim=...) on 1D arrays returns a scalar; call without dim."));
    }
    if axis != ndim - 1 {
        return Err(PyValueError::new_err(
            "argmax/argmin are only supported on the innermost axis for nested ragged arrays (dim=-1).",
        ));
    }
    if x.layout.has_union() {
        return argreduce_union_last_axis(x, op);
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: argreduce_last_layout(&x.layout, x.dtype, op)? })
}

fn normalize_axis(dim: isize, ndim: usize) -> PyResult<usize> {
    let mut d = dim;
    if d < 0 {
        d += ndim as isize;
    }
    if d < 0 || d as usize >= ndim {
        return Err(dim_out_of_range(dim, ndim));
    }
    Ok(d as usize)
}

fn layout_ndim_for_sort(layout: &Layout) -> PyResult<usize> {
    if layout.has_union() {
        layout_ndim(layout)
    } else {
        let depth = crate::layout::list_chain_depth(layout).ok_or_else(|| {
            unsupported(
                "sort/search",
                "requires a list-chain or union array.",
                "build inputs with gr.array(...) as a list-chain or UnionScalarList layout.",
            )
        })?;
        Ok(depth + 1)
    }
}

fn map_union_last_axis(
    _py: Python<'_>,
    x: &GrumpyArray,
    mut f: impl FnMut(&GrumpyArray) -> PyResult<GrumpyArray>,
) -> PyResult<GrumpyArray> {
    let layout = map_union_axis0(&x.layout, x.dtype, |sub| {
        Ok(f(&GrumpyArray {
            dtype: x.dtype,
            layout: sub,
        })?
        .layout)
    })?;
    Ok(GrumpyArray {
        dtype: x.dtype,
        layout,
    })
}

fn argreduce_union_last_axis(x: &GrumpyArray, op: ArgOp) -> PyResult<GrumpyArray> {
    Python::with_gil(|py| {
        let n = x.len();
        let mut scalars = Leaf::new(DType::Int64);
        scalars.len = n;
        scalars.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
        scalars.has_nulls = false;
        scalars.buffer = LeafBuffer::I64(Arc::new(vec![0i64; n]));
        for i in 0..n {
            let sub = drop_axis0_select_element(&x.layout, i)?;
            let sub_arr = GrumpyArray {
                dtype: x.dtype,
                layout: sub,
            };
            let ix = argreduce_subtree_last(py, &sub_arr, op)?;
            match &mut scalars.buffer {
                LeafBuffer::I64(v) => Arc::make_mut(v)[i] = ix,
                _ => unreachable!(),
            }
        }
        let lists = ListOffset {
            offsets: Arc::new(vec![0i64]),
            content: Box::new(Layout::Leaf(Leaf::new(DType::Int64))),
        };
        Ok(GrumpyArray {
            dtype: DType::Int64,
            layout: Layout::UnionScalarList(UnionScalarList {
                tags: (0..n).map(|_| 0u8).collect(),
                index: (0..n as i64).collect(),
                scalars,
                lists,
            }),
        })
    })
}

fn argreduce_subtree_last(py: Python<'_>, x: &GrumpyArray, op: ArgOp) -> PyResult<i64> {
    if layout_ndim(&x.layout)? <= 1 {
        return match argreduce(py, x, op)? {
            ArgOut::Scalar(o) => o.extract(py),
        };
    }
    let out = argreduce_last_layout(&x.layout, x.dtype, op)?;
    read_first_i64_layout(&out)
}

fn read_first_i64_layout(layout: &Layout) -> PyResult<i64> {
    match layout {
        Layout::Leaf(l) => match &l.buffer {
            LeafBuffer::I64(v) => Ok(v[0]),
            _ => Err(internal("sort/search", "expected int64 leaf")),
        },
        Layout::ListOffset(lo) => read_first_i64_layout(lo.content.as_ref()),
        _ => Err(internal("sort/search", "expected int64 layout")),
    }
}

fn sort_last_layout(layout: &Layout, dt: DType) -> PyResult<Layout> {
    map_last_axis(layout, LastAxisLeafMode::PromoteShortLeaf, &|lo, leaf| {
        Ok(sort_listoffset_leaf(lo, leaf, dt)?.layout)
    })
}

fn argsort_last_layout(layout: &Layout, dt: DType) -> PyResult<Layout> {
    map_last_axis(layout, LastAxisLeafMode::PromoteShortLeaf, &|lo, leaf| {
        Ok(argsort_listoffset_leaf(lo, leaf, dt)?.layout)
    })
}

fn argreduce_last_layout(layout: &Layout, dt: DType, op: ArgOp) -> PyResult<Layout> {
    map_last_axis(layout, LastAxisLeafMode::PromoteShortLeaf, &|lo, leaf| {
        Ok(argreduce_listoffset_leaf(lo, leaf, dt, op)?.layout)
    })
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
        _ => return Err(dtype_unsupported("sort(dim=-1)", dt)),
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
        _ => return Err(dtype_unsupported("argsort(dim=-1)", dt)),
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
        _ => return Err(dtype_unsupported("argmax/argmin(dim=-1)", dt)),
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
    let xl = array_as_leaf_1d(x)?;
    let vl = array_as_leaf_1d(v)?;
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
        _ => return Err(dtype_unsupported("search_sorted", x.dtype)),
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(out) })
}

pub fn partition(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    let leaf = array_as_leaf_1d(x)?;
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
        _ => Err(dtype_unsupported("partition", x.dtype)),
    }
}

pub fn partition_dim1(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Ok(GrumpyArray {
            dtype: x.dtype,
            layout: partition_last_layout(&x.layout, x.dtype, kth)?,
        });
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
        _ => Err(dtype_unsupported("partition(dim=1)", x.dtype)),
    }
}

pub fn argpartition(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    let leaf = array_as_leaf_1d(x)?;
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
        _ => return Err(dtype_unsupported("argpartition", x.dtype)),
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx.into_iter().map(|i| i as i64).collect())) })
}

pub fn argpartition_dim1(_py: Python<'_>, x: &GrumpyArray, kth: usize) -> PyResult<GrumpyArray> {
    if x.layout.has_union() {
        return Ok(GrumpyArray {
            dtype: DType::Int64,
            layout: argpartition_last_layout(&x.layout, x.dtype, kth)?,
        });
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
        _ => return Err(dtype_unsupported("argpartition(dim=1)", x.dtype)),
    }
    Ok(GrumpyArray {
        dtype: DType::Int64,
        layout: Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))) }),
    })
}

fn partition_listoffset_leaf(lo: &ListOffset, leaf: &Leaf, dt: DType, kth: usize) -> PyResult<Layout> {
    if leaf.has_nulls {
        return Err(PyValueError::new_err("partition(dim=1) does not support nulls yet."));
    }
    let nrows = lo.len();
    for r in 0..nrows {
        let s = lo.offsets[r] as usize;
        let e = lo.offsets[r + 1] as usize;
        if kth >= (e - s) {
            return Err(PyValueError::new_err("kth out of bounds for at least one row."));
        }
    }
    match dt {
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
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(Layout::Leaf(new_leaf_i32_from(outv))),
            }))
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
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))),
            }))
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
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(Layout::Leaf(new_leaf_f64_from(outv))),
            }))
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
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(Layout::Leaf(new_leaf_f32_from(outv))),
            }))
        }
        _ => Err(dtype_unsupported("partition(dim=1)", dt)),
    }
}

fn argpartition_listoffset_leaf(lo: &ListOffset, leaf: &Leaf, dt: DType, kth: usize) -> PyResult<Layout> {
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
        _ => return Err(dtype_unsupported("argpartition(dim=1)", dt)),
    }
    Ok(Layout::ListOffset(ListOffset {
        offsets: lo.offsets.clone(),
        content: Box::new(Layout::Leaf(new_leaf_i64_from(outv))),
    }))
}

fn partition_last_layout(layout: &Layout, dt: DType, kth: usize) -> PyResult<Layout> {
    map_last_axis(layout, LastAxisLeafMode::RequireListOffset, &|lo, leaf| {
        partition_listoffset_leaf(lo, leaf, dt, kth)
    })
}

fn argpartition_last_layout(layout: &Layout, dt: DType, kth: usize) -> PyResult<Layout> {
    map_last_axis(layout, LastAxisLeafMode::RequireListOffset, &|lo, leaf| {
        argpartition_listoffset_leaf(lo, leaf, dt, kth)
    })
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


