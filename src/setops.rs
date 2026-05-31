use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::sync::Arc;

pub fn unique(py: Python<'_>, a: &GrumpyArray) -> PyResult<GrumpyArray> {
    if a.layout.has_union() {
        return unique_union(py, a);
    }
    let leaf = find_leaf(&a.layout)?;
    match a.dtype {
        DType::Int32 => unique_i32(py, leaf),
        DType::Int64 => unique_i64(py, leaf),
        DType::UInt32 => unique_u32(py, leaf),
        DType::UInt64 => unique_u64(py, leaf),
        DType::Bool => unique_bool(py, leaf),
        DType::Char => unique_char(py, leaf),
        DType::String => unique_string(py, leaf),
        DType::Float32 => unique_f32(py, leaf),
        DType::Float64 => unique_f64(py, leaf),
        _ => Err(PyValueError::new_err("unique() not implemented for this dtype.")),
    }
}

pub fn isin(a: &GrumpyArray, test: &GrumpyArray) -> PyResult<GrumpyArray> {
    let (a2, test2) = if a.dtype != test.dtype {
        crate::cast::cast_array_pair(a, test)?
    } else {
        (a.clone(), test.clone())
    };
    let test_leaf = find_leaf(&test2.layout)?;
    let set = build_membership_set(test2.dtype, test_leaf)?;
    let out_layout = isin_layout(&a2.layout, a2.dtype, &set)?;
    Ok(GrumpyArray {
        dtype: DType::Bool,
        layout: out_layout,
    })
}

pub fn setunion(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<GrumpyArray> {
    let (a2, b2) = if a.dtype != b.dtype {
        crate::cast::cast_array_pair(a, b)?
    } else {
        (a.clone(), b.clone())
    };
    // NumPy union1d is unique(concatenate(a,b)).
    let la = find_leaf(&a2.layout)?;
    let lb = find_leaf(&b2.layout)?;
    match a2.dtype {
        DType::Float32 => {
            let mut v = collect_f32(la);
            v.extend(collect_f32(lb));
            unique_f32_from_values(py, a2.dtype, &v)
        }
        DType::Float64 => {
            let mut v = collect_f64(la);
            v.extend(collect_f64(lb));
            unique_f64_from_values(py, a2.dtype, &v)
        }
        DType::String => {
            let mut v = collect_string(la);
            v.extend(collect_string(lb));
            unique_string_from_values(py, &v)
        }
        _ => {
            // generic: unique(concat) via hash then sort
            let mut vals = collect_scalar_bits(a2.dtype, la)?;
            vals.extend(collect_scalar_bits(a2.dtype, lb)?);
            unique_from_scalar_bits(py, a2.dtype, &vals)
        }
    }
}

pub fn setdiff(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<GrumpyArray> {
    let (a2, b2) = if a.dtype != b.dtype {
        crate::cast::cast_array_pair(a, b)?
    } else {
        (a.clone(), b.clone())
    };
    // NumPy setdiff1d is: unique(a) filtered by isin(..., invert=True) where NaN never matches.
    let ua = unique(py, a)?;
    let ub = unique(py, b)?;
    let la = find_leaf(&ua.layout)?;
    let lb = find_leaf(&ub.layout)?;
    match a.dtype {
        DType::Float32 => setdiff_f32(py, la, lb),
        DType::Float64 => setdiff_f64(py, la, lb),
        _ => {
            let mut av = collect_scalar_bits(a.dtype, la)?;
            let bv = collect_scalar_bits(a.dtype, lb)?;
            av.sort();
            let mut bvs = bv;
            bvs.sort();
            // filter with two-pointer
            let mut out: Vec<u64> = Vec::new();
            let mut j = 0usize;
            for &x in &av {
                while j < bvs.len() && bvs[j] < x {
                    j += 1;
                }
                if j == bvs.len() || bvs[j] != x {
                    out.push(x);
                }
            }
            unique_from_scalar_bits(py, a2.dtype, &out)
        }
    }
}

pub fn setxor(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<GrumpyArray> {
    let (a2, b2) = if a.dtype != b.dtype {
        crate::cast::cast_array_pair(a, b)?
    } else {
        (a.clone(), b.clone())
    };
    let ua = unique(py, &a2)?;
    let ub = unique(py, &b2)?;
    let la = find_leaf(&ua.layout)?;
    let lb = find_leaf(&ub.layout)?;
    match a2.dtype {
        DType::Float32 => setxor_f32(py, la, lb),
        DType::Float64 => setxor_f64(py, la, lb),
        _ => {
            let mut av = collect_scalar_bits(a.dtype, la)?;
            let mut bv = collect_scalar_bits(a.dtype, lb)?;
            av.sort();
            bv.sort();
            let mut out: Vec<u64> = Vec::new();
            let mut i = 0usize;
            let mut j = 0usize;
            while i < av.len() || j < bv.len() {
                if j == bv.len() || (i < av.len() && av[i] < bv[j]) {
                    out.push(av[i]);
                    i += 1;
                } else if i == av.len() || (j < bv.len() && bv[j] < av[i]) {
                    out.push(bv[j]);
                    j += 1;
                } else {
                    // equal -> skip both
                    i += 1;
                    j += 1;
                }
            }
            unique_from_scalar_bits(py, a2.dtype, &out)
        }
    }
}

// -------- helpers: layout traversal / leaf building --------

fn unique_union(py: Python<'_>, a: &GrumpyArray) -> PyResult<GrumpyArray> {
    match a.dtype {
        DType::Int32 => {
            let vals = collect_union_i32(&a.layout);
            unique_i32_from_values(py, a.dtype, &vals)
        }
        DType::Int64 => {
            let vals = collect_union_i64(&a.layout);
            unique_i64_from_values(py, a.dtype, &vals)
        }
        DType::Float64 => {
            let vals = collect_union_f64(&a.layout);
            unique_f64_from_values(py, a.dtype, &vals)
        }
        _ => Err(PyValueError::new_err(
            "unique() on union arrays supports int32/int64/float64 for now.",
        )),
    }
}

fn collect_union_i32(layout: &Layout) -> Vec<i32> {
    let mut out = Vec::new();
    walk_union_collect(layout, &mut |v_i64, _v_f64| {
        out.push(v_i64 as i32);
        Ok(())
    })
    .ok();
    out
}

fn collect_union_i64(layout: &Layout) -> Vec<i64> {
    let mut out = Vec::new();
    walk_union_collect(layout, &mut |v_i64, _v_f64| {
        out.push(v_i64);
        Ok(())
    })
    .ok();
    out
}

fn collect_union_f64(layout: &Layout) -> Vec<f64> {
    let mut out = Vec::new();
    walk_union_collect(layout, &mut |_v_i64, v_f64| {
        out.push(v_f64);
        Ok(())
    })
    .ok();
    out
}

fn walk_union_collect(
    layout: &Layout,
    f: &mut dyn FnMut(i64, f64) -> PyResult<()>,
) -> PyResult<()> {
    use crate::layout::drop_axis0_select_element;
    match layout {
        Layout::Leaf(l) => {
            for i in 0..l.len {
                if !l.validity[i] {
                    continue;
                }
                match &l.buffer {
                    LeafBuffer::I32(v) => f(v[i] as i64, v[i] as f64)?,
                    LeafBuffer::I64(v) => f(v[i], v[i] as f64)?,
                    LeafBuffer::F64(v) => f(v[i] as i64, v[i])?,
                    _ => {}
                }
            }
        }
        Layout::ListOffset(lo) => {
            for i in 0..lo.len() {
                walk_union_collect(&drop_axis0_select_element(layout, i)?, f)?;
            }
        }
        Layout::OffsetView(v) => {
            for i in 0..v.len() {
                walk_union_collect(&drop_axis0_select_element(layout, i)?, f)?;
            }
        }
        Layout::Indexed(ix) => {
            for i in 0..ix.len() {
                walk_union_collect(&drop_axis0_select_element(layout, i)?, f)?;
            }
        }
        Layout::UnionScalarList(u) => {
            for i in 0..u.len() {
                walk_union_collect(&drop_axis0_select_element(layout, i)?, f)?;
            }
        }
    }
    Ok(())
}

fn unique_i32_from_values(_py: Python<'_>, dt: DType, vals: &[i32]) -> PyResult<GrumpyArray> {
    let mut set: FxHashSet<i32> = FxHashSet::default();
    let mut out: Vec<i32> = Vec::new();
    for &v in vals {
        if set.insert(v) {
            out.push(v);
        }
    }
    out.sort_unstable();
    let mut leaf = Leaf::new(dt);
    leaf.len = out.len();
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; out.len()]);
    leaf.has_nulls = false;
    leaf.buffer = LeafBuffer::I32(Arc::new(out));
    Ok(GrumpyArray {
        dtype: dt,
        layout: Layout::Leaf(leaf),
    })
}

fn unique_i64_from_values(py: Python<'_>, dt: DType, vals: &[i64]) -> PyResult<GrumpyArray> {
    let _ = py;
    let mut set: FxHashSet<i64> = FxHashSet::default();
    let mut out: Vec<i64> = Vec::new();
    for &v in vals {
        if set.insert(v) {
            out.push(v);
        }
    }
    out.sort_unstable();
    let mut leaf = Leaf::new(dt);
    leaf.len = out.len();
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; out.len()]);
    leaf.has_nulls = false;
    leaf.buffer = LeafBuffer::I64(Arc::new(out));
    Ok(GrumpyArray {
        dtype: dt,
        layout: Layout::Leaf(leaf),
    })
}

fn find_leaf<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::ListOffset(lo) => find_leaf(lo.content.as_ref()),
        Layout::Indexed(ix) => find_leaf(ix.content.as_ref()),
        Layout::OffsetView(v) => find_leaf(v.content.as_ref()),
        Layout::UnionScalarList(u) => {
            if u.scalars.len > 0 {
                Ok(&u.scalars)
            } else {
                find_leaf(u.lists.content.as_ref())
            }
        }
    }
}

// -------- unique implementations --------

fn collect_i32(leaf: &Leaf) -> Vec<i32> {
    let v = match &leaf.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}
fn collect_i64(leaf: &Leaf) -> Vec<i64> {
    let v = match &leaf.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}
fn collect_u32(leaf: &Leaf) -> Vec<u32> {
    let v = match &leaf.buffer { LeafBuffer::U32(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}
fn collect_u64(leaf: &Leaf) -> Vec<u64> {
    let v = match &leaf.buffer { LeafBuffer::U64(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}
fn collect_char(leaf: &Leaf) -> Vec<u32> {
    let v = match &leaf.buffer { LeafBuffer::Char(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}
fn collect_bool(leaf: &Leaf) -> Vec<u8> {
    let v = match &leaf.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(if v[i] != 0 { 1 } else { 0 }); }
    }
    out
}
fn collect_f32(leaf: &Leaf) -> Vec<f32> {
    let v = match &leaf.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}
fn collect_f64(leaf: &Leaf) -> Vec<f64> {
    let v = match &leaf.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => &[] };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] { out.push(v[i]); }
    }
    out
}

fn unique_i32(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let mut v = collect_i32(leaf);
    v.sort();
    v.dedup();
    new_leaf_i32(v)
}
fn unique_i64(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let mut v = collect_i64(leaf);
    v.sort();
    v.dedup();
    new_leaf_i64(v)
}
fn unique_u32(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let mut v = collect_u32(leaf);
    v.sort();
    v.dedup();
    new_leaf_u32(v)
}
fn unique_u64(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let mut v = collect_u64(leaf);
    v.sort();
    v.dedup();
    new_leaf_u64(v)
}
fn unique_char(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let mut v = collect_char(leaf);
    v.sort();
    v.dedup();
    new_leaf_char(v)
}
fn collect_string(leaf: &Leaf) -> Vec<String> {
    let v = match &leaf.buffer {
        LeafBuffer::String(v) => v.as_slice(),
        _ => &[],
    };
    let mut out = Vec::new();
    out.reserve(leaf.len);
    for i in 0..leaf.len {
        if leaf.validity[i] {
            out.push(v[i].clone());
        }
    }
    out
}
fn unique_string(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    unique_string_from_values(_py, &collect_string(leaf))
}
fn unique_string_from_values(_py: Python<'_>, values: &[String]) -> PyResult<GrumpyArray> {
    let mut v = values.to_vec();
    v.sort();
    v.dedup();
    new_leaf_string(v)
}
fn new_leaf_string(v: Vec<String>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::String);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::String(Arc::new(v));
    Ok(GrumpyArray {
        dtype: DType::String,
        layout: Layout::Leaf(leaf),
    })
}
fn unique_bool(_py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let mut v = collect_bool(leaf);
    v.sort();
    v.dedup();
    new_leaf_bool(v)
}

fn unique_f32_from_values(_py: Python<'_>, _dt: DType, values: &[f32]) -> PyResult<GrumpyArray> {
    let mut v: Vec<f32> = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let mut out: Vec<f32> = Vec::new();
    for x in v {
        if out.is_empty() {
            out.push(x);
            continue;
        }
        let last = *out.last().unwrap();
        if x.is_nan() && last.is_nan() {
            continue;
        }
        if x == last {
            continue;
        }
        out.push(x);
    }
    new_leaf_f32(out)
}
fn unique_f64_from_values(_py: Python<'_>, _dt: DType, values: &[f64]) -> PyResult<GrumpyArray> {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let mut out: Vec<f64> = Vec::new();
    for x in v {
        if out.is_empty() {
            out.push(x);
            continue;
        }
        let last = *out.last().unwrap();
        if x.is_nan() && last.is_nan() {
            continue;
        }
        if x == last {
            continue;
        }
        out.push(x);
    }
    new_leaf_f64(out)
}

fn unique_f32(py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = collect_f32(leaf);
    unique_f32_from_values(py, DType::Float32, &v)
}
fn unique_f64(py: Python<'_>, leaf: &Leaf) -> PyResult<GrumpyArray> {
    let v = collect_f64(leaf);
    unique_f64_from_values(py, DType::Float64, &v)
}

fn new_leaf_i32(v: Vec<i32>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::Int32, layout: Layout::Leaf(leaf) })
}
fn new_leaf_i64(v: Vec<i64>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I64(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(leaf) })
}
fn new_leaf_u32(v: Vec<u32>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U32(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::UInt32, layout: Layout::Leaf(leaf) })
}
fn new_leaf_u64(v: Vec<u64>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U64(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::UInt64, layout: Layout::Leaf(leaf) })
}
fn new_leaf_f32(v: Vec<f32>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F32(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::Float32, layout: Layout::Leaf(leaf) })
}
fn new_leaf_f64(v: Vec<f64>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F64(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::Float64, layout: Layout::Leaf(leaf) })
}
fn new_leaf_bool(v: Vec<u8>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Bool);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::Bool(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::Bool, layout: Layout::Leaf(leaf) })
}
fn new_leaf_char(v: Vec<u32>) -> PyResult<GrumpyArray> {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Char);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::Char(Arc::new(v));
    Ok(GrumpyArray { dtype: DType::Char, layout: Layout::Leaf(leaf) })
}

// -------- scalar-bits path for ints/bool/char --------

fn collect_scalar_bits(dt: DType, leaf: &Leaf) -> PyResult<Vec<u64>> {
    let mut out: Vec<u64> = Vec::new();
    out.reserve(leaf.len);
    match (dt, &leaf.buffer) {
        (DType::Int32, LeafBuffer::I32(v)) => {
            for i in 0..leaf.len { if leaf.validity[i] { out.push((v[i] as i64) as u64); } }
        }
        (DType::Int64, LeafBuffer::I64(v)) => {
            for i in 0..leaf.len { if leaf.validity[i] { out.push(v[i] as u64); } }
        }
        (DType::UInt32, LeafBuffer::U32(v)) => {
            for i in 0..leaf.len { if leaf.validity[i] { out.push(v[i] as u64); } }
        }
        (DType::UInt64, LeafBuffer::U64(v)) => {
            for i in 0..leaf.len { if leaf.validity[i] { out.push(v[i]); } }
        }
        (DType::Bool, LeafBuffer::Bool(v)) => {
            for i in 0..leaf.len { if leaf.validity[i] { out.push((v[i] != 0) as u64); } }
        }
        (DType::Char, LeafBuffer::Char(v)) => {
            for i in 0..leaf.len { if leaf.validity[i] { out.push(v[i] as u64); } }
        }
        _ => return Err(PyValueError::new_err("Unsupported dtype for scalar_bits.")),
    }
    Ok(out)
}

fn unique_from_scalar_bits(_py: Python<'_>, dt: DType, bits: &[u64]) -> PyResult<GrumpyArray> {
    let mut v: Vec<u64> = bits.to_vec();
    v.sort();
    v.dedup();
    match dt {
        DType::Int32 => new_leaf_i32(v.into_iter().map(|x| x as i32).collect()),
        DType::Int64 => new_leaf_i64(v.into_iter().map(|x| x as i64).collect()),
        DType::UInt32 => new_leaf_u32(v.into_iter().map(|x| x as u32).collect()),
        DType::UInt64 => new_leaf_u64(v),
        DType::Bool => new_leaf_bool(v.into_iter().map(|x| (x != 0) as u8).collect()),
        DType::Char => new_leaf_char(v.into_iter().map(|x| x as u32).collect()),
        _ => Err(PyValueError::new_err("Unsupported dtype for unique_from_scalar_bits.")),
    }
}

// -------- float setdiff/setxor --------

fn setdiff_f64(py: Python<'_>, a: &Leaf, b: &Leaf) -> PyResult<GrumpyArray> {
    let mut av = collect_f64(a);
    let mut bv = collect_f64(b);
    av.sort_by(|x, y| x.total_cmp(y));
    bv.sort_by(|x, y| x.total_cmp(y));
    let mut out: Vec<f64> = Vec::new();
    let mut j = 0usize;
    for &x in &av {
        if x.is_nan() {
            out.push(x);
            continue;
        }
        while j < bv.len() && (bv[j].is_nan() || bv[j].total_cmp(&x).is_lt()) {
            if bv[j].is_nan() {
                j += 1;
                continue;
            }
            j += 1;
        }
        if j == bv.len() || bv[j].is_nan() || bv[j] != x {
            out.push(x);
        }
    }
    unique_f64_from_values(py, DType::Float64, &out)
}

fn setdiff_f32(py: Python<'_>, a: &Leaf, b: &Leaf) -> PyResult<GrumpyArray> {
    let mut av = collect_f32(a);
    let mut bv = collect_f32(b);
    av.sort_by(|x, y| x.total_cmp(y));
    bv.sort_by(|x, y| x.total_cmp(y));
    let mut out: Vec<f32> = Vec::new();
    let mut j = 0usize;
    for &x in &av {
        if x.is_nan() {
            out.push(x);
            continue;
        }
        while j < bv.len() && (bv[j].is_nan() || bv[j].total_cmp(&x).is_lt()) {
            if bv[j].is_nan() {
                j += 1;
                continue;
            }
            j += 1;
        }
        if j == bv.len() || bv[j].is_nan() || bv[j] != x {
            out.push(x);
        }
    }
    unique_f32_from_values(py, DType::Float32, &out)
}

fn setxor_f64(_py: Python<'_>, a: &Leaf, b: &Leaf) -> PyResult<GrumpyArray> {
    let mut av = collect_f64(a);
    let mut bv = collect_f64(b);
    av.sort_by(|x, y| x.total_cmp(y));
    bv.sort_by(|x, y| x.total_cmp(y));
    let mut out: Vec<f64> = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < av.len() || j < bv.len() {
        if j == bv.len() {
            out.push(av[i]);
            i += 1;
            continue;
        }
        if i == av.len() {
            out.push(bv[j]);
            j += 1;
            continue;
        }
        let x = av[i];
        let y = bv[j];
        // NaNs never match -> emit both sides
        if x.is_nan() && y.is_nan() {
            out.push(x);
            out.push(y);
            i += 1;
            j += 1;
            continue;
        }
        if x.is_nan() {
            out.push(x);
            i += 1;
            continue;
        }
        if y.is_nan() {
            out.push(y);
            j += 1;
            continue;
        }
        if x.total_cmp(&y).is_lt() {
            out.push(x);
            i += 1;
        } else if y.total_cmp(&x).is_lt() {
            out.push(y);
            j += 1;
        } else {
            // equal (non-NaN): skip both
            i += 1;
            j += 1;
        }
    }
    // IMPORTANT: do NOT collapse NaNs here; NumPy setxor keeps both NaNs if both inputs contain NaN.
    // But we still need to unique non-NaNs; easiest is: stable scan collapsing only when x==last and neither is NaN.
    out.sort_by(|x, y| x.total_cmp(y));
    let mut final_out: Vec<f64> = Vec::new();
    for x in out {
        if final_out.is_empty() { final_out.push(x); continue; }
        let last = *final_out.last().unwrap();
        if !x.is_nan() && !last.is_nan() && x == last { continue; }
        final_out.push(x);
    }
    new_leaf_f64(final_out)
}

fn setxor_f32(_py: Python<'_>, a: &Leaf, b: &Leaf) -> PyResult<GrumpyArray> {
    let mut av = collect_f32(a);
    let mut bv = collect_f32(b);
    av.sort_by(|x, y| x.total_cmp(y));
    bv.sort_by(|x, y| x.total_cmp(y));
    let mut out: Vec<f32> = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < av.len() || j < bv.len() {
        if j == bv.len() { out.push(av[i]); i += 1; continue; }
        if i == av.len() { out.push(bv[j]); j += 1; continue; }
        let x = av[i];
        let y = bv[j];
        if x.is_nan() && y.is_nan() {
            out.push(x);
            out.push(y);
            i += 1;
            j += 1;
            continue;
        }
        if x.is_nan() { out.push(x); i += 1; continue; }
        if y.is_nan() { out.push(y); j += 1; continue; }
        if x.total_cmp(&y).is_lt() { out.push(x); i += 1; }
        else if y.total_cmp(&x).is_lt() { out.push(y); j += 1; }
        else { i += 1; j += 1; }
    }
    out.sort_by(|x, y| x.total_cmp(y));
    let mut final_out: Vec<f32> = Vec::new();
    for x in out {
        if final_out.is_empty() { final_out.push(x); continue; }
        let last = *final_out.last().unwrap();
        if !x.is_nan() && !last.is_nan() && x == last { continue; }
        final_out.push(x);
    }
    new_leaf_f32(final_out)
}

// -------- isin implementation --------

enum MembershipSet {
    I32(FxHashSet<i32>),
    I64(FxHashSet<i64>),
    U32(FxHashSet<u32>),
    U64(FxHashSet<u64>),
    Bool(FxHashSet<u8>),
    Char(FxHashSet<u32>),
    String(FxHashSet<String>),
    F32(FxHashSet<u32>), // bits, with 0 normalized; NaNs excluded
    F64(FxHashSet<u64>), // bits, with 0 normalized; NaNs excluded
}

fn build_membership_set(dt: DType, leaf: &Leaf) -> PyResult<MembershipSet> {
    match (dt, &leaf.buffer) {
        (DType::Int32, LeafBuffer::I32(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            if leaf.has_nulls {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        s.insert(v[i]);
                    }
                }
            } else {
                s.extend(v.iter().copied());
            }
            Ok(MembershipSet::I32(s))
        }
        (DType::Int64, LeafBuffer::I64(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            if leaf.has_nulls {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        s.insert(v[i]);
                    }
                }
            } else {
                s.extend(v.iter().copied());
            }
            Ok(MembershipSet::I64(s))
        }
        (DType::UInt32, LeafBuffer::U32(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            if leaf.has_nulls {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        s.insert(v[i]);
                    }
                }
            } else {
                s.extend(v.iter().copied());
            }
            Ok(MembershipSet::U32(s))
        }
        (DType::UInt64, LeafBuffer::U64(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            if leaf.has_nulls {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        s.insert(v[i]);
                    }
                }
            } else {
                s.extend(v.iter().copied());
            }
            Ok(MembershipSet::U64(s))
        }
        (DType::Bool, LeafBuffer::Bool(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            for i in 0..leaf.len {
                if leaf.validity[i] {
                    s.insert(if v[i] != 0 { 1u8 } else { 0u8 });
                }
            }
            Ok(MembershipSet::Bool(s))
        }
        (DType::Char, LeafBuffer::Char(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            if leaf.has_nulls {
                for i in 0..leaf.len {
                    if leaf.validity[i] {
                        s.insert(v[i]);
                    }
                }
            } else {
                s.extend(v.iter().copied());
            }
            Ok(MembershipSet::Char(s))
        }
        (DType::String, LeafBuffer::String(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            for i in 0..leaf.len {
                if leaf.validity[i] {
                    s.insert(v[i].clone());
                }
            }
            Ok(MembershipSet::String(s))
        }
        (DType::Float32, LeafBuffer::F32(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            for i in 0..leaf.len {
                if !leaf.validity[i] {
                    continue;
                }
                let x = v[i];
                if x.is_nan() {
                    continue;
                }
                let bits = if x == 0.0 { 0.0f32.to_bits() } else { x.to_bits() };
                s.insert(bits);
            }
            Ok(MembershipSet::F32(s))
        }
        (DType::Float64, LeafBuffer::F64(v)) => {
            let mut s = FxHashSet::default();
            s.reserve(leaf.len.min(131_072));
            for i in 0..leaf.len {
                if !leaf.validity[i] {
                    continue;
                }
                let x = v[i];
                if x.is_nan() {
                    continue;
                }
                let bits = if x == 0.0 { 0.0f64.to_bits() } else { x.to_bits() };
                s.insert(bits);
            }
            Ok(MembershipSet::F64(s))
        }
        _ => Err(PyValueError::new_err("isin() not implemented for this dtype.")),
    }
}

fn isin_layout(layout: &Layout, dt: DType, set: &MembershipSet) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => Ok(Layout::Leaf(isin_leaf(l, dt, set)?)),
        Layout::ListOffset(lo) => {
            let content = isin_layout(lo.content.as_ref(), dt, set)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            let content = isin_layout(v.content.as_ref(), dt, set)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = isin_layout(ix.content.as_ref(), dt, set)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(u) => {
            let scalars = isin_leaf(&u.scalars, dt, set)?;
            let list_content = isin_layout(u.lists.content.as_ref(), dt, set)?;
            Ok(Layout::UnionScalarList(crate::layout::UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars,
                lists: ListOffset {
                    offsets: u.lists.offsets.clone(),
                    content: Box::new(list_content),
                },
            }))
        }
    }
}

fn isin_leaf(leaf: &Leaf, dt: DType, set: &MembershipSet) -> PyResult<Leaf> {
    let n = leaf.len;
    let mut out = Leaf::new(DType::Bool);
    out.len = n;
    out.has_nulls = leaf.has_nulls;
    out.validity = leaf.validity.clone();
    out.buffer = LeafBuffer::Bool(Arc::new(vec![0u8; n]));
    let oo = match &mut out.buffer { LeafBuffer::Bool(v) => Arc::make_mut(v), _ => unreachable!() };
    match (dt, set, &leaf.buffer) {
        (DType::Int32, MembershipSet::I32(s), LeafBuffer::I32(v)) => {
            if !leaf.has_nulls {
                oo.par_iter_mut()
                    .zip(v.par_iter())
                    .for_each(|(o, &x)| *o = s.contains(&x) as u8);
            } else {
                for i in 0..n {
                    if leaf.validity[i] {
                        oo[i] = s.contains(&v[i]) as u8;
                    }
                }
            }
        }
        (DType::Int64, MembershipSet::I64(s), LeafBuffer::I64(v)) => {
            for i in 0..n { if leaf.validity[i] { oo[i] = s.contains(&v[i]) as u8; } }
        }
        (DType::UInt32, MembershipSet::U32(s), LeafBuffer::U32(v)) => {
            for i in 0..n { if leaf.validity[i] { oo[i] = s.contains(&v[i]) as u8; } }
        }
        (DType::UInt64, MembershipSet::U64(s), LeafBuffer::U64(v)) => {
            for i in 0..n { if leaf.validity[i] { oo[i] = s.contains(&v[i]) as u8; } }
        }
        (DType::Bool, MembershipSet::Bool(s), LeafBuffer::Bool(v)) => {
            for i in 0..n { if leaf.validity[i] { oo[i] = s.contains(&(if v[i] != 0 { 1 } else { 0 })) as u8; } }
        }
        (DType::Char, MembershipSet::Char(s), LeafBuffer::Char(v)) => {
            for i in 0..n { if leaf.validity[i] { oo[i] = s.contains(&v[i]) as u8; } }
        }
        (DType::String, MembershipSet::String(s), LeafBuffer::String(v)) => {
            for i in 0..n {
                if leaf.validity[i] {
                    oo[i] = s.contains(&v[i]) as u8;
                }
            }
        }
        (DType::Float32, MembershipSet::F32(s), LeafBuffer::F32(v)) => {
            for i in 0..n {
                if !leaf.validity[i] { continue; }
                let x = v[i];
                if x.is_nan() { oo[i] = 0; continue; }
                let bits = if x == 0.0 { 0.0f32.to_bits() } else { x.to_bits() };
                oo[i] = s.contains(&bits) as u8;
            }
        }
        (DType::Float64, MembershipSet::F64(s), LeafBuffer::F64(v)) => {
            for i in 0..n {
                if !leaf.validity[i] { continue; }
                let x = v[i];
                if x.is_nan() { oo[i] = 0; continue; }
                let bits = if x == 0.0 { 0.0f64.to_bits() } else { x.to_bits() };
                oo[i] = s.contains(&bits) as u8;
            }
        }
        _ => return Err(PyValueError::new_err("Internal error: dtype mismatch in isin.")),
    }
    Ok(out)
}


