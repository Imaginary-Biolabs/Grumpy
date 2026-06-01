use crate::dtype::DType;
use crate::layout::{drop_axis0_select_element, layout_ndim, GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatOp {
    Var,
    Std,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantileMode {
    Quantile,   // q in [0,1]
    Percentile, // q in [0,100]
}

pub fn var(py: Python<'_>, a: &GrumpyArray, dim: Option<isize>, ddof: isize, nan: bool) -> PyResult<GrumpyArray> {
    stat_reduce(py, a, dim, ddof, nan, StatOp::Var)
}

pub fn std(py: Python<'_>, a: &GrumpyArray, dim: Option<isize>, ddof: isize, nan: bool) -> PyResult<GrumpyArray> {
    stat_reduce(py, a, dim, ddof, nan, StatOp::Std)
}

pub fn median(py: Python<'_>, a: &GrumpyArray, dim: isize, nan: bool) -> PyResult<GrumpyArray> {
    quantile_impl(py, a, dim, vec![0.5], QuantileMode::Quantile, nan)
}

pub fn quantile(
    py: Python<'_>,
    a: &GrumpyArray,
    dim: isize,
    q: Vec<f64>,
    mode: QuantileMode,
    nan: bool,
) -> PyResult<GrumpyArray> {
    quantile_impl(py, a, dim, q, mode, nan)
}

fn out_dtype_for_var_std(in_dt: DType) -> PyResult<DType> {
    Ok(match in_dt {
        DType::Float32 => DType::Float32,
        DType::Float64 => DType::Float64,
        DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64 => DType::Float64,
        _ => return Err(PyValueError::new_err("std/var only supported for numeric dtypes.")),
    })
}

fn stat_reduce(_py: Python<'_>, a: &GrumpyArray, dim: Option<isize>, ddof: isize, nan: bool, op: StatOp) -> PyResult<GrumpyArray> {
    if a.layout.has_union() {
        return stat_reduce_union(a, dim, ddof, nan, op);
    }
    if ddof < 0 {
        return Err(PyValueError::new_err("ddof must be >= 0."));
    }
    let dim = dim.ok_or_else(|| PyValueError::new_err("std/var on this layout requires an explicit dim."))?;
    let out_dt = out_dtype_for_var_std(a.dtype)?;
    let depth = crate::layout::list_chain_depth(&a.layout).ok_or_else(|| PyValueError::new_err("Not a pure list chain."))?;
    if depth == 0 {
        // 1D leaf: produce scalar leaf length=1? For consistency with existing API, return a Python scalar is nicer,
        // but current plumbing uses GrumpyArray methods returning GrumpyArray. We'll return 1D leaf of len=1.
        let leaf = match &a.layout { Layout::Leaf(l) => l, _ => unreachable!() };
        let val = var_std_slice(leaf, a.dtype, out_dt, ddof as usize, nan, op)?;
        return scalar_leaf(out_dt, val);
    }
    if depth != 1 {
        return Err(PyValueError::new_err("std/var currently only supports 1D and 2D arrays."));
    }
    let dim_u = if dim < 0 { (1isize + dim + 1) as usize } else { dim as usize };
    let lo = match &a.layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Expected list layout.")),
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf content.")),
    };

    match dim_u {
        1 => {
            if let Some(out) = stat_rect2d_dim1_fast(&a.layout, leaf, a.dtype, out_dt, ddof as usize, nan, op)? {
                Ok(out)
            } else {
                stat_dim1(lo, leaf, a.dtype, out_dt, ddof as usize, nan, op)
            }
        }
        0 => stat_dim0(lo, leaf, a.dtype, out_dt, ddof as usize, nan, op),
        _ => Err(PyValueError::new_err("Invalid dim.")),
    }
}

fn rect2d_shape(layout: &Layout) -> Option<(usize, usize, &[i64])> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return None,
    };
    let nrows = lo.len();
    if nrows == 0 {
        return Some((0, 0, lo.offsets.as_slice()));
    }
    let first = (lo.offsets[1] - lo.offsets[0]) as usize;
    for i in 0..nrows {
        let len_i = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
        if len_i != first {
            return None;
        }
    }
    Some((nrows, first, lo.offsets.as_slice()))
}

fn stat_rect2d_dim1_fast(
    layout: &Layout,
    leaf: &Leaf,
    in_dt: DType,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<Option<GrumpyArray>> {
    let (nrows, ncols, _off) = match rect2d_shape(layout) {
        Some(x) => x,
        None => return Ok(None),
    };
    if leaf.has_nulls {
        return Ok(None);
    }
    // If nan-variant and integer input, nan flag doesn't matter.
    if ncols == 0 {
        // all rows are empty => all null
        let mut out = Leaf::new(out_dt);
        out.len = nrows;
        out.has_nulls = true;
        out.validity = Arc::new(bitvec![u8, Lsb0; 0; nrows]);
        out.buffer = match out_dt {
            DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; nrows])),
            DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; nrows])),
            _ => return Err(PyValueError::new_err("Internal error: out dtype.")),
        };
        return Ok(Some(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) }));
    }

    let mut out = Leaf::new(out_dt);
    out.len = nrows;
    out.has_nulls = false;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; nrows]);
    out.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; nrows])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; nrows])),
        _ => return Err(PyValueError::new_err("Internal error: out dtype.")),
    };
    let out_valid = Arc::make_mut(&mut out.validity);

    match (in_dt, out_dt, nan, op, &leaf.buffer, &mut out.buffer) {
        (DType::Float64, DType::Float64, false, _, LeafBuffer::F64(v), LeafBuffer::F64(o)) => {
            let o = Arc::make_mut(o);
            for r in 0..nrows {
                let base = r * ncols;
                let (ok, vv) = sumsq_var_std_f64(&v[base..base + ncols], ddof, op);
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(r, false);
                } else {
                    o[r] = vv;
                }
            }
        }
        (DType::Float64, DType::Float64, true, _, LeafBuffer::F64(v), LeafBuffer::F64(o)) => {
            let o = Arc::make_mut(o);
            for r in 0..nrows {
                let base = r * ncols;
                let (ok, vv) = welford_var_std_f64(&v[base..base + ncols], ddof, op, true);
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(r, false);
                } else {
                    o[r] = vv;
                }
            }
        }
        (DType::Float32, DType::Float32, false, _, LeafBuffer::F32(v), LeafBuffer::F32(o)) => {
            let o = Arc::make_mut(o);
            for r in 0..nrows {
                let base = r * ncols;
                let (ok, vv) = sumsq_var_std_f32(&v[base..base + ncols], ddof, op);
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(r, false);
                } else {
                    o[r] = vv;
                }
            }
        }
        (DType::Float32, DType::Float32, true, _, LeafBuffer::F32(v), LeafBuffer::F32(o)) => {
            let o = Arc::make_mut(o);
            for r in 0..nrows {
                let base = r * ncols;
                let (ok, vv) = welford_var_std_f32(&v[base..base + ncols], ddof, op, true);
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(r, false);
                } else {
                    o[r] = vv;
                }
            }
        }
        // ints -> float64
        (DType::Int32, DType::Float64, _, _, LeafBuffer::I32(v), LeafBuffer::F64(o)) => {
            let o = Arc::make_mut(o);
            for r in 0..nrows {
                let base = r * ncols;
                let (ok, vv) = welford_var_std_i32(&v[base..base + ncols], ddof, op);
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(r, false);
                } else {
                    o[r] = vv;
                }
            }
        }
        (DType::Int64, DType::Float64, _, _, LeafBuffer::I64(v), LeafBuffer::F64(o)) => {
            let o = Arc::make_mut(o);
            for r in 0..nrows {
                let base = r * ncols;
                let (ok, vv) = welford_var_std_i64(&v[base..base + ncols], ddof, op);
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(r, false);
                } else {
                    o[r] = vv;
                }
            }
        }
        _ => return Ok(None),
    }

    Ok(Some(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) }))
}

#[inline]
fn welford_var_std_f64(xs: &[f64], ddof: usize, op: StatOp, nan: bool) -> (bool, f64) {
    let mut n: usize = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    for &x in xs {
        if nan && x.is_nan() {
            continue;
        }
        n += 1;
        let delta = x - mean;
        mean += delta / (n as f64);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    if n == 0 || ddof >= n {
        return (false, 0.0);
    }
    let var = m2 / ((n - ddof) as f64);
    (true, if op == StatOp::Std { var.sqrt() } else { var })
}

#[inline]
fn sumsq_var_std_f64(xs: &[f64], ddof: usize, op: StatOp) -> (bool, f64) {
    let n = xs.len();
    if n == 0 || ddof >= n {
        return (false, 0.0);
    }
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &x in xs {
        sum += x;
        sumsq += x * x;
    }
    let nf = n as f64;
    let mut var = (sumsq - (sum * sum) / nf) / ((n - ddof) as f64);
    if var < 0.0 {
        var = 0.0;
    }
    (true, if op == StatOp::Std { var.sqrt() } else { var })
}

#[inline]
fn welford_var_std_f32(xs: &[f32], ddof: usize, op: StatOp, nan: bool) -> (bool, f32) {
    // Accumulate in f64 and cast back to match NumPy float32 output closely.
    let mut n: usize = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    for &x in xs {
        if nan && x.is_nan() {
            continue;
        }
        let xf = x as f64;
        n += 1;
        let delta = xf - mean;
        mean += delta / (n as f64);
        let delta2 = xf - mean;
        m2 += delta * delta2;
    }
    if n == 0 || ddof >= n {
        return (false, 0.0);
    }
    let var = m2 / ((n - ddof) as f64);
    let out = if op == StatOp::Std { var.sqrt() } else { var };
    (true, out as f32)
}

#[inline]
fn sumsq_var_std_f32(xs: &[f32], ddof: usize, op: StatOp) -> (bool, f32) {
    let n = xs.len();
    if n == 0 || ddof >= n {
        return (false, 0.0);
    }
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &x in xs {
        let xf = x as f64;
        sum += xf;
        sumsq += xf * xf;
    }
    let nf = n as f64;
    let mut var = (sumsq - (sum * sum) / nf) / ((n - ddof) as f64);
    if var < 0.0 {
        var = 0.0;
    }
    let out = if op == StatOp::Std { var.sqrt() } else { var };
    (true, out as f32)
}

#[inline]
fn welford_var_std_i32(xs: &[i32], ddof: usize, op: StatOp) -> (bool, f64) {
    let mut n: usize = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    for &x in xs {
        let xf = x as f64;
        n += 1;
        let delta = xf - mean;
        mean += delta / (n as f64);
        let delta2 = xf - mean;
        m2 += delta * delta2;
    }
    if n == 0 || ddof >= n {
        return (false, 0.0);
    }
    let var = m2 / ((n - ddof) as f64);
    (true, if op == StatOp::Std { var.sqrt() } else { var })
}

#[inline]
fn welford_var_std_i64(xs: &[i64], ddof: usize, op: StatOp) -> (bool, f64) {
    let mut n: usize = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    for &x in xs {
        let xf = x as f64;
        n += 1;
        let delta = xf - mean;
        mean += delta / (n as f64);
        let delta2 = xf - mean;
        m2 += delta * delta2;
    }
    if n == 0 || ddof >= n {
        return (false, 0.0);
    }
    let var = m2 / ((n - ddof) as f64);
    (true, if op == StatOp::Std { var.sqrt() } else { var })
}

fn scalar_leaf(out_dt: DType, val: f64) -> PyResult<GrumpyArray> {
    let mut leaf = Leaf::new(out_dt);
    leaf.len = 1;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; 1]);
    leaf.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![val as f32])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![val])),
        _ => return Err(PyValueError::new_err("Internal error: scalar_leaf dtype.")),
    };
    Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(leaf) })
}

fn stat_dim1(
    lo: &ListOffset,
    leaf: &Leaf,
    in_dt: DType,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<GrumpyArray> {
    let nrows = lo.len();
    let mut out = Leaf::new(out_dt);
    out.len = nrows;
    out.has_nulls = false;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; nrows]);
    out.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; nrows])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; nrows])),
        _ => return Err(PyValueError::new_err("Internal error: out dtype.")),
    };
    let out_valid = Arc::make_mut(&mut out.validity);

    for i in 0..nrows {
        let s = lo.offsets[i] as usize;
        let e = lo.offsets[i + 1] as usize;
        let (ok, v) = var_std_range(leaf, in_dt, out_dt, s, e, ddof, nan, op)?;
        if !ok {
            out.has_nulls = true;
            out_valid.set(i, false);
        } else {
            match &mut out.buffer {
                LeafBuffer::F32(buf) => Arc::make_mut(buf)[i] = v as f32,
                LeafBuffer::F64(buf) => Arc::make_mut(buf)[i] = v,
                _ => unreachable!(),
            }
        }
    }
    Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) })
}

fn stat_dim0(
    lo: &ListOffset,
    leaf: &Leaf,
    in_dt: DType,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<GrumpyArray> {
    // Like our dim=0 reductions: produce length=maxlen, but require all rows have a valid (and for nan: non-NaN) value at each position.
    let nrows = lo.len();
    let mut maxlen: usize = 0;
    for i in 0..nrows {
        let len = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
        maxlen = maxlen.max(len);
    }
    let mut out = Leaf::new(out_dt);
    out.len = maxlen;
    out.has_nulls = false;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; maxlen]);
    out.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; maxlen])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; maxlen])),
        _ => return Err(PyValueError::new_err("Internal error: out dtype.")),
    };
    let out_valid = Arc::make_mut(&mut out.validity);

    for j in 0..maxlen {
        // Gather the j-th element from each row; if any missing -> null.
        let mut tmp: Vec<f64> = Vec::with_capacity(nrows);
        let mut ok = true;
        for i in 0..nrows {
            let s = lo.offsets[i] as usize;
            let e = lo.offsets[i + 1] as usize;
            if s + j >= e {
                ok = false;
                break;
            }
            let ix = s + j;
            if !leaf.validity[ix] {
                ok = false;
                break;
            }
            let x = scalar_as_f64(leaf, in_dt, ix)?;
            if nan && x.is_nan() {
                ok = false;
                break;
            }
            tmp.push(x);
        }
        if !ok {
            out.has_nulls = true;
            out_valid.set(j, false);
            continue;
        }
        let (ok2, v) = var_std_values(&tmp, ddof, op)?;
        if !ok2 {
            out.has_nulls = true;
            out_valid.set(j, false);
            continue;
        }
        match &mut out.buffer {
            LeafBuffer::F32(buf) => Arc::make_mut(buf)[j] = v as f32,
            LeafBuffer::F64(buf) => Arc::make_mut(buf)[j] = v,
            _ => unreachable!(),
        }
    }
    Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) })
}

fn scalar_as_f64(leaf: &Leaf, dt: DType, ix: usize) -> PyResult<f64> {
    Ok(match (dt, &leaf.buffer) {
        (DType::Float32, LeafBuffer::F32(v)) => v[ix] as f64,
        (DType::Float64, LeafBuffer::F64(v)) => v[ix],
        (DType::Int32, LeafBuffer::I32(v)) => v[ix] as f64,
        (DType::Int64, LeafBuffer::I64(v)) => v[ix] as f64,
        (DType::UInt32, LeafBuffer::U32(v)) => v[ix] as f64,
        (DType::UInt64, LeafBuffer::U64(v)) => v[ix] as f64,
        (DType::Int16, LeafBuffer::I16(v)) => v[ix] as f64,
        (DType::Int8, LeafBuffer::I8(v)) => v[ix] as f64,
        (DType::UInt16, LeafBuffer::U16(v)) => v[ix] as f64,
        (DType::UInt8, LeafBuffer::U8(v)) => v[ix] as f64,
        _ => return Err(PyValueError::new_err("Unsupported dtype for std/var.")),
    })
}

fn var_std_slice(leaf: &Leaf, in_dt: DType, out_dt: DType, ddof: usize, nan: bool, op: StatOp) -> PyResult<f64> {
    let (ok, v) = var_std_range(leaf, in_dt, out_dt, 0, leaf.len, ddof, nan, op)?;
    if !ok {
        return Err(PyValueError::new_err("std/var on empty slice."));
    }
    Ok(v)
}

fn var_std_range(
    leaf: &Leaf,
    in_dt: DType,
    _out_dt: DType,
    start: usize,
    end: usize,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<(bool, f64)> {
    let mut vals: Vec<f64> = Vec::new();
    vals.reserve(end - start);
    for ix in start..end {
        if !leaf.validity[ix] {
            continue;
        }
        let x = scalar_as_f64(leaf, in_dt, ix)?;
        if nan && x.is_nan() {
            continue;
        }
        vals.push(x);
    }
    var_std_values(&vals, ddof, op)
}

fn var_std_values(vals: &[f64], ddof: usize, op: StatOp) -> PyResult<(bool, f64)> {
    let n = vals.len();
    if n == 0 || ddof >= n {
        return Ok((false, 0.0));
    }
    let mean = vals.iter().sum::<f64>() / (n as f64);
    let mut acc = 0.0f64;
    for &x in vals {
        let d = x - mean;
        acc += d * d;
    }
    let var = acc / ((n - ddof) as f64);
    let out = match op {
        StatOp::Var => var,
        StatOp::Std => var.sqrt(),
    };
    Ok((true, out))
}

fn quantile_impl(
    _py: Python<'_>,
    a: &GrumpyArray,
    dim: isize,
    q: Vec<f64>,
    mode: QuantileMode,
    nan: bool,
) -> PyResult<GrumpyArray> {
    if a.layout.has_union() {
        return Err(PyValueError::new_err("quantile on union layouts is not implemented yet."));
    }
    // Only float/int for now, output float64 for ints (match NumPy default).
    let out_dt = match a.dtype {
        DType::Float32 => DType::Float32,
        DType::Float64 => DType::Float64,
        DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64 => DType::Float64,
        _ => return Err(PyValueError::new_err("quantile/percentile only supported for numeric dtypes.")),
    };

    // Normalize q to [0,1]
    let mut qs: Vec<f64> = Vec::with_capacity(q.len());
    for &x in &q {
        let qq = match mode {
            QuantileMode::Quantile => x,
            QuantileMode::Percentile => x / 100.0,
        };
        if !(0.0..=1.0).contains(&qq) {
            return Err(PyValueError::new_err("q out of range."));
        }
        qs.push(qq);
    }
    // For now: require a single q; we can extend to multi-q by adding another outer axis.
    if qs.len() != 1 {
        return Err(PyValueError::new_err("Only scalar q is supported for now."));
    }
    let q0 = qs[0];

    let depth = crate::layout::list_chain_depth(&a.layout).ok_or_else(|| PyValueError::new_err("Not a pure list chain."))?;
    if depth == 0 {
        let leaf = match &a.layout { Layout::Leaf(l) => l, _ => unreachable!() };
        let (ok, v) = quantile_range(leaf, a.dtype, 0, leaf.len, q0, nan)?;
        if !ok {
            return Err(PyValueError::new_err("quantile on empty slice."));
        }
        return scalar_leaf_q(out_dt, v);
    }
    if depth != 1 {
        return Err(PyValueError::new_err("quantile currently only supports 1D and 2D arrays."));
    }
    let dim_u = if dim < 0 { (1isize + dim + 1) as usize } else { dim as usize };
    let lo = match &a.layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Expected list layout.")),
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf content.")),
    };
    match dim_u {
        1 => {
            let nrows = lo.len();
            let mut out = Leaf::new(out_dt);
            out.len = nrows;
            out.has_nulls = false;
            out.validity = Arc::new(bitvec![u8, Lsb0; 1; nrows]);
            out.buffer = match out_dt {
                DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; nrows])),
                DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; nrows])),
                _ => unreachable!(),
            };
            let out_valid = Arc::make_mut(&mut out.validity);
            for i in 0..nrows {
                let s = lo.offsets[i] as usize;
                let e = lo.offsets[i + 1] as usize;
                let (ok, v) = quantile_range(leaf, a.dtype, s, e, q0, nan)?;
                if !ok {
                    out.has_nulls = true;
                    out_valid.set(i, false);
                } else {
                    match &mut out.buffer {
                        LeafBuffer::F32(buf) => Arc::make_mut(buf)[i] = v as f32,
                        LeafBuffer::F64(buf) => Arc::make_mut(buf)[i] = v,
                        _ => unreachable!(),
                    }
                }
            }
            Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(out) })
        }
        _ => Err(PyValueError::new_err("quantile dim=0 not implemented yet.")),
    }
}

fn scalar_leaf_q(out_dt: DType, val: f64) -> PyResult<GrumpyArray> {
    let mut leaf = Leaf::new(out_dt);
    leaf.len = 1;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; 1]);
    leaf.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![val as f32])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![val])),
        _ => return Err(PyValueError::new_err("Internal error.")),
    };
    Ok(GrumpyArray { dtype: out_dt, layout: Layout::Leaf(leaf) })
}

fn quantile_range(leaf: &Leaf, in_dt: DType, start: usize, end: usize, q: f64, nan: bool) -> PyResult<(bool, f64)> {
    let mut vals: Vec<f64> = Vec::new();
    vals.reserve(end - start);
    for ix in start..end {
        if !leaf.validity[ix] { continue; }
        let x = scalar_as_f64(leaf, in_dt, ix)?;
        if nan && x.is_nan() { continue; }
        vals.push(x);
    }
    if vals.is_empty() {
        return Ok((false, 0.0));
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    // NumPy default method='linear' (Hyndman-Fan type 7): (n-1)*q
    let n = vals.len();
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        return Ok((true, vals[lo]));
    }
    let w = h - lo as f64;
    Ok((true, vals[lo] * (1.0 - w) + vals[hi] * w))
}

fn normalize_stat_dim(dim: isize, ndim: usize) -> PyResult<usize> {
    let mut d = dim;
    if d < 0 {
        d += ndim as isize;
    }
    if d < 0 || d as usize >= ndim {
        return Err(PyValueError::new_err("dim out of range."));
    }
    Ok(d as usize)
}

fn collect_layout_f64(layout: &Layout, dt: DType, nan: bool) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    collect_layout_f64_rec(layout, dt, nan, &mut out)?;
    Ok(out)
}

fn collect_layout_f64_rec(
    layout: &Layout,
    dt: DType,
    nan: bool,
    out: &mut Vec<f64>,
) -> PyResult<()> {
    match layout {
        Layout::Leaf(l) => {
            for i in 0..l.len {
                if !l.validity[i] {
                    continue;
                }
                let x = scalar_as_f64(l, dt, i)?;
                if nan && x.is_nan() {
                    continue;
                }
                out.push(x);
            }
        }
        Layout::ListOffset(lo) => {
            for i in 0..lo.len() {
                let s = lo.offsets[i] as usize;
                let e = lo.offsets[i + 1] as usize;
                collect_layout_f64_rec(
                    &crate::layout::take_range(lo.content.as_ref(), s, e)?,
                    dt,
                    nan,
                    out,
                )?;
            }
        }
        Layout::OffsetView(v) => {
            for i in 0..layout.len() {
                collect_layout_f64_rec(&drop_axis0_select_element(layout, i)?, dt, nan, out)?;
            }
        }
        Layout::Indexed(ix) => {
            for i in 0..ix.len() {
                collect_layout_f64_rec(&drop_axis0_select_element(layout, i)?, dt, nan, out)?;
            }
        }
        Layout::UnionScalarList(u) => {
            for i in 0..u.len() {
                collect_layout_f64_rec(&drop_axis0_select_element(layout, i)?, dt, nan, out)?;
            }
        }
    }
    Ok(())
}

fn stat_reduce_union(a: &GrumpyArray, dim: Option<isize>, ddof: isize, nan: bool, op: StatOp) -> PyResult<GrumpyArray> {
    if ddof < 0 {
        return Err(PyValueError::new_err("ddof must be >= 0."));
    }
    let out_dt = out_dtype_for_var_std(a.dtype)?;
    if dim.is_none() {
        let vals = collect_layout_f64(&a.layout, a.dtype, nan)?;
        let (ok, v) = var_std_values(&vals, ddof as usize, op)?;
        if !ok {
            return Err(PyValueError::new_err("std/var on empty array."));
        }
        return scalar_leaf(out_dt, v);
    }
    let dim = dim.unwrap();
    let ndim = layout_ndim(&a.layout)?;
    let dim_u = normalize_stat_dim(dim, ndim)?;
    if ndim == 1 {
        let vals = collect_layout_f64(&a.layout, a.dtype, nan)?;
        let (ok, v) = var_std_values(&vals, ddof as usize, op)?;
        if !ok {
            return Err(PyValueError::new_err("std/var on empty array."));
        }
        return scalar_leaf(out_dt, v);
    }
    match dim_u {
        0 => stat_union_axis0(a, out_dt, ddof as usize, nan, op),
        x if x == ndim - 1 => stat_union_last_axis(a, out_dt, ddof as usize, nan, op),
        _ => Err(PyValueError::new_err(
            "std/var on union layouts currently supports dim=0 and innermost dim only.",
        )),
    }
}

fn stat_union_axis0(
    a: &GrumpyArray,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<GrumpyArray> {
    let n = a.len();
    let mut scalars = Leaf::new(out_dt);
    scalars.len = n;
    scalars.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    scalars.has_nulls = false;
    scalars.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        _ => unreachable!(),
    };
    let out_valid = Arc::make_mut(&mut scalars.validity);
    for i in 0..n {
        let sub = drop_axis0_select_element(&a.layout, i)?;
        let vals = collect_layout_f64(&sub, a.dtype, nan)?;
        let (ok, v) = var_std_values(&vals, ddof, op)?;
        if !ok {
            out_valid.set(i, false);
            scalars.has_nulls = true;
            continue;
        }
        match &mut scalars.buffer {
            LeafBuffer::F32(buf) => Arc::make_mut(buf)[i] = v as f32,
            LeafBuffer::F64(buf) => Arc::make_mut(buf)[i] = v,
            _ => unreachable!(),
        }
    }
    let lists = ListOffset {
        offsets: Arc::new(vec![0i64]),
        content: Box::new(Layout::Leaf(Leaf::new(out_dt))),
    };
    Ok(GrumpyArray {
        dtype: out_dt,
        layout: Layout::UnionScalarList(UnionScalarList {
            tags: (0..n).map(|_| 0u8).collect(),
            index: (0..n as i64).collect(),
            scalars,
            lists,
        }),
    })
}

fn stat_union_last_axis(
    a: &GrumpyArray,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<GrumpyArray> {
    Ok(GrumpyArray {
        dtype: out_dt,
        layout: stat_last_layout(&a.layout, a.dtype, out_dt, ddof, nan, op)?,
    })
}

fn stat_last_layout(
    layout: &Layout,
    in_dt: DType,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => {
            if l.len <= 1 {
                return Ok(layout.clone());
            }
            let lo = ListOffset {
                offsets: Arc::new(vec![0i64, l.len as i64]),
                content: Box::new(Layout::Leaf(l.clone())),
            };
            Ok(Layout::ListOffset(stat_listoffset_leaf(
                &lo,
                l,
                in_dt,
                out_dt,
                ddof,
                nan,
                op,
            )?))
        }
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(leaf) => Ok(Layout::ListOffset(stat_listoffset_leaf(
                lo, leaf, in_dt, out_dt, ddof, nan, op,
            )?)),
            _ => Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(stat_last_layout(
                    lo.content.as_ref(),
                    in_dt,
                    out_dt,
                    ddof,
                    nan,
                    op,
                )?),
            })),
        },
        Layout::UnionScalarList(u) => {
            let list_content =
                stat_last_layout(u.lists.content.as_ref(), in_dt, out_dt, ddof, nan, op)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars: u.scalars.clone(),
                lists: ListOffset {
                    offsets: u.lists.offsets.clone(),
                    content: Box::new(list_content),
                },
            }))
        }
        Layout::OffsetView(v) => {
            let content = stat_last_layout(v.content.as_ref(), in_dt, out_dt, ddof, nan, op)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = stat_last_layout(ix.content.as_ref(), in_dt, out_dt, ddof, nan, op)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
    }
}

fn stat_listoffset_leaf(
    lo: &ListOffset,
    leaf: &Leaf,
    in_dt: DType,
    out_dt: DType,
    ddof: usize,
    nan: bool,
    op: StatOp,
) -> PyResult<ListOffset> {
    let nrows = lo.len();
    let mut out = Leaf::new(out_dt);
    out.len = nrows;
    out.has_nulls = false;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; nrows]);
    out.buffer = match out_dt {
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; nrows])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; nrows])),
        _ => unreachable!(),
    };
    let out_valid = Arc::make_mut(&mut out.validity);
    for i in 0..nrows {
        let s = lo.offsets[i] as usize;
        let e = lo.offsets[i + 1] as usize;
        let (ok, v) = var_std_range(leaf, in_dt, out_dt, s, e, ddof, nan, op)?;
        if !ok {
            out.has_nulls = true;
            out_valid.set(i, false);
            continue;
        }
        match &mut out.buffer {
            LeafBuffer::F32(buf) => Arc::make_mut(buf)[i] = v as f32,
            LeafBuffer::F64(buf) => Arc::make_mut(buf)[i] = v,
            _ => unreachable!(),
        }
    }
    Ok(ListOffset {
        offsets: lo.offsets.clone(),
        content: Box::new(Layout::Leaf(out)),
    })
}
