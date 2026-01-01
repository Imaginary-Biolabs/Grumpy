use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

pub fn bincount(
    _py: Python<'_>,
    x: &GrumpyArray,
    weights: Option<&GrumpyArray>,
    minlength: usize,
) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&x.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("bincount does not support nulls."));
    }
    let wleaf = if let Some(w) = weights {
        if w.dtype != DType::Float64 && w.dtype != DType::Float32 {
            return Err(PyValueError::new_err("bincount weights must be float32/float64."));
        }
        let wl = leaf_1d(&w.layout)?;
        if wl.has_nulls {
            return Err(PyValueError::new_err("bincount weights must not contain nulls."));
        }
        if wl.len != leaf.len {
            return Err(PyValueError::new_err("bincount weights length mismatch."));
        }
        Some((w.dtype, wl))
    } else {
        None
    };

    // Determine max value
    let mut maxv: i64 = -1;
    match (x.dtype, &leaf.buffer) {
        (DType::Int32, LeafBuffer::I32(v)) => {
            for &a in v.as_slice().iter().take(leaf.len) {
                if a < 0 {
                    return Err(PyValueError::new_err("bincount input must be non-negative."));
                }
                maxv = maxv.max(a as i64);
            }
        }
        (DType::Int64, LeafBuffer::I64(v)) => {
            for &a in v.as_slice().iter().take(leaf.len) {
                if a < 0 {
                    return Err(PyValueError::new_err("bincount input must be non-negative."));
                }
                maxv = maxv.max(a);
            }
        }
        (DType::UInt32, LeafBuffer::U32(v)) => {
            for &a in v.as_slice().iter().take(leaf.len) {
                maxv = maxv.max(a as i64);
            }
        }
        (DType::UInt64, LeafBuffer::U64(v)) => {
            for &a in v.as_slice().iter().take(leaf.len) {
                if a > i64::MAX as u64 {
                    return Err(PyValueError::new_err("bincount value too large."));
                }
                maxv = maxv.max(a as i64);
            }
        }
        _ => return Err(PyValueError::new_err("bincount only supports integer inputs.")),
    }

    let n = std::cmp::max(minlength, (maxv + 1).max(0) as usize);
    if wleaf.is_some() {
        let mut out = new_leaf_f64(n);
        let oo = match &mut out.buffer {
            LeafBuffer::F64(v) => Arc::make_mut(v),
            _ => unreachable!(),
        };
        match (x.dtype, &leaf.buffer) {
            (DType::Int32, LeafBuffer::I32(v)) => add_weights(v.as_slice(), leaf.len, wleaf.unwrap(), oo)?,
            (DType::Int64, LeafBuffer::I64(v)) => add_weights_i64(v.as_slice(), leaf.len, wleaf.unwrap(), oo)?,
            (DType::UInt32, LeafBuffer::U32(v)) => add_weights_u32(v.as_slice(), leaf.len, wleaf.unwrap(), oo)?,
            (DType::UInt64, LeafBuffer::U64(v)) => add_weights_u64(v.as_slice(), leaf.len, wleaf.unwrap(), oo)?,
            _ => unreachable!(),
        }
        Ok(GrumpyArray { dtype: DType::Float64, layout: Layout::Leaf(out) })
    } else {
        let mut out = new_leaf_i64(n);
        let oo = match &mut out.buffer {
            LeafBuffer::I64(v) => Arc::make_mut(v),
            _ => unreachable!(),
        };
        match (x.dtype, &leaf.buffer) {
            (DType::Int32, LeafBuffer::I32(v)) => {
                for &a in v.as_slice().iter().take(leaf.len) {
                    oo[a as usize] += 1;
                }
            }
            (DType::Int64, LeafBuffer::I64(v)) => {
                for &a in v.as_slice().iter().take(leaf.len) {
                    oo[a as usize] += 1;
                }
            }
            (DType::UInt32, LeafBuffer::U32(v)) => {
                for &a in v.as_slice().iter().take(leaf.len) {
                    oo[a as usize] += 1;
                }
            }
            (DType::UInt64, LeafBuffer::U64(v)) => {
                for &a in v.as_slice().iter().take(leaf.len) {
                    oo[a as usize] += 1;
                }
            }
            _ => unreachable!(),
        }
        Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(out) })
    }
}

pub fn digitize(_py: Python<'_>, x: &GrumpyArray, bins: &GrumpyArray, right: bool) -> PyResult<GrumpyArray> {
    let xl = leaf_1d(&x.layout)?;
    let bl = leaf_1d(&bins.layout)?;
    if xl.has_nulls || bl.has_nulls {
        return Err(PyValueError::new_err("digitize does not support nulls."));
    }
    // Support float64 only for now (covers common stats usage).
    if x.dtype != DType::Float64 || bins.dtype != DType::Float64 {
        return Err(PyValueError::new_err("digitize currently only supports float64 inputs."));
    }
    let xv = match &xl.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
    let bv = match &bl.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
    let nb = bl.len;
    // Verify monotonic increasing (like NumPy requirement).
    for i in 1..nb {
        if !(bv[i - 1] <= bv[i]) {
            return Err(PyValueError::new_err("bins must be monotonically increasing."));
        }
    }
    let mut out = new_leaf_i64(xl.len);
    let oo = match &mut out.buffer { LeafBuffer::I64(v) => Arc::make_mut(v), _ => unreachable!() };
    for i in 0..xl.len {
        let xx = xv[i];
        // NumPy returns nb for NaN (as observed).
        if xx.is_nan() {
            oo[i] = nb as i64;
            continue;
        }
        let idx = if right {
            upper_bound(bv, nb, xx)
        } else {
            lower_bound(bv, nb, xx)
        };
        oo[i] = idx as i64;
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(out) })
}

pub fn histogram(
    _py: Python<'_>,
    x: &GrumpyArray,
    bins: usize,
    range: Option<(f64, f64)>,
    density: bool,
    weights: Option<&GrumpyArray>,
) -> PyResult<(GrumpyArray, GrumpyArray)> {
    let xl = leaf_1d(&x.layout)?;
    if xl.has_nulls {
        return Err(PyValueError::new_err("histogram does not support nulls."));
    }
    if bins == 0 {
        return Err(PyValueError::new_err("bins must be > 0."));
    }
    // float64 only for now.
    if x.dtype != DType::Float64 {
        return Err(PyValueError::new_err("histogram currently only supports float64 inputs."));
    }
    let xv = match &xl.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };

    let w = if let Some(w) = weights {
        if w.dtype != DType::Float64 {
            return Err(PyValueError::new_err("histogram weights must be float64 for now."));
        }
        let wl = leaf_1d(&w.layout)?;
        if wl.has_nulls || wl.len != xl.len {
            return Err(PyValueError::new_err("histogram weights must be same length and all-valid."));
        }
        Some(match &wl.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() })
    } else {
        None
    };

    // Determine range
    let (lo, hi) = if let Some(r) = range {
        r
    } else {
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for i in 0..xl.len {
            let x = xv[i];
            if x.is_nan() {
                continue;
            }
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
        }
        if !mn.is_finite() || !mx.is_finite() {
            return Err(PyValueError::new_err("histogram: could not infer range."));
        }
        (mn, mx)
    };
    if !(lo < hi) {
        return Err(PyValueError::new_err("histogram range must satisfy lo < hi."));
    }
    let width = (hi - lo) / (bins as f64);

    let mut counts = vec![0f64; bins];
    let mut total_w = 0f64;
    for i in 0..xl.len {
        let x = xv[i];
        if x.is_nan() {
            continue;
        }
        if x < lo || x > hi {
            continue;
        }
        let mut b: isize = if x == hi {
            (bins - 1) as isize
        } else {
            ((x - lo) / width).floor() as isize
        };
        if b < 0 {
            b = 0;
        }
        if b as usize >= bins {
            b = (bins - 1) as isize;
        }
        let ww = if let Some(wv) = w { wv[i] } else { 1.0 };
        counts[b as usize] += ww;
        total_w += ww;
    }
    if density {
        if total_w != 0.0 {
            for c in &mut counts {
                *c /= total_w * width;
            }
        }
    }

    let mut hist_leaf = new_leaf_f64(bins);
    if let LeafBuffer::F64(v) = &mut hist_leaf.buffer {
        *Arc::make_mut(v) = counts;
    }
    let hist = GrumpyArray { dtype: DType::Float64, layout: Layout::Leaf(hist_leaf) };

    // bin edges
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lo + (i as f64) * width);
    }
    let edges = GrumpyArray { dtype: DType::Float64, layout: Layout::Leaf(new_leaf_f64_from(edges)) };
    Ok((hist, edges))
}

// -------- internal helpers --------

fn leaf_1d<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::OffsetView(v) => leaf_1d(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_1d(ix.content.as_ref()),
        Layout::ListOffset(_) => Err(PyValueError::new_err("Expected 1D leaf array.")),
        Layout::UnionScalarList(_) => Err(PyValueError::new_err("Union not supported.")),
    }
}

fn new_leaf_i64(n: usize) -> Leaf {
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I64(Arc::new(vec![0i64; n]));
    leaf
}

fn new_leaf_f64(n: usize) -> Leaf {
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F64(Arc::new(vec![0f64; n]));
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

fn lower_bound(bins: &[f64], nb: usize, x: f64) -> usize {
    // first index i where bins[i] >= x, return i
    let mut lo = 0usize;
    let mut hi = nb;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if bins[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn upper_bound(bins: &[f64], nb: usize, x: f64) -> usize {
    // first index i where bins[i] > x
    let mut lo = 0usize;
    let mut hi = nb;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if bins[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn add_weights(v: &[i32], n: usize, w: (DType, &Leaf), out: &mut [f64]) -> PyResult<()> {
    let (wdt, wl) = w;
    match (wdt, &wl.buffer) {
        (DType::Float64, LeafBuffer::F64(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i];
            }
        }
        (DType::Float32, LeafBuffer::F32(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i] as f64;
            }
        }
        _ => return Err(PyValueError::new_err("Invalid weights dtype.")),
    }
    Ok(())
}
fn add_weights_i64(v: &[i64], n: usize, w: (DType, &Leaf), out: &mut [f64]) -> PyResult<()> {
    let (wdt, wl) = w;
    match (wdt, &wl.buffer) {
        (DType::Float64, LeafBuffer::F64(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i];
            }
        }
        (DType::Float32, LeafBuffer::F32(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i] as f64;
            }
        }
        _ => return Err(PyValueError::new_err("Invalid weights dtype.")),
    }
    Ok(())
}
fn add_weights_u32(v: &[u32], n: usize, w: (DType, &Leaf), out: &mut [f64]) -> PyResult<()> {
    let (wdt, wl) = w;
    match (wdt, &wl.buffer) {
        (DType::Float64, LeafBuffer::F64(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i];
            }
        }
        (DType::Float32, LeafBuffer::F32(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i] as f64;
            }
        }
        _ => return Err(PyValueError::new_err("Invalid weights dtype.")),
    }
    Ok(())
}
fn add_weights_u64(v: &[u64], n: usize, w: (DType, &Leaf), out: &mut [f64]) -> PyResult<()> {
    let (wdt, wl) = w;
    match (wdt, &wl.buffer) {
        (DType::Float64, LeafBuffer::F64(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i];
            }
        }
        (DType::Float32, LeafBuffer::F32(wv)) => {
            for i in 0..n {
                out[v[i] as usize] += wv[i] as f64;
            }
        }
        _ => return Err(PyValueError::new_err("Invalid weights dtype.")),
    }
    Ok(())
}


