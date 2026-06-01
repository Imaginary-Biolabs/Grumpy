use crate::dtype::DType;
use crate::error::{
    dtype_mismatch, dtype_unsupported, internal_dtype_buffer_mismatch, layout_unsupported,
    shape_mismatch, unsupported,
};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::prelude::*;
use std::sync::Arc;

pub fn dot(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<PyObject> {
    let la = leaf_1d(&a.layout)?;
    let lb = leaf_1d(&b.layout)?;
    if a.dtype != b.dtype {
        return Err(dtype_mismatch(a.dtype, b.dtype, "in dot"));
    }
    if la.has_nulls || lb.has_nulls {
        return Err(unsupported(
            "dot",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling dot.",
        ));
    }
    if la.len != lb.len {
        return Err(shape_mismatch(
            "dot",
            "requires equal-length 1D arrays",
            "ensure both operands are 1D leaf arrays with the same length.",
        ));
    }
    let n = la.len;
    match a.dtype {
        DType::Int32 => {
            let av = as_i32(la)?;
            let bv = as_i32(lb)?;
            let mut acc: i64 = 0;
            for i in 0..n {
                acc += (av[i] as i64) * (bv[i] as i64);
            }
            Ok(acc.into_py(py))
        }
        DType::Int64 => {
            let av = as_i64(la)?;
            let bv = as_i64(lb)?;
            let mut acc: i128 = 0;
            for i in 0..n {
                acc += (av[i] as i128) * (bv[i] as i128);
            }
            Ok((acc as i64).into_py(py))
        }
        DType::UInt32 => {
            let av = as_u32(la)?;
            let bv = as_u32(lb)?;
            let mut acc: u128 = 0;
            for i in 0..n {
                acc += (av[i] as u128) * (bv[i] as u128);
            }
            Ok((acc as u64).into_py(py))
        }
        DType::UInt64 => {
            let av = as_u64(la)?;
            let bv = as_u64(lb)?;
            let mut acc: u128 = 0;
            for i in 0..n {
                acc += (av[i] as u128) * (bv[i] as u128);
            }
            Ok((acc as u64).into_py(py))
        }
        DType::Float32 => {
            let av = as_f32(la)?;
            let bv = as_f32(lb)?;
            let mut acc: f64 = 0.0;
            for i in 0..n {
                acc += (av[i] as f64) * (bv[i] as f64);
            }
            Ok(acc.into_py(py))
        }
        DType::Float64 => {
            let av = as_f64(la)?;
            let bv = as_f64(lb)?;
            let mut acc: f64 = 0.0;
            for i in 0..n {
                acc += av[i] * bv[i];
            }
            Ok(acc.into_py(py))
        }
        _ => Err(dtype_unsupported("dot", a.dtype)),
    }
}

pub fn inner(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<PyObject> {
    dot(py, a, b)
}

pub fn norm(py: Python<'_>, a: &GrumpyArray) -> PyResult<PyObject> {
    let la = leaf_1d(&a.layout)?;
    if la.has_nulls {
        return Err(unsupported(
            "norm",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling norm.",
        ));
    }
    let n = la.len;
    let mut acc: f64 = 0.0;
    match a.dtype {
        DType::Int32 => {
            let av = as_i32(la)?;
            for i in 0..n {
                let x = av[i] as f64;
                acc += x * x;
            }
        }
        DType::Int64 => {
            let av = as_i64(la)?;
            for i in 0..n {
                let x = av[i] as f64;
                acc += x * x;
            }
        }
        DType::UInt32 => {
            let av = as_u32(la)?;
            for i in 0..n {
                let x = av[i] as f64;
                acc += x * x;
            }
        }
        DType::UInt64 => {
            let av = as_u64(la)?;
            for i in 0..n {
                let x = av[i] as f64;
                acc += x * x;
            }
        }
        DType::Float32 => {
            let av = as_f32(la)?;
            for i in 0..n {
                let x = av[i] as f64;
                acc += x * x;
            }
        }
        DType::Float64 => {
            let av = as_f64(la)?;
            for i in 0..n {
                let x = av[i];
                acc += x * x;
            }
        }
        _ => return Err(dtype_unsupported("norm", a.dtype)),
    }
    Ok(acc.sqrt().into_py(py))
}

pub fn trace(py: Python<'_>, a: &GrumpyArray) -> PyResult<PyObject> {
    if a.layout.has_union() {
        return Err(layout_unsupported("trace", "UnionScalarList layout is not supported"));
    }
    let (lo, leaf, nrows, ncols) = rectangular_2d_list_leaf(&a.layout)?;
    if leaf.has_nulls {
        return Err(unsupported(
            "trace",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling trace.",
        ));
    }
    let n = nrows.min(ncols);
    match a.dtype {
        DType::Int32 => {
            let v = as_i32(leaf)?;
            let mut acc: i64 = 0;
            for i in 0..n {
                acc += v[lo.offsets[i] as usize + i] as i64;
            }
            Ok(acc.into_py(py))
        }
        DType::Int64 => {
            let v = as_i64(leaf)?;
            let mut acc: i128 = 0;
            for i in 0..n {
                acc += v[lo.offsets[i] as usize + i] as i128;
            }
            Ok((acc as i64).into_py(py))
        }
        DType::Float64 => {
            let v = as_f64(leaf)?;
            let mut acc: f64 = 0.0;
            for i in 0..n {
                acc += v[lo.offsets[i] as usize + i];
            }
            Ok(acc.into_py(py))
        }
        DType::Float32 => {
            let v = as_f32(leaf)?;
            let mut acc: f64 = 0.0;
            for i in 0..n {
                acc += v[lo.offsets[i] as usize + i] as f64;
            }
            Ok(acc.into_py(py))
        }
        _ => Err(dtype_unsupported("trace", a.dtype)),
    }
}

pub fn outer(_py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<GrumpyArray> {
    let la = leaf_1d(&a.layout)?;
    let lb = leaf_1d(&b.layout)?;
    if a.dtype != b.dtype {
        return Err(dtype_mismatch(a.dtype, b.dtype, "in outer"));
    }
    if la.has_nulls || lb.has_nulls {
        return Err(unsupported(
            "outer",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling outer.",
        ));
    }
    let n = la.len;
    let m = lb.len;
    let offsets: Vec<i64> = (0..=n).map(|i| (i as i64) * (m as i64)).collect();
    let total = n * m;

    let content = match a.dtype {
        DType::Int32 => {
            let av = as_i32(la)?;
            let bv = as_i32(lb)?;
            let mut out = vec![0i32; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i].wrapping_mul(bv[j]);
                }
            }
            Layout::Leaf(new_leaf_i32(out))
        }
        DType::Int64 => {
            let av = as_i64(la)?;
            let bv = as_i64(lb)?;
            let mut out = vec![0i64; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i].wrapping_mul(bv[j]);
                }
            }
            Layout::Leaf(new_leaf_i64(out))
        }
        DType::UInt32 => {
            let av = as_u32(la)?;
            let bv = as_u32(lb)?;
            let mut out = vec![0u32; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i].wrapping_mul(bv[j]);
                }
            }
            Layout::Leaf(new_leaf_u32(out))
        }
        DType::UInt64 => {
            let av = as_u64(la)?;
            let bv = as_u64(lb)?;
            let mut out = vec![0u64; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i].wrapping_mul(bv[j]);
                }
            }
            Layout::Leaf(new_leaf_u64(out))
        }
        DType::Float32 => {
            let av = as_f32(la)?;
            let bv = as_f32(lb)?;
            let mut out = vec![0f32; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i] * bv[j];
                }
            }
            Layout::Leaf(new_leaf_f32(out))
        }
        DType::Float64 => {
            let av = as_f64(la)?;
            let bv = as_f64(lb)?;
            let mut out = vec![0f64; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i] * bv[j];
                }
            }
            Layout::Leaf(new_leaf_f64(out))
        }
        _ => return Err(dtype_unsupported("outer", a.dtype)),
    };

    Ok(GrumpyArray {
        dtype: a.dtype,
        layout: Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(content) }),
    })
}

pub fn cross(_py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<GrumpyArray> {
    if a.dtype != b.dtype {
        return Err(dtype_mismatch(a.dtype, b.dtype, "in cross"));
    }
    // Support 1D length-3 vectors and 2D list->leaf with row length 3.
    if let (Ok(la), Ok(lb)) = (leaf_1d(&a.layout), leaf_1d(&b.layout)) {
        if la.has_nulls || lb.has_nulls {
            return Err(unsupported(
                "cross",
                "null/missing values are not supported yet",
                "fill or drop nulls before calling cross.",
            ));
        }
        if la.len != 3 || lb.len != 3 {
            return Err(shape_mismatch(
                "cross",
                "1D cross product requires length-3 vectors",
                "pass two 1D arrays of length 3.",
            ));
        }
        return cross_vec3(a.dtype, la, lb);
    }

    let (alo, aleaf, anrows, ancols) = rectangular_2d_list_leaf(&a.layout)?;
    let (blo, bleaf, bnrows, bncols) = rectangular_2d_list_leaf(&b.layout)?;
    if anrows != bnrows || ancols != bncols || ancols != 3 {
        return Err(shape_mismatch(
            "cross",
            "2D cross requires matching shapes with last dimension 3",
            "ensure both operands are 2D with the same shape and inner dimension 3.",
        ));
    }
    if aleaf.has_nulls || bleaf.has_nulls {
        return Err(unsupported(
            "cross",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling cross.",
        ));
    }
    let nrows = anrows;

    let out_layout = match a.dtype {
        DType::Float64 => {
            let av = as_f64(aleaf)?;
            let bv = as_f64(bleaf)?;
            let mut out = vec![0f64; nrows * 3];
            for r in 0..nrows {
                let s = alo.offsets[r] as usize;
                let t = blo.offsets[r] as usize;
                let ax = av[s];
                let ay = av[s + 1];
                let az = av[s + 2];
                let bx = bv[t];
                let by = bv[t + 1];
                let bz = bv[t + 2];
                out[s] = ay * bz - az * by;
                out[s + 1] = az * bx - ax * bz;
                out[s + 2] = ax * by - ay * bx;
            }
            Layout::ListOffset(ListOffset { offsets: alo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_f64(out))) })
        }
        DType::Float32 => {
            let av = as_f32(aleaf)?;
            let bv = as_f32(bleaf)?;
            let mut out = vec![0f32; nrows * 3];
            for r in 0..nrows {
                let s = alo.offsets[r] as usize;
                let t = blo.offsets[r] as usize;
                let ax = av[s];
                let ay = av[s + 1];
                let az = av[s + 2];
                let bx = bv[t];
                let by = bv[t + 1];
                let bz = bv[t + 2];
                out[s] = ay * bz - az * by;
                out[s + 1] = az * bx - ax * bz;
                out[s + 2] = ax * by - ay * bx;
            }
            Layout::ListOffset(ListOffset { offsets: alo.offsets.clone(), content: Box::new(Layout::Leaf(new_leaf_f32(out))) })
        }
        _ => return Err(dtype_unsupported("cross", a.dtype)),
    };

    Ok(GrumpyArray { dtype: a.dtype, layout: out_layout })
}

pub fn det(py: Python<'_>, a: &GrumpyArray) -> PyResult<PyObject> {
    if a.dtype != DType::Float64 {
        return Err(dtype_unsupported("det", a.dtype));
    }
    if a.layout.has_union() {
        return Err(layout_unsupported("det", "UnionScalarList layout is not supported"));
    }
    let (lo, leaf, nrows, ncols) = rectangular_2d_list_leaf(&a.layout)?;
    if nrows != ncols {
        return Err(shape_mismatch(
            "det",
            "requires a square 2D matrix",
            "pass a square list->leaf matrix (nrows == ncols).",
        ));
    }
    if leaf.has_nulls {
        return Err(unsupported(
            "det",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling det.",
        ));
    }
    let n = nrows;
    let v = as_f64(leaf)?;
    let mut mat = vec![0f64; n * n];
    for r in 0..n {
        let s = lo.offsets[r] as usize;
        for c in 0..n {
            mat[r * n + c] = v[s + c];
        }
    }
    let (sign, diag_prod) = lu_det_inplace(&mut mat, n)?;
    Ok((sign as f64 * diag_prod).into_py(py))
}

pub fn inv(_py: Python<'_>, a: &GrumpyArray) -> PyResult<GrumpyArray> {
    if a.dtype != DType::Float64 {
        return Err(dtype_unsupported("inv", a.dtype));
    }
    if a.layout.has_union() {
        return Err(layout_unsupported("inv", "UnionScalarList layout is not supported"));
    }
    let (lo, leaf, nrows, ncols) = rectangular_2d_list_leaf(&a.layout)?;
    if nrows != ncols {
        return Err(shape_mismatch(
            "inv",
            "requires a square 2D matrix",
            "pass a square list->leaf matrix (nrows == ncols).",
        ));
    }
    if leaf.has_nulls {
        return Err(unsupported(
            "inv",
            "null/missing values are not supported yet",
            "fill or drop nulls before calling inv.",
        ));
    }
    let n = nrows;
    let v = as_f64(leaf)?;
    let mut lu = vec![0f64; n * n];
    for r in 0..n {
        let s = lo.offsets[r] as usize;
        for c in 0..n {
            lu[r * n + c] = v[s + c];
        }
    }
    let piv = lu_decompose_inplace(&mut lu, n)?;

    // Solve for inverse columns.
    let mut inv = vec![0f64; n * n];
    let mut y = vec![0f64; n];
    let mut xvec = vec![0f64; n];
    for col in 0..n {
        // Forward solve L*y = P*e_col
        for i in 0..n {
            let mut sum = if piv[i] == col { 1.0 } else { 0.0 };
            for j in 0..i {
                sum -= lu[i * n + j] * y[j];
            }
            y[i] = sum;
        }
        // Backward solve U*x = y
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= lu[i * n + j] * xvec[j];
            }
            let pivv = lu[i * n + i];
            if pivv == 0.0 {
                return Err(unsupported(
                    "inv",
                    "matrix is singular and cannot be inverted",
                    "ensure the matrix is full rank.",
                ));
            }
            xvec[i] = sum / pivv;
        }
        for i in 0..n {
            inv[i * n + col] = xvec[i];
        }
    }

    let offsets: Vec<i64> = (0..=n).map(|i| (i as i64) * (n as i64)).collect();
    Ok(GrumpyArray {
        dtype: DType::Float64,
        layout: Layout::ListOffset(ListOffset {
            offsets: Arc::new(offsets),
            content: Box::new(Layout::Leaf(new_leaf_f64(inv))),
        }),
    })
}

// ---------- helpers ----------

fn leaf_1d<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::OffsetView(v) => leaf_1d(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_1d(ix.content.as_ref()),
        Layout::ListOffset(_) => Err(layout_unsupported("linalg", "expected a 1D leaf array")),
        Layout::UnionScalarList(_) => Err(layout_unsupported("linalg", "UnionScalarList layout is not supported")),
    }
}

fn rectangular_2d_list_leaf<'a>(layout: &'a Layout) -> PyResult<(&'a ListOffset, &'a Leaf, usize, usize)> {
    let (lo, leaf) = match layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(l) => (lo, l),
            _ => return Err(layout_unsupported("linalg", "expected a 2D list->leaf array")),
        },
        Layout::OffsetView(v) => return rectangular_2d_list_leaf(v.content.as_ref()),
        Layout::Indexed(ix) => return rectangular_2d_list_leaf(ix.content.as_ref()),
        _ => return Err(layout_unsupported("linalg", "expected a 2D list->leaf array")),
    };
    let nrows = lo.len();
    if nrows == 0 {
        return Ok((lo, leaf, 0, 0));
    }
    let row0 = (lo.offsets[1] - lo.offsets[0]) as usize;
    for r in 0..nrows {
        let len = (lo.offsets[r + 1] - lo.offsets[r]) as usize;
        if len != row0 {
            return Err(shape_mismatch(
                "linalg",
                "expected a rectangular 2D array with constant row length",
                "ensure every row has the same number of columns.",
            ));
        }
    }
    Ok((lo, leaf, nrows, row0))
}

fn as_i32<'a>(leaf: &'a Leaf) -> PyResult<&'a [i32]> {
    match &leaf.buffer {
        LeafBuffer::I32(v) => Ok(v.as_slice()),
        _ => Err(internal_dtype_buffer_mismatch("linalg", leaf.dtype)),
    }
}
fn as_i64<'a>(leaf: &'a Leaf) -> PyResult<&'a [i64]> {
    match &leaf.buffer {
        LeafBuffer::I64(v) => Ok(v.as_slice()),
        _ => Err(internal_dtype_buffer_mismatch("linalg", leaf.dtype)),
    }
}
fn as_u32<'a>(leaf: &'a Leaf) -> PyResult<&'a [u32]> {
    match &leaf.buffer {
        LeafBuffer::U32(v) => Ok(v.as_slice()),
        _ => Err(internal_dtype_buffer_mismatch("linalg", leaf.dtype)),
    }
}
fn as_u64<'a>(leaf: &'a Leaf) -> PyResult<&'a [u64]> {
    match &leaf.buffer {
        LeafBuffer::U64(v) => Ok(v.as_slice()),
        _ => Err(internal_dtype_buffer_mismatch("linalg", leaf.dtype)),
    }
}
fn as_f32<'a>(leaf: &'a Leaf) -> PyResult<&'a [f32]> {
    match &leaf.buffer {
        LeafBuffer::F32(v) => Ok(v.as_slice()),
        _ => Err(internal_dtype_buffer_mismatch("linalg", leaf.dtype)),
    }
}
fn as_f64<'a>(leaf: &'a Leaf) -> PyResult<&'a [f64]> {
    match &leaf.buffer {
        LeafBuffer::F64(v) => Ok(v.as_slice()),
        _ => Err(internal_dtype_buffer_mismatch("linalg", leaf.dtype)),
    }
}

fn new_leaf_i32(v: Vec<i32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(v));
    leaf
}
fn new_leaf_i64(v: Vec<i64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I64(Arc::new(v));
    leaf
}
fn new_leaf_u32(v: Vec<u32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U32(Arc::new(v));
    leaf
}
fn new_leaf_u64(v: Vec<u64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U64(Arc::new(v));
    leaf
}
fn new_leaf_f32(v: Vec<f32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F32(Arc::new(v));
    leaf
}
fn new_leaf_f64(v: Vec<f64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F64(Arc::new(v));
    leaf
}

fn cross_vec3(dtype: DType, a: &Leaf, b: &Leaf) -> PyResult<GrumpyArray> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let bv = as_f64(b)?;
            let out = vec![
                av[1] * bv[2] - av[2] * bv[1],
                av[2] * bv[0] - av[0] * bv[2],
                av[0] * bv[1] - av[1] * bv[0],
            ];
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_f64(out)) })
        }
        DType::Float32 => {
            let av = as_f32(a)?;
            let bv = as_f32(b)?;
            let out = vec![
                av[1] * bv[2] - av[2] * bv[1],
                av[2] * bv[0] - av[0] * bv[2],
                av[0] * bv[1] - av[1] * bv[0],
            ];
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_f32(out)) })
        }
        _ => Err(dtype_unsupported("cross", dtype)),
    }
}

fn lu_decompose_inplace(a: &mut [f64], n: usize) -> PyResult<Vec<usize>> {
    // Returns permutation vector piv such that P*A = L*U and (P*b)[i] = b[piv[i]].
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Pivot row.
        let mut piv_row = k;
        let mut piv_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = i;
            }
        }
        if piv_val == 0.0 {
            return Err(unsupported(
                "inv",
                "matrix is singular and cannot be inverted",
                "ensure the matrix is full rank.",
            ));
        }
        if piv_row != k {
            // swap rows in a
            for j in 0..n {
                a.swap(k * n + j, piv_row * n + j);
            }
            piv.swap(k, piv_row);
        }
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            a[i * n + k] /= akk;
            let lik = a[i * n + k];
            for j in (k + 1)..n {
                a[i * n + j] -= lik * a[k * n + j];
            }
        }
    }
    Ok(piv)
}

fn lu_det_inplace(a: &mut [f64], n: usize) -> PyResult<(i32, f64)> {
    let mut sign: i32 = 1;
    for k in 0..n {
        let mut piv_row = k;
        let mut piv_val = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = i;
            }
        }
        if piv_val == 0.0 {
            return Ok((0, 0.0));
        }
        if piv_row != k {
            for j in 0..n {
                a.swap(k * n + j, piv_row * n + j);
            }
            sign = -sign;
        }
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            a[i * n + k] /= akk;
            let lik = a[i * n + k];
            for j in (k + 1)..n {
                a[i * n + j] -= lik * a[k * n + j];
            }
        }
    }
    let mut prod = 1.0;
    for i in 0..n {
        prod *= a[i * n + i];
    }
    Ok((sign, prod))
}


