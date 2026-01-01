use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

pub enum TensorOut {
    Scalar(PyObject),
    Array(GrumpyArray),
}

/// Restricted einsum:
/// - 1 or 2 operands only
/// - no ellipsis
/// - only 1D leaf vectors and 2D rectangular list->leaf matrices
/// - dtype: float64/int32 only for now
///
/// Supported patterns (examples):
/// - "i,i->" dot
/// - "ij,jk->ik" matmul
/// - "ij->ji" transpose
/// - "ii->" trace
/// - "ij->i" row-sum, "ij->j" col-sum, "ij->" total sum
/// - "ij,ij->" sum of elementwise product
/// - "i,j->ij" outer
pub fn einsum(py: Python<'_>, subscripts: &str, operands: &[GrumpyArray]) -> PyResult<TensorOut> {
    if subscripts.contains("...") {
        return Err(PyValueError::new_err("einsum: ellipsis not supported yet."));
    }
    if operands.is_empty() || operands.len() > 2 {
        return Err(PyValueError::new_err("einsum: only 1 or 2 operands supported."));
    }
    let (lhs, rhs_opt) = match subscripts.split_once("->") {
        Some((l, r)) => (l.trim(), Some(r.trim())),
        None => (subscripts.trim(), None),
    };
    let in_terms: Vec<&str> = lhs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if in_terms.len() != operands.len() {
        return Err(PyValueError::new_err("einsum: number of subscripts does not match operands."));
    }
    for t in &in_terms {
        if t.is_empty() {
            return Err(PyValueError::new_err("einsum: empty subscript term."));
        }
    }
    let rhs = rhs_opt.unwrap_or("");

    // quick dispatch for common 1D/2D patterns (no general contraction engine yet)
    Ok(match operands.len() {
        1 => einsum1(py, in_terms[0], rhs, &operands[0])?,
        2 => einsum2(py, in_terms[0], in_terms[1], rhs, &operands[0], &operands[1])?,
        _ => unreachable!(),
    })
}

pub fn tensordot(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray, axes: usize) -> PyResult<TensorOut> {
    // Restricted tensordot for 1D/2D rectangular:
    // - axes=0: outer (vector-vector) or outer-like for 1D×1D -> 2D
    // - axes=1: dot (1D×1D -> scalar) or matmul (2D×2D -> 2D) or matvec/vecmat
    // - axes=2: 2D×2D Frobenius inner product -> scalar
    if a.dtype != b.dtype {
        return Err(PyValueError::new_err("tensordot requires matching dtypes (for now)."));
    }
    if a.layout.has_union() || b.layout.has_union() {
        return Err(PyValueError::new_err("tensordot on union layouts not implemented."));
    }
    match axes {
        0 => {
            // Only vector-vector -> matrix
            let la = leaf_1d(&a.layout)?;
            let lb = leaf_1d(&b.layout)?;
            if la.has_nulls || lb.has_nulls {
                return Err(PyValueError::new_err("tensordot: nulls not supported yet."));
            }
            let out = outer_vec_vec(a.dtype, la, lb)?;
            Ok(TensorOut::Array(out))
        }
        1 => {
            // dot / matmul / matvec / vecmat
            if let (Ok(la), Ok(lb)) = (leaf_1d(&a.layout), leaf_1d(&b.layout)) {
                if la.has_nulls || lb.has_nulls {
                    return Err(PyValueError::new_err("tensordot: nulls not supported yet."));
                }
                if la.len != lb.len {
                    return Err(PyValueError::new_err("tensordot axes=1: vector lengths must match."));
                }
                let s = dot_vec_vec(py, a.dtype, la, lb)?;
                return Ok(TensorOut::Scalar(s));
            }
            // 2D×2D -> 2D
            if let (Ok((alo, aleaf, ar, ac)), Ok((blo, bleaf, br, bc))) =
                (rect2d(&a.layout), rect2d(&b.layout))
            {
                if aleaf.has_nulls || bleaf.has_nulls {
                    return Err(PyValueError::new_err("tensordot: nulls not supported yet."));
                }
                if ac != br {
                    return Err(PyValueError::new_err("tensordot axes=1: inner dimensions must match."));
                }
                let out = matmul(a.dtype, alo, aleaf, ar, ac, blo, bleaf, br, bc)?;
                return Ok(TensorOut::Array(out));
            }
            // matvec / vecmat
            if let (Ok((alo, aleaf, ar, ac)), Ok(lb)) = (rect2d(&a.layout), leaf_1d(&b.layout)) {
                if aleaf.has_nulls || lb.has_nulls {
                    return Err(PyValueError::new_err("tensordot: nulls not supported yet."));
                }
                if ac != lb.len {
                    return Err(PyValueError::new_err("tensordot axes=1: inner dimensions must match."));
                }
                let out = matvec(a.dtype, alo, aleaf, ar, ac, lb)?;
                return Ok(TensorOut::Array(out));
            }
            if let (Ok(la), Ok((blo, bleaf, br, bc))) = (leaf_1d(&a.layout), rect2d(&b.layout)) {
                if la.has_nulls || bleaf.has_nulls {
                    return Err(PyValueError::new_err("tensordot: nulls not supported yet."));
                }
                if la.len != br {
                    return Err(PyValueError::new_err("tensordot axes=1: inner dimensions must match."));
                }
                let out = vecmat(a.dtype, la, blo, bleaf, br, bc)?;
                return Ok(TensorOut::Array(out));
            }
            Err(PyValueError::new_err("tensordot axes=1: unsupported shapes/layouts."))
        }
        2 => {
            // 2D×2D Frobenius inner product
            let (alo, aleaf, ar, ac) = rect2d(&a.layout)?;
            let (blo, bleaf, br, bc) = rect2d(&b.layout)?;
            if ar != br || ac != bc {
                return Err(PyValueError::new_err("tensordot axes=2: shapes must match."));
            }
            if aleaf.has_nulls || bleaf.has_nulls {
                return Err(PyValueError::new_err("tensordot: nulls not supported yet."));
            }
            let av = as_f64_or_i32(a.dtype, aleaf)?;
            let bv = as_f64_or_i32(a.dtype, bleaf)?;
            // offsets unused aside from confirming rectangular
            let _ = alo;
            let _ = blo;
            match (av, bv) {
                (NumSlice::F64(av), NumSlice::F64(bv)) => {
                    let mut acc = 0.0;
                    for i in 0..(ar * ac) {
                        acc += av[i] * bv[i];
                    }
                    Ok(TensorOut::Scalar(acc.into_py(py)))
                }
                (NumSlice::I32(av), NumSlice::I32(bv)) => {
                    let mut acc: i64 = 0;
                    for i in 0..(ar * ac) {
                        acc += (av[i] as i64) * (bv[i] as i64);
                    }
                    Ok(TensorOut::Scalar(acc.into_py(py)))
                }
                _ => Err(PyValueError::new_err("tensordot: dtype not supported.")),
            }
        }
        _ => Err(PyValueError::new_err("tensordot: only axes=0/1/2 supported.")),
    }
}

fn einsum1(py: Python<'_>, term: &str, rhs: &str, a: &GrumpyArray) -> PyResult<TensorOut> {
    // Unary patterns on matrix: transpose, trace, sums.
    if term.len() == 1 {
        // "i->" sum
        if rhs.is_empty() {
            let leaf = leaf_1d(&a.layout)?;
            if leaf.has_nulls {
                return Err(PyValueError::new_err("einsum: nulls not supported yet."));
            }
            return Ok(TensorOut::Scalar(sum_vec(py, a.dtype, leaf)?));
        }
        return Err(PyValueError::new_err("einsum: unsupported 1D unary pattern."));
    }
    if term.len() != 2 {
        return Err(PyValueError::new_err("einsum: only 1D or 2D unary patterns supported."));
    }
    let (lo, leaf, nrows, ncols) = rect2d(&a.layout)?;
    if leaf.has_nulls {
        return Err(PyValueError::new_err("einsum: nulls not supported yet."));
    }
    let chars: Vec<char> = term.chars().collect();
    let i = chars[0];
    let j = chars[1];
    if rhs.is_empty() {
        // "ij->" sum all
        return Ok(TensorOut::Scalar(sum_mat(py, a.dtype, leaf, nrows, ncols)?));
    }
    if rhs.len() == 2 {
        let outc: Vec<char> = rhs.chars().collect();
        if outc[0] == j && outc[1] == i {
            // transpose
            let out = transpose(a.dtype, lo, leaf, nrows, ncols)?;
            return Ok(TensorOut::Array(out));
        }
    }
    if rhs.len() == 1 {
        let out = rhs.chars().next().unwrap();
        if out == i {
            let out = sum_rows(a.dtype, lo, leaf, nrows, ncols)?;
            return Ok(TensorOut::Array(out));
        }
        if out == j {
            let out = sum_cols(a.dtype, lo, leaf, nrows, ncols)?;
            return Ok(TensorOut::Array(out));
        }
    }
    // "ii->" trace
    if i == j && rhs.is_empty() {
        return Ok(TensorOut::Scalar(trace_mat(py, lo, leaf, nrows, ncols)?));
    }
    Err(PyValueError::new_err("einsum: unary pattern not implemented."))
}

fn einsum2(py: Python<'_>, ta: &str, tb: &str, rhs: &str, a: &GrumpyArray, b: &GrumpyArray) -> PyResult<TensorOut> {
    if a.dtype != b.dtype {
        return Err(PyValueError::new_err("einsum: operand dtypes must match (for now)."));
    }
    if ta.len() == 1 && tb.len() == 1 && rhs.is_empty() && ta == tb {
        // dot
        let la = leaf_1d(&a.layout)?;
        let lb = leaf_1d(&b.layout)?;
        if la.has_nulls || lb.has_nulls {
            return Err(PyValueError::new_err("einsum: nulls not supported yet."));
        }
        if la.len != lb.len {
            return Err(PyValueError::new_err("einsum: vector lengths must match."));
        }
        return Ok(TensorOut::Scalar(dot_vec_vec(py, a.dtype, la, lb)?));
    }
    if ta.len() == 1 && tb.len() == 1 && rhs.len() == 2 && rhs.chars().collect::<Vec<_>>() == vec![ta.chars().next().unwrap(), tb.chars().next().unwrap()] {
        // outer
        let la = leaf_1d(&a.layout)?;
        let lb = leaf_1d(&b.layout)?;
        if la.has_nulls || lb.has_nulls {
            return Err(PyValueError::new_err("einsum: nulls not supported yet."));
        }
        let out = outer_vec_vec(a.dtype, la, lb)?;
        return Ok(TensorOut::Array(out));
    }
    // "ij,ij->" sum elementwise product
    if ta.len() == 2 && tb.len() == 2 && rhs.is_empty() && ta == tb {
        let (_, aleaf, ar, ac) = rect2d(&a.layout)?;
        let (_, bleaf, br, bc) = rect2d(&b.layout)?;
        if ar != br || ac != bc {
            return Err(PyValueError::new_err("einsum: shapes must match for ij,ij->."));
        }
        if aleaf.has_nulls || bleaf.has_nulls {
            return Err(PyValueError::new_err("einsum: nulls not supported yet."));
        }
        return Ok(TensorOut::Scalar(frob_inner(py, a.dtype, aleaf, bleaf, ar * ac)?));
    }
    // "ij,jk->ik" matmul
    if ta.len() == 2 && tb.len() == 2 && rhs.len() == 2 {
        let ca: Vec<char> = ta.chars().collect();
        let cb: Vec<char> = tb.chars().collect();
        let cr: Vec<char> = rhs.chars().collect();
        if ca[1] == cb[0] && cr[0] == ca[0] && cr[1] == cb[1] {
            let (alo, aleaf, ar, ac) = rect2d(&a.layout)?;
            let (blo, bleaf, br, bc) = rect2d(&b.layout)?;
            if ac != br {
                return Err(PyValueError::new_err("einsum: inner dimensions must match for ij,jk->ik."));
            }
            if aleaf.has_nulls || bleaf.has_nulls {
                return Err(PyValueError::new_err("einsum: nulls not supported yet."));
            }
            let out = matmul(a.dtype, alo, aleaf, ar, ac, blo, bleaf, br, bc)?;
            return Ok(TensorOut::Array(out));
        }
    }

    // fall back to error (no general contraction engine yet)
    // Provide helpful hint about what *is* supported.
    let supported = "Supported einsum patterns include: i,i-> ; ij,jk->ik ; ij->ji ; ii-> ; ij->i/j ; ij,ij-> ; i,j->ij";
    Err(PyValueError::new_err(format!("einsum: pattern not implemented. {supported}")))
}

// ---------- kernels ----------

fn dot_vec_vec(py: Python<'_>, dtype: DType, a: &Leaf, b: &Leaf) -> PyResult<PyObject> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let bv = as_f64(b)?;
            let mut acc = 0.0;
            for i in 0..a.len {
                acc += av[i] * bv[i];
            }
            Ok(acc.into_py(py))
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let bv = as_i32(b)?;
            let mut acc: i64 = 0;
            for i in 0..a.len {
                acc += (av[i] as i64) * (bv[i] as i64);
            }
            Ok(acc.into_py(py))
        }
        _ => Err(PyValueError::new_err("einsum/tensordot: dtype not supported (use float64 or int32).")),
    }
}

fn outer_vec_vec(dtype: DType, a: &Leaf, b: &Leaf) -> PyResult<GrumpyArray> {
    let n = a.len;
    let m = b.len;
    let offsets: Vec<i64> = (0..=n).map(|i| (i as i64) * (m as i64)).collect();
    let total = n * m;
    let content = match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let bv = as_f64(b)?;
            let mut out = vec![0f64; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i] * bv[j];
                }
            }
            Layout::Leaf(new_leaf_f64(out))
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let bv = as_i32(b)?;
            let mut out = vec![0i32; total];
            for i in 0..n {
                for j in 0..m {
                    out[i * m + j] = av[i].wrapping_mul(bv[j]);
                }
            }
            Layout::Leaf(new_leaf_i32(out))
        }
        _ => return Err(PyValueError::new_err("outer: dtype not supported.")),
    };
    Ok(GrumpyArray { dtype, layout: Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(content) }) })
}

fn matmul(
    dtype: DType,
    alo: &ListOffset,
    aleaf: &Leaf,
    ar: usize,
    ac: usize,
    blo: &ListOffset,
    bleaf: &Leaf,
    _br: usize,
    bc: usize,
) -> PyResult<GrumpyArray> {
    let _ = alo;
    let _ = blo;
    let offsets: Vec<i64> = (0..=ar).map(|i| (i as i64) * (bc as i64)).collect();
    let total = ar * bc;
    let out_layout = match dtype {
        DType::Float64 => {
            let av = as_f64(aleaf)?;
            let bv = as_f64(bleaf)?;
            let mut out = vec![0f64; total];
            for i in 0..ar {
                for k in 0..ac {
                    let aik = av[i * ac + k];
                    let bk = &bv[k * bc..(k + 1) * bc];
                    let row = &mut out[i * bc..(i + 1) * bc];
                    for j in 0..bc {
                        row[j] += aik * bk[j];
                    }
                }
            }
            Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(Layout::Leaf(new_leaf_f64(out))) })
        }
        DType::Int32 => {
            let av = as_i32(aleaf)?;
            let bv = as_i32(bleaf)?;
            let mut out = vec![0i32; total];
            for i in 0..ar {
                for k in 0..ac {
                    let aik = av[i * ac + k] as i64;
                    for j in 0..bc {
                        let val = out[i * bc + j] as i64 + aik * (bv[k * bc + j] as i64);
                        out[i * bc + j] = val as i32;
                    }
                }
            }
            Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(Layout::Leaf(new_leaf_i32(out))) })
        }
        _ => return Err(PyValueError::new_err("matmul: dtype not supported.")),
    };
    Ok(GrumpyArray { dtype, layout: out_layout })
}

fn matvec(dtype: DType, _alo: &ListOffset, aleaf: &Leaf, ar: usize, ac: usize, b: &Leaf) -> PyResult<GrumpyArray> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(aleaf)?;
            let bv = as_f64(b)?;
            let mut out = vec![0f64; ar];
            for i in 0..ar {
                let mut acc = 0.0;
                for k in 0..ac {
                    acc += av[i * ac + k] * bv[k];
                }
                out[i] = acc;
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_f64(out)) })
        }
        DType::Int32 => {
            let av = as_i32(aleaf)?;
            let bv = as_i32(b)?;
            let mut out = vec![0i32; ar];
            for i in 0..ar {
                let mut acc: i64 = 0;
                for k in 0..ac {
                    acc += (av[i * ac + k] as i64) * (bv[k] as i64);
                }
                out[i] = acc as i32;
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_i32(out)) })
        }
        _ => Err(PyValueError::new_err("matvec: dtype not supported.")),
    }
}

fn vecmat(dtype: DType, a: &Leaf, _blo: &ListOffset, bleaf: &Leaf, br: usize, bc: usize) -> PyResult<GrumpyArray> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let bv = as_f64(bleaf)?;
            let mut out = vec![0f64; bc];
            for k in 0..br {
                let ak = av[k];
                for j in 0..bc {
                    out[j] += ak * bv[k * bc + j];
                }
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_f64(out)) })
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let bv = as_i32(bleaf)?;
            let mut out = vec![0i32; bc];
            for k in 0..br {
                let ak = av[k] as i64;
                for j in 0..bc {
                    out[j] = (out[j] as i64 + ak * (bv[k * bc + j] as i64)) as i32;
                }
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_i32(out)) })
        }
        _ => Err(PyValueError::new_err("vecmat: dtype not supported.")),
    }
}

fn frob_inner(py: Python<'_>, dtype: DType, a: &Leaf, b: &Leaf, n: usize) -> PyResult<PyObject> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let bv = as_f64(b)?;
            let mut acc = 0.0;
            for i in 0..n {
                acc += av[i] * bv[i];
            }
            Ok(acc.into_py(py))
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let bv = as_i32(b)?;
            let mut acc: i64 = 0;
            for i in 0..n {
                acc += (av[i] as i64) * (bv[i] as i64);
            }
            Ok(acc.into_py(py))
        }
        _ => Err(PyValueError::new_err("einsum: dtype not supported.")),
    }
}

fn sum_vec(py: Python<'_>, dtype: DType, a: &Leaf) -> PyResult<PyObject> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let mut acc = 0.0;
            for i in 0..a.len {
                acc += av[i];
            }
            Ok(acc.into_py(py))
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let mut acc: i64 = 0;
            for i in 0..a.len {
                acc += av[i] as i64;
            }
            Ok(acc.into_py(py))
        }
        _ => Err(PyValueError::new_err("einsum: dtype not supported.")),
    }
}

fn sum_mat(py: Python<'_>, dtype: DType, a: &Leaf, nrows: usize, ncols: usize) -> PyResult<PyObject> {
    let n = nrows * ncols;
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let mut acc = 0.0;
            for i in 0..n {
                acc += av[i];
            }
            Ok(acc.into_py(py))
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let mut acc: i64 = 0;
            for i in 0..n {
                acc += av[i] as i64;
            }
            Ok(acc.into_py(py))
        }
        _ => Err(PyValueError::new_err("einsum: dtype not supported.")),
    }
}

fn sum_rows(dtype: DType, lo: &ListOffset, a: &Leaf, nrows: usize, ncols: usize) -> PyResult<GrumpyArray> {
    let _ = lo;
    let _ = ncols;
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let mut out = vec![0f64; nrows];
            for r in 0..nrows {
                let mut acc = 0.0;
                let s = r * ncols;
                for c in 0..ncols {
                    acc += av[s + c];
                }
                out[r] = acc;
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_f64(out)) })
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let mut out = vec![0i32; nrows];
            for r in 0..nrows {
                let mut acc: i64 = 0;
                let s = r * ncols;
                for c in 0..ncols {
                    acc += av[s + c] as i64;
                }
                out[r] = acc as i32;
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_i32(out)) })
        }
        _ => Err(PyValueError::new_err("einsum: dtype not supported.")),
    }
}

fn sum_cols(dtype: DType, _lo: &ListOffset, a: &Leaf, nrows: usize, ncols: usize) -> PyResult<GrumpyArray> {
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let mut out = vec![0f64; ncols];
            for r in 0..nrows {
                let s = r * ncols;
                for c in 0..ncols {
                    out[c] += av[s + c];
                }
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_f64(out)) })
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let mut out = vec![0i32; ncols];
            for r in 0..nrows {
                let s = r * ncols;
                for c in 0..ncols {
                    out[c] = (out[c] as i64 + av[s + c] as i64) as i32;
                }
            }
            Ok(GrumpyArray { dtype, layout: Layout::Leaf(new_leaf_i32(out)) })
        }
        _ => Err(PyValueError::new_err("einsum: dtype not supported.")),
    }
}

fn trace_mat(py: Python<'_>, lo: &ListOffset, a: &Leaf, nrows: usize, ncols: usize) -> PyResult<PyObject> {
    let n = nrows.min(ncols);
    match a.dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let mut acc = 0.0;
            for i in 0..n {
                acc += av[lo.offsets[i] as usize + i];
            }
            Ok(acc.into_py(py))
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let mut acc: i64 = 0;
            for i in 0..n {
                acc += av[lo.offsets[i] as usize + i] as i64;
            }
            Ok(acc.into_py(py))
        }
        _ => Err(PyValueError::new_err("einsum trace: dtype not supported.")),
    }
}

fn transpose(dtype: DType, _lo: &ListOffset, a: &Leaf, nrows: usize, ncols: usize) -> PyResult<GrumpyArray> {
    // produce ncols x nrows rectangular list->leaf
    let offsets: Vec<i64> = (0..=ncols).map(|i| (i as i64) * (nrows as i64)).collect();
    let total = nrows * ncols;
    match dtype {
        DType::Float64 => {
            let av = as_f64(a)?;
            let mut out = vec![0f64; total];
            for r in 0..nrows {
                for c in 0..ncols {
                    out[c * nrows + r] = av[r * ncols + c];
                }
            }
            Ok(GrumpyArray { dtype, layout: Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(Layout::Leaf(new_leaf_f64(out))) }) })
        }
        DType::Int32 => {
            let av = as_i32(a)?;
            let mut out = vec![0i32; total];
            for r in 0..nrows {
                for c in 0..ncols {
                    out[c * nrows + r] = av[r * ncols + c];
                }
            }
            Ok(GrumpyArray { dtype, layout: Layout::ListOffset(ListOffset { offsets: Arc::new(offsets), content: Box::new(Layout::Leaf(new_leaf_i32(out))) }) })
        }
        _ => Err(PyValueError::new_err("transpose: dtype not supported.")),
    }
}

// ---------- layout helpers ----------

fn leaf_1d<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::OffsetView(v) => leaf_1d(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_1d(ix.content.as_ref()),
        Layout::ListOffset(_) => Err(PyValueError::new_err("Expected 1D leaf array.")),
        Layout::UnionScalarList(_) => Err(PyValueError::new_err("Union not supported.")),
    }
}

fn rect2d<'a>(layout: &'a Layout) -> PyResult<(&'a ListOffset, &'a Leaf, usize, usize)> {
    let (lo, leaf) = match layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(l) => (lo, l),
            _ => return Err(PyValueError::new_err("Expected 2D list->leaf array.")),
        },
        Layout::OffsetView(v) => return rect2d(v.content.as_ref()),
        Layout::Indexed(ix) => return rect2d(ix.content.as_ref()),
        _ => return Err(PyValueError::new_err("Expected 2D list->leaf array.")),
    };
    let nrows = lo.len();
    if nrows == 0 {
        return Ok((lo, leaf, 0, 0));
    }
    let row0 = (lo.offsets[1] - lo.offsets[0]) as usize;
    for r in 0..nrows {
        let len = (lo.offsets[r + 1] - lo.offsets[r]) as usize;
        if len != row0 {
            return Err(PyValueError::new_err("Expected rectangular 2D array (constant row length)."));
        }
    }
    Ok((lo, leaf, nrows, row0))
}

fn as_f64<'a>(leaf: &'a Leaf) -> PyResult<&'a [f64]> {
    match &leaf.buffer {
        LeafBuffer::F64(v) => Ok(v.as_slice()),
        _ => Err(PyValueError::new_err("dtype mismatch (expected float64).")),
    }
}
fn as_i32<'a>(leaf: &'a Leaf) -> PyResult<&'a [i32]> {
    match &leaf.buffer {
        LeafBuffer::I32(v) => Ok(v.as_slice()),
        _ => Err(PyValueError::new_err("dtype mismatch (expected int32).")),
    }
}

enum NumSlice<'a> {
    F64(&'a [f64]),
    I32(&'a [i32]),
}
fn as_f64_or_i32<'a>(_dtype: DType, leaf: &'a Leaf) -> PyResult<NumSlice<'a>> {
    match &leaf.buffer {
        LeafBuffer::F64(v) => Ok(NumSlice::F64(v.as_slice())),
        LeafBuffer::I32(v) => Ok(NumSlice::I32(v.as_slice())),
        _ => Err(PyValueError::new_err("dtype not supported (use float64 or int32).")),
    }
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
fn new_leaf_i32(v: Vec<i32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(v));
    leaf
}


