use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredOp {
    IsNan,
    IsFinite,
    IsInf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Xor,
}

pub fn compare(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray, op: CmpOp) -> PyResult<GrumpyArray> {
    if a.layout.has_union() || b.layout.has_union() {
        return Err(PyValueError::new_err("Comparisons on union layouts are not implemented yet."));
    }
    // For now: require identical dtype (like our elementwise ops).
    if a.dtype != b.dtype {
        return Err(PyValueError::new_err(
            "Comparison requires matching dtypes (casting not implemented yet).",
        ));
    }
    let layout = compare_layout(py, &a.layout, &b.layout, a.dtype, op)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

pub fn predicate(py: Python<'_>, a: &GrumpyArray, op: PredOp) -> PyResult<GrumpyArray> {
    if a.layout.has_union() {
        return Err(PyValueError::new_err("Predicates on union layouts are not implemented yet."));
    }
    let layout = pred_layout(py, &a.layout, a.dtype, op)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

pub fn logical_bin(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray, op: LogicOp) -> PyResult<GrumpyArray> {
    if a.layout.has_union() || b.layout.has_union() {
        return Err(PyValueError::new_err("Logical ops on union layouts are not implemented yet."));
    }
    if a.dtype != DType::Bool || b.dtype != DType::Bool {
        return Err(PyValueError::new_err("logical_* requires bool arrays."));
    }
    let layout = logical_layout(py, &a.layout, &b.layout, op)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

pub fn logical_not(py: Python<'_>, a: &GrumpyArray) -> PyResult<GrumpyArray> {
    if a.layout.has_union() {
        return Err(PyValueError::new_err("logical_not on union layouts is not implemented yet."));
    }
    if a.dtype != DType::Bool {
        return Err(PyValueError::new_err("logical_not requires bool array."));
    }
    let layout = logical_not_layout(py, &a.layout)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

fn new_bool_leaf(n: usize) -> Leaf {
    let mut out = Leaf::new(DType::Bool);
    out.len = n;
    out.has_nulls = false;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out.buffer = LeafBuffer::Bool(Arc::new(vec![0u8; n]));
    out
}

fn logical_layout(py: Python<'_>, a: &Layout, b: &Layout, op: LogicOp) -> PyResult<Layout> {
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => Ok(Layout::Leaf(logical_leaf(py, la, lb, op)?)),
        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => {
            if oa.offsets != ob.offsets || oa.len() != ob.len() {
                return Err(PyValueError::new_err("Logical op requires identical ragged structure for now."));
            }
            let content = logical_layout(py, oa.content.as_ref(), ob.content.as_ref(), op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }))
        }
        _ => Err(PyValueError::new_err("Logical op requires matching layouts for now.")),
    }
}

fn logical_leaf(_py: Python<'_>, a: &Leaf, b: &Leaf, op: LogicOp) -> PyResult<Leaf> {
    if a.len != b.len {
        return Err(PyValueError::new_err("Leaf lengths differ."));
    }
    let n = a.len;
    let mut out = new_bool_leaf(n);
    out.has_nulls = a.has_nulls || b.has_nulls;
    let out_valid = Arc::make_mut(&mut out.validity);
    for i in 0..n {
        out_valid.set(i, a.validity[i] && b.validity[i]);
    }
    let aa = match &a.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => return Err(PyValueError::new_err("Expected bool leaf.")) };
    let bb = match &b.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => return Err(PyValueError::new_err("Expected bool leaf.")) };
    let oo = match &mut out.buffer { LeafBuffer::Bool(v) => Arc::make_mut(v), _ => unreachable!() };
    for i in 0..n {
        if !out_valid[i] { continue; }
        let x = aa[i] != 0;
        let y = bb[i] != 0;
        let z = match op { LogicOp::And => x & y, LogicOp::Or => x | y, LogicOp::Xor => x ^ y };
        oo[i] = z as u8;
    }
    Ok(out)
}

fn logical_not_layout(py: Python<'_>, a: &Layout) -> PyResult<Layout> {
    match a {
        Layout::Leaf(la) => Ok(Layout::Leaf(logical_not_leaf(py, la)?)),
        Layout::ListOffset(lo) => {
            let content = logical_not_layout(py, lo.content.as_ref())?;
            Ok(Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(content) }))
        }
        _ => Err(PyValueError::new_err("logical_not requires leaf or list layout.")),
    }
}

fn logical_not_leaf(_py: Python<'_>, a: &Leaf) -> PyResult<Leaf> {
    let n = a.len;
    let mut out = new_bool_leaf(n);
    out.has_nulls = a.has_nulls;
    out.validity = a.validity.clone();
    let aa = match &a.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => return Err(PyValueError::new_err("Expected bool leaf.")) };
    let oo = match &mut out.buffer { LeafBuffer::Bool(v) => Arc::make_mut(v), _ => unreachable!() };
    for i in 0..n {
        if !a.validity[i] { continue; }
        oo[i] = (aa[i] == 0) as u8;
    }
    Ok(out)
}

fn compare_layout(py: Python<'_>, a: &Layout, b: &Layout, dt: DType, op: CmpOp) -> PyResult<Layout> {
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => Ok(Layout::Leaf(compare_leaf(py, la, lb, dt, op)?)),
        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => {
            if oa.offsets != ob.offsets || oa.len() != ob.len() {
                return Err(PyValueError::new_err("Comparison requires identical ragged structure for now."));
            }
            let content = compare_layout(py, oa.content.as_ref(), ob.content.as_ref(), dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }))
        }
        _ => Err(PyValueError::new_err("Comparison requires matching layouts for now.")),
    }
}

fn compare_leaf(_py: Python<'_>, a: &Leaf, b: &Leaf, dt: DType, op: CmpOp) -> PyResult<Leaf> {
    if a.len != b.len {
        return Err(PyValueError::new_err("Leaf lengths differ."));
    }
    let n = a.len;
    let mut out = new_bool_leaf(n);
    // validity = a & b
    out.has_nulls = a.has_nulls || b.has_nulls;
    let out_valid = Arc::make_mut(&mut out.validity);
    for i in 0..n {
        out_valid.set(i, a.validity[i] && b.validity[i]);
    }

    let o = match &mut out.buffer {
        LeafBuffer::Bool(o) => Arc::make_mut(o),
        _ => unreachable!(),
    };

    match dt {
        DType::Int32 => {
            let aa = match &a.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            let bb = match &b.buffer { LeafBuffer::I32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n {
                if !out_valid[i] { continue; }
                let x = aa[i];
                let y = bb[i];
                o[i] = cmp_bool_i32(x, y, op) as u8;
            }
        }
        DType::Int64 => {
            let aa = match &a.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            let bb = match &b.buffer { LeafBuffer::I64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n {
                if !out_valid[i] { continue; }
                let x = aa[i];
                let y = bb[i];
                o[i] = cmp_bool_i64(x, y, op) as u8;
            }
        }
        DType::Float32 => {
            let aa = match &a.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            let bb = match &b.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n {
                if !out_valid[i] { continue; }
                let x = aa[i];
                let y = bb[i];
                o[i] = cmp_bool_f32(x, y, op) as u8;
            }
        }
        DType::Float64 => {
            let aa = match &a.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            let bb = match &b.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n {
                if !out_valid[i] { continue; }
                let x = aa[i];
                let y = bb[i];
                o[i] = cmp_bool_f64(x, y, op) as u8;
            }
        }
        _ => return Err(PyValueError::new_err("Comparison not implemented for this dtype.")),
    }
    Ok(out)
}

fn pred_layout(py: Python<'_>, a: &Layout, dt: DType, op: PredOp) -> PyResult<Layout> {
    match a {
        Layout::Leaf(la) => Ok(Layout::Leaf(pred_leaf(py, la, dt, op)?)),
        Layout::ListOffset(lo) => {
            let content = pred_layout(py, lo.content.as_ref(), dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(content) }))
        }
        _ => Err(PyValueError::new_err("Predicate requires leaf or list layout.")),
    }
}

fn pred_leaf(_py: Python<'_>, a: &Leaf, dt: DType, op: PredOp) -> PyResult<Leaf> {
    let n = a.len;
    let mut out = new_bool_leaf(n);
    out.has_nulls = a.has_nulls;
    out.validity = a.validity.clone();
    let o = match &mut out.buffer {
        LeafBuffer::Bool(o) => Arc::make_mut(o),
        _ => unreachable!(),
    };
    match (dt, op) {
        (DType::Float32, PredOp::IsNan) => {
            let aa = match &a.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n { if a.validity[i] { o[i] = aa[i].is_nan() as u8; } }
        }
        (DType::Float64, PredOp::IsNan) => {
            let aa = match &a.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n { if a.validity[i] { o[i] = aa[i].is_nan() as u8; } }
        }
        (DType::Float32, PredOp::IsInf) => {
            let aa = match &a.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n { if a.validity[i] { o[i] = aa[i].is_infinite() as u8; } }
        }
        (DType::Float64, PredOp::IsInf) => {
            let aa = match &a.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n { if a.validity[i] { o[i] = aa[i].is_infinite() as u8; } }
        }
        (DType::Float32, PredOp::IsFinite) => {
            let aa = match &a.buffer { LeafBuffer::F32(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n { if a.validity[i] { o[i] = aa[i].is_finite() as u8; } }
        }
        (DType::Float64, PredOp::IsFinite) => {
            let aa = match &a.buffer { LeafBuffer::F64(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n { if a.validity[i] { o[i] = aa[i].is_finite() as u8; } }
        }
        _ => return Err(PyValueError::new_err("Predicate not implemented for this dtype.")),
    }
    Ok(out)
}

#[inline]
fn cmp_bool_i32(x: i32, y: i32, op: CmpOp) -> bool {
    match op { CmpOp::Eq => x == y, CmpOp::Ne => x != y, CmpOp::Lt => x < y, CmpOp::Le => x <= y, CmpOp::Gt => x > y, CmpOp::Ge => x >= y }
}
#[inline]
fn cmp_bool_i64(x: i64, y: i64, op: CmpOp) -> bool {
    match op { CmpOp::Eq => x == y, CmpOp::Ne => x != y, CmpOp::Lt => x < y, CmpOp::Le => x <= y, CmpOp::Gt => x > y, CmpOp::Ge => x >= y }
}
#[inline]
fn cmp_bool_f32(x: f32, y: f32, op: CmpOp) -> bool {
    match op { CmpOp::Eq => x == y, CmpOp::Ne => x != y, CmpOp::Lt => x < y, CmpOp::Le => x <= y, CmpOp::Gt => x > y, CmpOp::Ge => x >= y }
}
#[inline]
fn cmp_bool_f64(x: f64, y: f64, op: CmpOp) -> bool {
    match op { CmpOp::Eq => x == y, CmpOp::Ne => x != y, CmpOp::Lt => x < y, CmpOp::Le => x <= y, CmpOp::Gt => x > y, CmpOp::Ge => x >= y }
}


