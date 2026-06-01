use crate::dtype::DType;
use crate::error::{
    dtype_unsupported, internal_dtype_buffer_mismatch, layout_unsupported, shape_mismatch, unsupported,
};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use bitvec::bitvec;
use bitvec::order::Lsb0;
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
    let (a2, b2) = if a.dtype != b.dtype {
        crate::cast::cast_array_pair(a, b)?
    } else {
        (a.clone(), b.clone())
    };
    let layout = compare_layout(py, &a2.layout, &b2.layout, a2.dtype, op)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

pub fn predicate(py: Python<'_>, a: &GrumpyArray, op: PredOp) -> PyResult<GrumpyArray> {
    let layout = pred_layout(py, &a.layout, a.dtype, op)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

pub fn logical_bin(py: Python<'_>, a: &GrumpyArray, b: &GrumpyArray, op: LogicOp) -> PyResult<GrumpyArray> {
    if a.dtype != DType::Bool || b.dtype != DType::Bool {
        return Err(unsupported(
            "logical_*",
            "requires bool arrays for both operands",
            "cast inputs with .astype(gr.bool_) or use comparison ops.",
        ));
    }
    let layout = logical_layout(py, &a.layout, &b.layout, op)?;
    Ok(GrumpyArray { dtype: DType::Bool, layout })
}

pub fn logical_not(py: Python<'_>, a: &GrumpyArray) -> PyResult<GrumpyArray> {
    if a.dtype != DType::Bool {
        return Err(unsupported(
            "logical_not",
            "requires a bool array",
            "cast the input with .astype(gr.bool_) or use comparison ops.",
        ));
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
                return Err(shape_mismatch(
                    "logical op",
                    "requires identical ragged structure",
                    "ensure both operands share the same list offsets.",
                ));
            }
            let content = logical_layout(py, oa.content.as_ref(), ob.content.as_ref(), op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }))
        }
        (Layout::UnionScalarList(ua), Layout::UnionScalarList(ub)) => {
            if ua.tags != ub.tags || ua.index != ub.index || ua.lists.offsets != ub.lists.offsets {
                return Err(shape_mismatch(
                    "logical op",
                    "requires identical union structure",
                    "ensure both operands share the same union tags, index, and list offsets.",
                ));
            }
            let scalars = logical_leaf(py, &ua.scalars, &ub.scalars, op)?;
            let list_content = logical_layout(py, ua.lists.content.as_ref(), ub.lists.content.as_ref(), op)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: ua.tags.clone(),
                index: ua.index.clone(),
                scalars,
                lists: ListOffset { offsets: ua.lists.offsets.clone(), content: Box::new(list_content) },
            }))
        }
        _ => Err(layout_unsupported("logical op", "requires matching layouts")),
    }
}

fn logical_leaf(_py: Python<'_>, a: &Leaf, b: &Leaf, op: LogicOp) -> PyResult<Leaf> {
    if a.len != b.len {
        return Err(shape_mismatch(
            "logical op",
            format!("leaf length mismatch: {} vs {}", a.len, b.len),
            "ensure both operands have the same number of elements.",
        ));
    }
    let n = a.len;
    let mut out = new_bool_leaf(n);
    out.has_nulls = a.has_nulls || b.has_nulls;
    let out_valid = Arc::make_mut(&mut out.validity);
    for i in 0..n {
        out_valid.set(i, a.validity[i] && b.validity[i]);
    }
    let aa = match &a.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => return Err(internal_dtype_buffer_mismatch("logical op", DType::Bool)) };
    let bb = match &b.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => return Err(internal_dtype_buffer_mismatch("logical op", DType::Bool)) };
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
        Layout::UnionScalarList(u) => {
            let scalars = logical_not_leaf(py, &u.scalars)?;
            let list_content = logical_not_layout(py, u.lists.content.as_ref())?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars,
                lists: ListOffset { offsets: u.lists.offsets.clone(), content: Box::new(list_content) },
            }))
        }
        Layout::OffsetView(v) => {
            let content = logical_not_layout(py, v.content.as_ref())?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = logical_not_layout(py, ix.content.as_ref())?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
    }
}

fn logical_not_leaf(_py: Python<'_>, a: &Leaf) -> PyResult<Leaf> {
    let n = a.len;
    let mut out = new_bool_leaf(n);
    out.has_nulls = a.has_nulls;
    out.validity = a.validity.clone();
    let aa = match &a.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => return Err(internal_dtype_buffer_mismatch("logical_not", DType::Bool)) };
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
                return Err(shape_mismatch(
                    "comparison",
                    "requires identical ragged structure",
                    "ensure both operands share the same list offsets.",
                ));
            }
            let content = compare_layout(py, oa.content.as_ref(), ob.content.as_ref(), dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: oa.offsets.clone(), content: Box::new(content) }))
        }
        (Layout::UnionScalarList(ua), Layout::UnionScalarList(ub)) => {
            if ua.tags != ub.tags || ua.index != ub.index || ua.lists.offsets != ub.lists.offsets {
                return Err(shape_mismatch(
                    "comparison",
                    "requires identical union structure",
                    "ensure both operands share the same union tags, index, and list offsets.",
                ));
            }
            let scalars = compare_leaf(py, &ua.scalars, &ub.scalars, dt, op)?;
            let list_content = compare_layout(py, ua.lists.content.as_ref(), ub.lists.content.as_ref(), dt, op)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: ua.tags.clone(),
                index: ua.index.clone(),
                scalars,
                lists: ListOffset { offsets: ua.lists.offsets.clone(), content: Box::new(list_content) },
            }))
        }
        (Layout::OffsetView(va), Layout::OffsetView(vb)) => {
            if va.start != vb.start || va.stop != vb.stop || va.offsets != vb.offsets {
                return Err(shape_mismatch(
                    "comparison",
                    "requires identical offset views",
                    "ensure both operands use the same offset view bounds.",
                ));
            }
            let content = compare_layout(py, va.content.as_ref(), vb.content.as_ref(), dt, op)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: va.offsets.clone(),
                start: va.start,
                stop: vb.stop,
                content: Box::new(content),
            }))
        }
        (Layout::Indexed(ia), Layout::Indexed(ib)) => {
            if ia.index != ib.index {
                return Err(shape_mismatch(
                    "comparison",
                    "requires identical index vectors",
                    "ensure both operands use the same fancy index.",
                ));
            }
            let content = compare_layout(py, ia.content.as_ref(), ib.content.as_ref(), dt, op)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ia.index.clone(),
                content: Box::new(content),
            }))
        }
        _ => Err(layout_unsupported("comparison", "requires matching layouts")),
    }
}

fn compare_leaf(_py: Python<'_>, a: &Leaf, b: &Leaf, dt: DType, op: CmpOp) -> PyResult<Leaf> {
    if a.len != b.len {
        return Err(shape_mismatch(
            "comparison",
            format!("leaf length mismatch: {} vs {}", a.len, b.len),
            "ensure both operands have the same number of elements.",
        ));
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
        DType::String => {
            let aa = match &a.buffer { LeafBuffer::String(v) => v.as_slice(), _ => unreachable!() };
            let bb = match &b.buffer { LeafBuffer::String(v) => v.as_slice(), _ => unreachable!() };
            for i in 0..n {
                if !out_valid[i] { continue; }
                let ord = aa[i].cmp(&bb[i]);
                o[i] = match op {
                    CmpOp::Eq => (ord == std::cmp::Ordering::Equal) as u8,
                    CmpOp::Ne => (ord != std::cmp::Ordering::Equal) as u8,
                    CmpOp::Lt => (ord == std::cmp::Ordering::Less) as u8,
                    CmpOp::Le => (ord != std::cmp::Ordering::Greater) as u8,
                    CmpOp::Gt => (ord == std::cmp::Ordering::Greater) as u8,
                    CmpOp::Ge => (ord != std::cmp::Ordering::Less) as u8,
                };
            }
        }
        _ => return Err(dtype_unsupported("comparison", dt)),
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
        Layout::UnionScalarList(u) => {
            let scalars = pred_leaf(py, &u.scalars, dt, op)?;
            let list_content = pred_layout(py, u.lists.content.as_ref(), dt, op)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars,
                lists: ListOffset { offsets: u.lists.offsets.clone(), content: Box::new(list_content) },
            }))
        }
        Layout::OffsetView(v) => {
            let content = pred_layout(py, v.content.as_ref(), dt, op)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = pred_layout(py, ix.content.as_ref(), dt, op)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
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
        _ => return Err(dtype_unsupported("predicate", dt)),
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


