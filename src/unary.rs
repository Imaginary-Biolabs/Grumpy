use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Log10,
    Log2,
    Sqrt,
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Reciprocal,
    Angle,
}

pub fn unary(py: Python<'_>, arr: &GrumpyArray, op: UnaryOp) -> PyResult<GrumpyArray> {
    let out_dt = unary_out_dtype(arr.dtype, op)?;
    let out_layout = unary_layout(py, &arr.layout, arr.dtype, out_dt, op)?;
    Ok(GrumpyArray { dtype: out_dt, layout: out_layout })
}

fn unary_out_dtype(dt: DType, op: UnaryOp) -> PyResult<DType> {
    match op {
        UnaryOp::Abs | UnaryOp::Sign | UnaryOp::Reciprocal | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round => Ok(dt),
        UnaryOp::Angle => match dt {
            DType::Float32 | DType::Float64 => Ok(dt),
            DType::Int32 | DType::Int64 => Ok(DType::Float64),
            _ => Err(PyValueError::new_err("angle only supported for numeric dtypes.")),
        },
        _ => match dt {
            DType::Float32 | DType::Float64 => Ok(dt),
            DType::Int32 | DType::Int64 => Ok(DType::Float64),
            _ => Err(PyValueError::new_err("Unary op only supported for numeric dtypes.")),
        },
    }
}

fn unary_layout(py: Python<'_>, layout: &Layout, in_dt: DType, out_dt: DType, op: UnaryOp) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => Ok(Layout::Leaf(unary_leaf(py, l, in_dt, out_dt, op)?)),
        Layout::ListOffset(lo) => {
            let content = unary_layout(py, lo.content.as_ref(), in_dt, out_dt, op)?;
            Ok(Layout::ListOffset(ListOffset { offsets: lo.offsets.clone(), content: Box::new(content) }))
        }
        Layout::OffsetView(v) => {
            // Apply to view's content, preserve view offsets.
            let content = unary_layout(py, v.content.as_ref(), in_dt, out_dt, op)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            // Materialize by applying to content, keep index wrapper.
            let content = unary_layout(py, ix.content.as_ref(), in_dt, out_dt, op)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(u) => {
            let scalars = unary_leaf(py, &u.scalars, in_dt, out_dt, op)?;
            let list_content = unary_layout(py, u.lists.content.as_ref(), in_dt, out_dt, op)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars,
                lists: ListOffset { offsets: u.lists.offsets.clone(), content: Box::new(list_content) },
            }))
        }
    }
}

fn new_out_leaf(n: usize, out_dt: DType) -> PyResult<Leaf> {
    let mut out = Leaf::new(out_dt);
    out.len = n;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out.has_nulls = false;
    out.buffer = match out_dt {
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        _ => return Err(PyValueError::new_err("Unsupported dtype for unary op.")),
    };
    Ok(out)
}

fn unary_leaf(_py: Python<'_>, leaf: &Leaf, in_dt: DType, out_dt: DType, op: UnaryOp) -> PyResult<Leaf> {
    let n = leaf.len;
    let mut out = new_out_leaf(n, out_dt)?;
    out.has_nulls = leaf.has_nulls;
    out.validity = leaf.validity.clone();

    // Fast all-valid branch.
    if !leaf.has_nulls {
        match (in_dt, out_dt, op, &leaf.buffer, &mut out.buffer) {
            (DType::Float64, DType::Float64, UnaryOp::Sin, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].sin(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Cos, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].cos(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Tan, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].tan(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Exp, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].exp(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Log, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].ln(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Log10, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].log10(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Log2, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].log2(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Sqrt, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].sqrt(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Abs, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].abs(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Sign, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i];
                    o[i] = if x.is_nan() { f64::NAN } else if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
                }
            }
            (DType::Float64, DType::Float64, UnaryOp::Floor, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].floor(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Ceil, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].ceil(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Round, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].round(); }
            }
            (DType::Float64, DType::Float64, UnaryOp::Reciprocal, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = 1.0 / a[i]; }
            }
            (DType::Float64, DType::Float64, UnaryOp::Angle, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i];
                    o[i] = if x.is_nan() { f64::NAN } else if x < 0.0 { std::f64::consts::PI } else { 0.0 };
                }
            }
            (DType::Float32, DType::Float32, UnaryOp::Sin, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].sin(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Cos, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].cos(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Tan, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].tan(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Exp, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].exp(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Log, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].ln(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Log10, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].log10(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Log2, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].log2(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Sqrt, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].sqrt(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Abs, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].abs(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Sign, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i];
                    o[i] = if x.is_nan() { f32::NAN } else if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
                }
            }
            (DType::Float32, DType::Float32, UnaryOp::Floor, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].floor(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Ceil, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].ceil(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Round, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].round(); }
            }
            (DType::Float32, DType::Float32, UnaryOp::Reciprocal, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = 1.0 / a[i]; }
            }
            (DType::Float32, DType::Float32, UnaryOp::Angle, LeafBuffer::F32(a), LeafBuffer::F32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i];
                    o[i] = if x.is_nan() { f32::NAN } else if x < 0.0 { std::f32::consts::PI } else { 0.0 };
                }
            }
            // int -> float64 for trig/log/exp/sqrt
            (DType::Int32, DType::Float64, uop, LeafBuffer::I32(a), LeafBuffer::F64(o))
                if matches!(uop, UnaryOp::Sin|UnaryOp::Cos|UnaryOp::Tan|UnaryOp::Exp|UnaryOp::Log|UnaryOp::Log10|UnaryOp::Log2|UnaryOp::Sqrt) =>
            {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i] as f64;
                    o[i] = match uop {
                        UnaryOp::Sin => x.sin(),
                        UnaryOp::Cos => x.cos(),
                        UnaryOp::Tan => x.tan(),
                        UnaryOp::Exp => x.exp(),
                        UnaryOp::Log => x.ln(),
                        UnaryOp::Log10 => x.log10(),
                        UnaryOp::Log2 => x.log2(),
                        UnaryOp::Sqrt => x.sqrt(),
                        _ => unreachable!(),
                    };
                }
            }
            (DType::Int64, DType::Float64, uop, LeafBuffer::I64(a), LeafBuffer::F64(o))
                if matches!(uop, UnaryOp::Sin|UnaryOp::Cos|UnaryOp::Tan|UnaryOp::Exp|UnaryOp::Log|UnaryOp::Log10|UnaryOp::Log2|UnaryOp::Sqrt) =>
            {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i] as f64;
                    o[i] = match uop {
                        UnaryOp::Sin => x.sin(),
                        UnaryOp::Cos => x.cos(),
                        UnaryOp::Tan => x.tan(),
                        UnaryOp::Exp => x.exp(),
                        UnaryOp::Log => x.ln(),
                        UnaryOp::Log10 => x.log10(),
                        UnaryOp::Log2 => x.log2(),
                        UnaryOp::Sqrt => x.sqrt(),
                        _ => unreachable!(),
                    };
                }
            }
            // int abs/sign/round/ceil/floor are identity-ish, reciprocal uses integer reciprocal like NumPy.
            (DType::Int32, DType::Int32, UnaryOp::Abs, LeafBuffer::I32(a), LeafBuffer::I32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].wrapping_abs(); }
            }
            (DType::Int64, DType::Int64, UnaryOp::Abs, LeafBuffer::I64(a), LeafBuffer::I64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = a[i].wrapping_abs(); }
            }
            (DType::Int32, DType::Int32, UnaryOp::Sign, LeafBuffer::I32(a), LeafBuffer::I32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = if a[i] > 0 { 1 } else if a[i] < 0 { -1 } else { 0 }; }
            }
            (DType::Int64, DType::Int64, UnaryOp::Sign, LeafBuffer::I64(a), LeafBuffer::I64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = if a[i] > 0 { 1 } else if a[i] < 0 { -1 } else { 0 }; }
            }
            (DType::Int32, DType::Int32, UnaryOp::Reciprocal, LeafBuffer::I32(a), LeafBuffer::I32(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i];
                    o[i] = if x == 0 { 0 } else { 1i32 / x };
                }
            }
            (DType::Int64, DType::Int64, UnaryOp::Reciprocal, LeafBuffer::I64(a), LeafBuffer::I64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n {
                    let x = a[i];
                    o[i] = if x == 0 { 0 } else { 1i64 / x };
                }
            }
            // floor/ceil/round on ints: identity
            (DType::Int32, DType::Int32, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round, LeafBuffer::I32(a), LeafBuffer::I32(o)) => {
                let o = Arc::make_mut(o);
                o[..n].copy_from_slice(&a[..n]);
            }
            (DType::Int64, DType::Int64, UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round, LeafBuffer::I64(a), LeafBuffer::I64(o)) => {
                let o = Arc::make_mut(o);
                o[..n].copy_from_slice(&a[..n]);
            }
            // angle for ints -> float64
            (DType::Int32, DType::Float64, UnaryOp::Angle, LeafBuffer::I32(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = if a[i] < 0 { std::f64::consts::PI } else { 0.0 }; }
            }
            (DType::Int64, DType::Float64, UnaryOp::Angle, LeafBuffer::I64(a), LeafBuffer::F64(o)) => {
                let o = Arc::make_mut(o);
                for i in 0..n { o[i] = if a[i] < 0 { std::f64::consts::PI } else { 0.0 }; }
            }
            _ => return Err(PyValueError::new_err("Unary op not implemented for this dtype.")),
        }
        return Ok(out);
    }

    // Null-aware branch: only implemented for float64 and int32/int64 for now.
    match (in_dt, out_dt, op, &leaf.buffer, &mut out.buffer) {
        (DType::Float64, DType::Float64, uop, LeafBuffer::F64(a), LeafBuffer::F64(o)) => {
            let o = Arc::make_mut(o);
            for i in 0..n {
                if !leaf.validity[i] {
                    o[i] = 0.0;
                    continue;
                }
                let x = a[i];
                o[i] = match uop {
                    UnaryOp::Sin => x.sin(),
                    UnaryOp::Cos => x.cos(),
                    UnaryOp::Tan => x.tan(),
                    UnaryOp::Exp => x.exp(),
                    UnaryOp::Log => x.ln(),
                    UnaryOp::Log10 => x.log10(),
                    UnaryOp::Log2 => x.log2(),
                    UnaryOp::Sqrt => x.sqrt(),
                    UnaryOp::Abs => x.abs(),
                    UnaryOp::Sign => if x.is_nan() { f64::NAN } else if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
                    UnaryOp::Floor => x.floor(),
                    UnaryOp::Ceil => x.ceil(),
                    UnaryOp::Round => x.round(),
                    UnaryOp::Reciprocal => 1.0 / x,
                    UnaryOp::Angle => if x.is_nan() { f64::NAN } else if x < 0.0 { std::f64::consts::PI } else { 0.0 },
                };
            }
        }
        (DType::Int32, DType::Int32, UnaryOp::Abs, LeafBuffer::I32(a), LeafBuffer::I32(o)) => {
            let o = Arc::make_mut(o);
            for i in 0..n { o[i] = if leaf.validity[i] { a[i].wrapping_abs() } else { 0 }; }
        }
        (DType::Int64, DType::Int64, UnaryOp::Abs, LeafBuffer::I64(a), LeafBuffer::I64(o)) => {
            let o = Arc::make_mut(o);
            for i in 0..n { o[i] = if leaf.validity[i] { a[i].wrapping_abs() } else { 0 }; }
        }
        _ => return Err(PyValueError::new_err("Null-aware unary op not implemented for this dtype.")),
    }
    Ok(out)
}


