//! NumPy-style dtype promotion and array casting.

use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// NumPy ufunc-style binary result type (numeric dtypes only).
pub fn promote_binary(a: DType, b: DType) -> PyResult<DType> {
    if a == b {
        return Ok(a);
    }
    if a == DType::String || b == DType::String || a == DType::Char || b == DType::Char {
        return Err(PyValueError::new_err(
            "Cannot promote string/char dtypes with other dtypes.",
        ));
    }
    if a == DType::Bool && b == DType::Bool {
        return Ok(DType::Bool);
    }
    let rank = |dt: DType| -> u8 {
        match dt {
            DType::Bool => 0,
            DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 3,
            DType::Int64 | DType::UInt64 | DType::Float64 => 4,
            DType::Char | DType::String => 255,
        }
    };
    let is_float = |dt: DType| {
        matches!(
            dt,
            DType::Float16 | DType::Float32 | DType::Float64
        )
    };
    let is_int = |dt: DType| {
        matches!(
            dt,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
                | DType::Bool
        )
    };
    if is_float(a) || is_float(b) {
        return Ok(if rank(a) >= rank(b) { a } else { b });
    }
    if is_int(a) && is_int(b) {
        return Ok(if rank(a) >= rank(b) { a } else { b });
    }
    Err(PyValueError::new_err(format!(
        "Cannot promote dtypes {} and {}.",
        a.name(),
        b.name()
    )))
}

pub fn cast_array(arr: &GrumpyArray, to: DType) -> PyResult<GrumpyArray> {
    if arr.dtype == to {
        return Ok(arr.clone());
    }
    let layout = cast_layout(&arr.layout, arr.dtype, to)?;
    Ok(GrumpyArray { dtype: to, layout })
}

pub fn cast_array_pair(a: &GrumpyArray, b: &GrumpyArray) -> PyResult<(GrumpyArray, GrumpyArray)> {
    let out = promote_binary(a.dtype, b.dtype)?;
    let aa = if a.dtype == out {
        a.clone()
    } else {
        cast_array(a, out)?
    };
    let bb = if b.dtype == out {
        b.clone()
    } else {
        cast_array(b, out)?
    };
    Ok((aa, bb))
}

fn cast_layout(layout: &Layout, from: DType, to: DType) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => Ok(Layout::Leaf(cast_leaf(l, from, to)?)),
        Layout::ListOffset(lo) => {
            let content = cast_layout(lo.content.as_ref(), from, to)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            let content = cast_layout(v.content.as_ref(), from, to)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = cast_layout(ix.content.as_ref(), from, to)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(u) => {
            let scalars = cast_leaf(&u.scalars, from, to)?;
            let list_content = cast_layout(u.lists.content.as_ref(), from, to)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
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

fn cast_leaf(leaf: &Leaf, from: DType, to: DType) -> PyResult<Leaf> {
    if from == to {
        return Ok(leaf.clone());
    }
    let n = leaf.len;
    let mut out = Leaf::new(to);
    out.len = n;
    out.validity = leaf.validity.clone();
    out.has_nulls = leaf.has_nulls;
    out.buffer = match (from, to) {
        (DType::Int32, DType::Float64) => {
            let src = match &leaf.buffer {
                LeafBuffer::I32(v) => v.as_slice(),
                _ => return Err(PyValueError::new_err("Internal cast dtype mismatch.")),
            };
            let mut dst = vec![0f64; n];
            for i in 0..n {
                if leaf.validity[i] {
                    dst[i] = src[i] as f64;
                }
            }
            LeafBuffer::F64(Arc::new(dst))
        }
        (DType::Int64, DType::Float64) => {
            let src = match &leaf.buffer {
                LeafBuffer::I64(v) => v.as_slice(),
                _ => return Err(PyValueError::new_err("Internal cast dtype mismatch.")),
            };
            let mut dst = vec![0f64; n];
            for i in 0..n {
                if leaf.validity[i] {
                    dst[i] = src[i] as f64;
                }
            }
            LeafBuffer::F64(Arc::new(dst))
        }
        (DType::Float32, DType::Float64) => {
            let src = match &leaf.buffer {
                LeafBuffer::F32(v) => v.as_slice(),
                _ => return Err(PyValueError::new_err("Internal cast dtype mismatch.")),
            };
            let mut dst = vec![0f64; n];
            for i in 0..n {
                if leaf.validity[i] {
                    dst[i] = src[i] as f64;
                }
            }
            LeafBuffer::F64(Arc::new(dst))
        }
        (DType::Int8, DType::Int32)
        | (DType::Int16, DType::Int32)
        | (DType::UInt8, DType::UInt32)
        | (DType::UInt16, DType::UInt32)
        | (DType::Bool, DType::Int32) => cast_int_widen(leaf, from, to, 4)?,
        (DType::Int32, DType::Int64) | (DType::UInt32, DType::UInt64) => {
            cast_int_widen(leaf, from, to, 8)?
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Cast from {} to {} is not implemented yet.",
                from.name(),
                to.name()
            )))
        }
    };
    Ok(out)
}

fn cast_int_widen(leaf: &Leaf, _from: DType, to: DType, width: usize) -> PyResult<LeafBuffer> {
    let n = leaf.len;
    match (width, to) {
        (4, DType::Int32) => {
            let mut dst = vec![0i32; n];
            for i in 0..n {
                if !leaf.validity[i] {
                    continue;
                }
                dst[i] = read_int_as_i64(&leaf.buffer, i)? as i32;
            }
            Ok(LeafBuffer::I32(Arc::new(dst)))
        }
        (8, DType::Int64) => {
            let mut dst = vec![0i64; n];
            for i in 0..n {
                if !leaf.validity[i] {
                    continue;
                }
                dst[i] = read_int_as_i64(&leaf.buffer, i)?;
            }
            Ok(LeafBuffer::I64(Arc::new(dst)))
        }
        (8, DType::UInt64) => {
            let mut dst = vec![0u64; n];
            for i in 0..n {
                if !leaf.validity[i] {
                    continue;
                }
                dst[i] = read_int_as_i64(&leaf.buffer, i)? as u64;
            }
            Ok(LeafBuffer::U64(Arc::new(dst)))
        }
        _ => Err(PyValueError::new_err("Internal cast error.")),
    }
}

fn read_int_as_i64(buf: &LeafBuffer, i: usize) -> PyResult<i64> {
    Ok(match buf {
        LeafBuffer::I8(v) => v[i] as i64,
        LeafBuffer::I16(v) => v[i] as i64,
        LeafBuffer::I32(v) => v[i] as i64,
        LeafBuffer::I64(v) => v[i],
        LeafBuffer::U8(v) => v[i] as i64,
        LeafBuffer::U16(v) => v[i] as i64,
        LeafBuffer::U32(v) => v[i] as i64,
        LeafBuffer::U64(v) => v[i] as i64,
        LeafBuffer::Bool(v) => v[i] as i64,
        _ => return Err(PyValueError::new_err("Expected integer leaf for cast.")),
    })
}
