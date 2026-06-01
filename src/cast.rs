//! NumPy-style dtype promotion, casting rules, and layout-preserving casts.
//!
//! Casting modes (NumPy-aligned):
//! - **Safe** (default for ``astype``): widen integers/floats without loss; bool to numeric.
//! - **SameKind**: safe casts plus float narrowing (``float64`` → ``float32`` → ``float16``).
//! - **Unsafe**: all numeric casts; float→int truncates toward zero; overflow wraps for integers.
//! - **Promote**: casts required for binary ufunc promotion (subset used internally).
//!
//! String/char never cast to/from numeric types. ``char`` → ``string`` is safe.

use crate::dtype::DType;
use crate::error::{cast_not_allowed, internal_dtype_buffer_mismatch};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, UnionScalarList};
use half::f16;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Casting policy for :func:`astype` and internal promotion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastMode {
    Safe,
    SameKind,
    Unsafe,
    Promote,
}

impl CastMode {
    pub fn parse(s: &str) -> PyResult<Self> {
        match s {
            "safe" => Ok(Self::Safe),
            "same_kind" => Ok(Self::SameKind),
            "unsafe" => Ok(Self::Unsafe),
            "promote" => Ok(Self::Promote),
            _ => Err(PyValueError::new_err(format!(
                "Invalid casting mode '{s}'. Expected 'safe', 'same_kind', 'unsafe', or 'promote'.",
            ))),
        }
    }

    fn table(self) -> &'static [[u8; 12]; 12] {
        match self {
            CastMode::Safe => &CAN_CAST_SAFE,
            CastMode::SameKind => &CAN_CAST_SAME_KIND,
            CastMode::Unsafe => &CAN_CAST_UNSAFE,
            CastMode::Promote => &CAN_CAST_PROMOTE,
        }
    }
}

// Numeric dtype indices (bool … float64). Generated from NumPy 2.x ``can_cast`` / ``promote_types``.
const CAN_CAST_SAFE: [[u8; 12]; 12] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
];
const CAN_CAST_SAME_KIND: [[u8; 12]; 12] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
];
const CAN_CAST_UNSAFE: [[u8; 12]; 12] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];
const CAN_CAST_PROMOTE: [[u8; 12]; 12] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
];
const PROMOTE_TYPES: [[u8; 12]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [1, 1, 2, 3, 4, 2, 3, 4, 11, 9, 10, 11],
    [2, 2, 2, 3, 4, 2, 3, 4, 11, 10, 10, 11],
    [3, 3, 3, 3, 4, 3, 3, 4, 11, 11, 11, 11],
    [4, 4, 4, 4, 4, 4, 4, 4, 11, 11, 11, 11],
    [5, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [6, 3, 3, 3, 4, 6, 6, 7, 8, 10, 10, 11],
    [7, 4, 4, 4, 4, 7, 7, 7, 8, 11, 11, 11],
    [8, 11, 11, 11, 11, 8, 8, 8, 8, 11, 11, 11],
    [9, 9, 10, 11, 11, 9, 10, 11, 11, 9, 10, 11],
    [10, 10, 10, 11, 11, 10, 10, 11, 11, 10, 10, 11],
    [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
];

fn numeric_index(dt: DType) -> Option<usize> {
    Some(match dt {
        DType::Bool => 0,
        DType::Int8 => 1,
        DType::Int16 => 2,
        DType::Int32 => 3,
        DType::Int64 => 4,
        DType::UInt8 => 5,
        DType::UInt16 => 6,
        DType::UInt32 => 7,
        DType::UInt64 => 8,
        DType::Float16 => 9,
        DType::Float32 => 10,
        DType::Float64 => 11,
        DType::Char | DType::String => return None,
    })
}

fn dtype_from_numeric_index(ix: usize) -> DType {
    match ix {
        0 => DType::Bool,
        1 => DType::Int8,
        2 => DType::Int16,
        3 => DType::Int32,
        4 => DType::Int64,
        5 => DType::UInt8,
        6 => DType::UInt16,
        7 => DType::UInt32,
        8 => DType::UInt64,
        9 => DType::Float16,
        10 => DType::Float32,
        11 => DType::Float64,
        _ => unreachable!(),
    }
}

/// Whether ``from`` can be cast to ``to`` under ``mode``.
pub fn can_cast(from: DType, to: DType, mode: CastMode) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        (DType::Char, DType::String) => true,
        (DType::String, DType::Char) => matches!(mode, CastMode::SameKind | CastMode::Unsafe),
        (DType::Char, DType::Char) | (DType::String, DType::String) => false,
        (DType::Char | DType::String, _) | (_, DType::Char | DType::String) => false,
        _ => {
            if let (Some(fi), Some(ti)) = (numeric_index(from), numeric_index(to)) {
                mode.table()[fi][ti] != 0
            } else {
                false
            }
        }
    }
}

/// NumPy ``promote_types`` for numeric dtypes; string/char only with themselves.
pub fn promote_binary(a: DType, b: DType) -> PyResult<DType> {
    if a == b {
        return Ok(a);
    }
    if a == DType::String || b == DType::String {
        return Err(PyValueError::new_err(
            "Cannot promote string dtype with other dtypes.",
        ));
    }
    if a == DType::Char || b == DType::Char {
        return Err(PyValueError::new_err(
            "Cannot promote char dtype with other dtypes.",
        ));
    }
    let ai = numeric_index(a).ok_or_else(|| {
        PyValueError::new_err(format!("Cannot promote dtype {}.", a.name()))
    })?;
    let bi = numeric_index(b).ok_or_else(|| {
        PyValueError::new_err(format!("Cannot promote dtype {}.", b.name()))
    })?;
    Ok(dtype_from_numeric_index(PROMOTE_TYPES[ai][bi] as usize))
}

pub fn promote_types(a: DType, b: DType) -> PyResult<DType> {
    promote_binary(a, b)
}

pub fn cast_array_with_mode(arr: &GrumpyArray, to: DType, mode: CastMode) -> PyResult<GrumpyArray> {
    if arr.dtype == to {
        return Ok(arr.clone());
    }
    if !can_cast(arr.dtype, to, mode) {
        return Err(cast_not_allowed(arr.dtype, to, mode_name(mode)));
    }
    let layout = cast_layout(&arr.layout, arr.dtype, to, mode)?;
    Ok(GrumpyArray { dtype: to, layout })
}

pub fn cast_array_pair(a: &GrumpyArray, b: &GrumpyArray) -> PyResult<(GrumpyArray, GrumpyArray)> {
    let out = promote_binary(a.dtype, b.dtype)?;
    let aa = if a.dtype == out {
        a.clone()
    } else {
        cast_array_with_mode(a, out, CastMode::Promote)?
    };
    let bb = if b.dtype == out {
        b.clone()
    } else {
        cast_array_with_mode(b, out, CastMode::Promote)?
    };
    Ok((aa, bb))
}

fn mode_name(mode: CastMode) -> &'static str {
    match mode {
        CastMode::Safe => "safe",
        CastMode::SameKind => "same_kind",
        CastMode::Unsafe => "unsafe",
        CastMode::Promote => "promote",
    }
}

fn cast_layout(layout: &Layout, from: DType, to: DType, mode: CastMode) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => Ok(Layout::Leaf(cast_leaf(l, from, to, mode)?)),
        Layout::ListOffset(lo) => {
            let content = cast_layout(lo.content.as_ref(), from, to, mode)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            let content = cast_layout(v.content.as_ref(), from, to, mode)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = cast_layout(ix.content.as_ref(), from, to, mode)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(u) => {
            let scalars = cast_leaf(&u.scalars, from, to, mode)?;
            let list_content = cast_layout(u.lists.content.as_ref(), from, to, mode)?;
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

fn cast_leaf(leaf: &Leaf, from: DType, to: DType, mode: CastMode) -> PyResult<Leaf> {
    if from == to {
        return Ok(leaf.clone());
    }
    let n = leaf.len;
    let mut out = Leaf::new(to);
    out.len = n;
    out.validity = leaf.validity.clone();
    out.has_nulls = leaf.has_nulls;

    if from == DType::Char && to == DType::String {
        let src = match &leaf.buffer {
            LeafBuffer::Char(v) => v.as_slice(),
            _ => return Err(internal_dtype_buffer_mismatch("cast", from)),
        };
        let mut dst = Vec::with_capacity(n);
        for i in 0..n {
            if leaf.validity[i] {
                let c = char::from_u32(src[i])
                    .ok_or_else(|| PyValueError::new_err("Invalid char value during cast."))?;
                dst.push(c.to_string());
            } else {
                dst.push(String::new());
            }
        }
        out.buffer = LeafBuffer::String(Arc::new(dst));
        return Ok(out);
    }

    if from == DType::String && to == DType::Char {
        let src = match &leaf.buffer {
            LeafBuffer::String(v) => v.as_slice(),
            _ => return Err(internal_dtype_buffer_mismatch("cast", from)),
        };
        let mut dst = vec![0u32; n];
        for i in 0..n {
            if !leaf.validity[i] {
                continue;
            }
            let s = &src[i];
            let mut it = s.chars();
            let c = it.next().ok_or_else(|| {
                PyValueError::new_err("Cannot cast empty string to char.")
            })?;
            if it.next().is_some() {
                return Err(PyValueError::new_err(
                    "Cannot cast multi-character string to char.",
                ));
            }
            dst[i] = c as u32;
        }
        out.buffer = LeafBuffer::Char(Arc::new(dst));
        return Ok(out);
    }

    out.buffer = cast_numeric_buffer(&leaf.buffer, from, to, &leaf.validity, n, mode)?;
    Ok(out)
}

fn cast_numeric_buffer(
    src_buf: &LeafBuffer,
    from: DType,
    to: DType,
    validity: &bitvec::slice::BitSlice<u8, bitvec::order::Lsb0>,
    n: usize,
    mode: CastMode,
) -> PyResult<LeafBuffer> {
    let mut out_buf = LeafBuffer::new(to);
    out_buf.reserve(n);
    for i in 0..n {
        if !validity[i] {
            push_default(&mut out_buf, to);
            continue;
        }
        let scalar = read_scalar(src_buf, from, i)?;
        write_scalar(&mut out_buf, to, scalar, mode)?;
    }
    Ok(out_buf)
}

enum ScalarValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
}

fn read_scalar(buf: &LeafBuffer, dt: DType, i: usize) -> PyResult<ScalarValue> {
    Ok(match (buf, dt) {
        (LeafBuffer::Bool(v), DType::Bool) => ScalarValue::Bool(v[i] != 0),
        (LeafBuffer::I8(v), DType::Int8) => ScalarValue::I64(v[i] as i64),
        (LeafBuffer::I16(v), DType::Int16) => ScalarValue::I64(v[i] as i64),
        (LeafBuffer::I32(v), DType::Int32) => ScalarValue::I64(v[i] as i64),
        (LeafBuffer::I64(v), DType::Int64) => ScalarValue::I64(v[i]),
        (LeafBuffer::U8(v), DType::UInt8) => ScalarValue::U64(v[i] as u64),
        (LeafBuffer::U16(v), DType::UInt16) => ScalarValue::U64(v[i] as u64),
        (LeafBuffer::U32(v), DType::UInt32) => ScalarValue::U64(v[i] as u64),
        (LeafBuffer::U64(v), DType::UInt64) => ScalarValue::U64(v[i]),
        (LeafBuffer::F16(v), DType::Float16) => ScalarValue::F64(f16::from_bits(v[i]).to_f64()),
        (LeafBuffer::F32(v), DType::Float32) => ScalarValue::F64(v[i] as f64),
        (LeafBuffer::F64(v), DType::Float64) => ScalarValue::F64(v[i]),
        _ => return Err(internal_dtype_buffer_mismatch("cast", dt)),
    })
}

fn write_scalar(buf: &mut LeafBuffer, dt: DType, scalar: ScalarValue, mode: CastMode) -> PyResult<()> {
    match (buf, dt, scalar) {
        (LeafBuffer::Bool(vb), DType::Bool, ScalarValue::Bool(b)) => Arc::make_mut(vb).push(if b { 1 } else { 0 }),
        (LeafBuffer::Bool(vb), DType::Bool, ScalarValue::I64(v)) => {
            Arc::make_mut(vb).push(if v != 0 { 1 } else { 0 })
        }
        (LeafBuffer::Bool(vb), DType::Bool, ScalarValue::U64(v)) => {
            Arc::make_mut(vb).push(if v != 0 { 1 } else { 0 })
        }
        (LeafBuffer::Bool(vb), DType::Bool, ScalarValue::F64(v)) => {
            Arc::make_mut(vb).push(if v != 0.0 { 1 } else { 0 })
        }
        (LeafBuffer::I8(vb), DType::Int8, s) => {
            Arc::make_mut(vb).push(scalar_to_i8(s, mode)?);
        }
        (LeafBuffer::I16(vb), DType::Int16, s) => {
            Arc::make_mut(vb).push(scalar_to_i16(s, mode)?);
        }
        (LeafBuffer::I32(vb), DType::Int32, s) => {
            Arc::make_mut(vb).push(scalar_to_i32(s, mode)?);
        }
        (LeafBuffer::I64(vb), DType::Int64, s) => {
            Arc::make_mut(vb).push(scalar_to_i64(s, mode)?);
        }
        (LeafBuffer::U8(vb), DType::UInt8, s) => {
            Arc::make_mut(vb).push(scalar_to_u8(s, mode)?);
        }
        (LeafBuffer::U16(vb), DType::UInt16, s) => {
            Arc::make_mut(vb).push(scalar_to_u16(s, mode)?);
        }
        (LeafBuffer::U32(vb), DType::UInt32, s) => {
            Arc::make_mut(vb).push(scalar_to_u32(s, mode)?);
        }
        (LeafBuffer::U64(vb), DType::UInt64, s) => {
            Arc::make_mut(vb).push(scalar_to_u64_value(s, mode)?);
        }
        (LeafBuffer::F16(vb), DType::Float16, s) => {
            let v = scalar_to_f64(s, mode)?;
            Arc::make_mut(vb).push(f16::from_f64(v).to_bits());
        }
        (LeafBuffer::F32(vb), DType::Float32, s) => {
            Arc::make_mut(vb).push(scalar_to_f64(s, mode)? as f32);
        }
        (LeafBuffer::F64(vb), DType::Float64, s) => {
            Arc::make_mut(vb).push(scalar_to_f64(s, mode)?);
        }
        _ => return Err(internal_dtype_buffer_mismatch("cast", dt)),
    }
    Ok(())
}

fn scalar_to_f64(s: ScalarValue, _mode: CastMode) -> PyResult<f64> {
    Ok(match s {
        ScalarValue::Bool(b) => {
            if b {
                1.0
            } else {
                0.0
            }
        }
        ScalarValue::I64(v) => v as f64,
        ScalarValue::U64(v) => v as f64,
        ScalarValue::F64(v) => v,
    })
}

fn scalar_to_i64(s: ScalarValue, mode: CastMode) -> PyResult<i64> {
    match s {
        ScalarValue::Bool(b) => Ok(if b { 1 } else { 0 }),
        ScalarValue::I64(v) => Ok(v),
        ScalarValue::U64(v) => {
            if v > i64::MAX as u64 && !matches!(mode, CastMode::Unsafe) {
                return Err(PyValueError::new_err(
                    "Cast overflow: unsigned value exceeds int64 range.",
                ));
            }
            Ok(v as i64)
        }
        ScalarValue::F64(v) => float_to_int(v, mode),
    }
}

fn scalar_to_i8(s: ScalarValue, mode: CastMode) -> PyResult<i8> {
    cast_to_i8(scalar_to_i64(s, mode)?, mode)
}
fn scalar_to_i16(s: ScalarValue, mode: CastMode) -> PyResult<i16> {
    cast_to_i16(scalar_to_i64(s, mode)?, mode)
}
fn scalar_to_i32(s: ScalarValue, mode: CastMode) -> PyResult<i32> {
    cast_to_i32(scalar_to_i64(s, mode)?, mode)
}
fn scalar_to_u8(s: ScalarValue, mode: CastMode) -> PyResult<u8> {
    Ok(cast_to_u8(scalar_to_u64_value(s, mode)?, mode)?)
}
fn scalar_to_u16(s: ScalarValue, mode: CastMode) -> PyResult<u16> {
    Ok(cast_to_u16(scalar_to_u64_value(s, mode)?, mode)?)
}
fn scalar_to_u32(s: ScalarValue, mode: CastMode) -> PyResult<u32> {
    Ok(cast_to_u32(scalar_to_u64_value(s, mode)?, mode)?)
}

fn scalar_to_u64_value(s: ScalarValue, mode: CastMode) -> PyResult<u64> {
    match s {
        ScalarValue::Bool(b) => Ok(if b { 1 } else { 0 }),
        ScalarValue::I64(v) => {
            if v < 0 && !matches!(mode, CastMode::Unsafe) {
                return Err(PyValueError::new_err(
                    "Cannot cast negative value to unsigned integer.",
                ));
            }
            Ok(v as u64)
        }
        ScalarValue::U64(v) => Ok(v),
        ScalarValue::F64(v) => float_to_uint(v, mode),
    }
}

fn push_default(buf: &mut LeafBuffer, dt: DType) {
    match (buf, dt) {
        (LeafBuffer::I8(v), DType::Int8) => Arc::make_mut(v).push(0),
        (LeafBuffer::I16(v), DType::Int16) => Arc::make_mut(v).push(0),
        (LeafBuffer::I32(v), DType::Int32) => Arc::make_mut(v).push(0),
        (LeafBuffer::I64(v), DType::Int64) => Arc::make_mut(v).push(0),
        (LeafBuffer::U8(v), DType::UInt8) => Arc::make_mut(v).push(0),
        (LeafBuffer::U16(v), DType::UInt16) => Arc::make_mut(v).push(0),
        (LeafBuffer::U32(v), DType::UInt32) => Arc::make_mut(v).push(0),
        (LeafBuffer::U64(v), DType::UInt64) => Arc::make_mut(v).push(0),
        (LeafBuffer::F16(v), DType::Float16) => Arc::make_mut(v).push(0),
        (LeafBuffer::F32(v), DType::Float32) => Arc::make_mut(v).push(0.0),
        (LeafBuffer::F64(v), DType::Float64) => Arc::make_mut(v).push(0.0),
        (LeafBuffer::Bool(v), DType::Bool) => Arc::make_mut(v).push(0),
        _ => {}
    }
}

fn float_to_int(v: f64, mode: CastMode) -> PyResult<i64> {
    if v.is_nan() {
        return Err(PyValueError::new_err("Cannot cast NaN to integer."));
    }
    if v.is_infinite() {
        return Err(PyValueError::new_err("Cannot cast infinity to integer."));
    }
    let truncated = v.trunc();
    if matches!(mode, CastMode::Safe | CastMode::Promote) && truncated.fract() != 0.0 {
        return Err(PyValueError::new_err(
            "Cannot safely cast non-integral float to integer.",
        ));
    }
    Ok(truncated as i64)
}

fn float_to_uint(v: f64, mode: CastMode) -> PyResult<u64> {
    let iv = float_to_int(v, mode)?;
    if iv < 0 && !matches!(mode, CastMode::Unsafe) {
        return Err(PyValueError::new_err(
            "Cannot cast negative value to unsigned integer.",
        ));
    }
    Ok(iv as u64)
}

fn cast_to_i8(v: i64, mode: CastMode) -> PyResult<i8> {
    check_int_range(v, i8::MIN as i64, i8::MAX as i64, mode)?;
    Ok(v as i8)
}

fn cast_to_i16(v: i64, mode: CastMode) -> PyResult<i16> {
    check_int_range(v, i16::MIN as i64, i16::MAX as i64, mode)?;
    Ok(v as i16)
}

fn cast_to_i32(v: i64, mode: CastMode) -> PyResult<i32> {
    check_int_range(v, i32::MIN as i64, i32::MAX as i64, mode)?;
    Ok(v as i32)
}

fn cast_to_u8(v: u64, mode: CastMode) -> PyResult<u8> {
    check_uint_range(v, u8::MAX as u64, mode)?;
    Ok(v as u8)
}

fn cast_to_u16(v: u64, mode: CastMode) -> PyResult<u16> {
    check_uint_range(v, u16::MAX as u64, mode)?;
    Ok(v as u16)
}

fn cast_to_u32(v: u64, mode: CastMode) -> PyResult<u32> {
    check_uint_range(v, u32::MAX as u64, mode)?;
    Ok(v as u32)
}

fn check_int_range(v: i64, min: i64, max: i64, mode: CastMode) -> PyResult<()> {
    if matches!(mode, CastMode::Unsafe) {
        return Ok(());
    }
    if v < min || v > max {
        return Err(PyValueError::new_err(format!(
            "Cast overflow: value {v} is out of bounds [{min}, {max}].",
        )));
    }
    Ok(())
}

fn check_uint_range(v: u64, max: u64, mode: CastMode) -> PyResult<()> {
    if matches!(mode, CastMode::Unsafe) {
        return Ok(());
    }
    if v > max {
        return Err(PyValueError::new_err(format!(
            "Cast overflow: value {v} exceeds maximum {max}.",
        )));
    }
    Ok(())
}
