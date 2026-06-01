use crate::dtype::{is_sequence_like, DType};
use bitvec::prelude::*;
use half::f16;
use numpy::PyUntypedArrayMethods;
use crate::error::{
    arg_invalid, concat_incompatible, dtype_mismatch, dtype_unsupported, index_out_of_bounds,
    index_out_of_bounds_simple, internal, internal_dtype_buffer_mismatch, invalid_slice_range,
    layout_unsupported, shape_mismatch, unsupported,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PySequence};
use pyo3::types::PyListMethods;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Layout {
    Leaf(Leaf),
    ListOffset(ListOffset),
    // View: select elements by index without materializing/copying the underlying content.
    Indexed(Indexed),
    // View: slice the outer ListOffset dimension without copying the offsets buffer.
    OffsetView(OffsetView),
    // A union of scalars and lists (needed for variable nesting depth).
    // tag=0 => scalar leaf element, tag=1 => list element.
    UnionScalarList(UnionScalarList),
}

impl Layout {
    pub fn len(&self) -> usize {
        match self {
            Layout::Leaf(l) => l.len,
            Layout::ListOffset(lo) => lo.len(),
            Layout::Indexed(ix) => ix.len(),
            Layout::OffsetView(v) => v.len(),
            Layout::UnionScalarList(u) => u.len(),
        }
    }

    pub fn element_to_py(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        match self {
            Layout::Leaf(l) => l.scalar_to_py(py, idx),
            Layout::ListOffset(lo) => lo.list_element_to_py(py, idx),
            Layout::Indexed(ix) => ix.element_to_py(py, idx),
            Layout::OffsetView(v) => v.list_element_to_py(py, idx),
            Layout::UnionScalarList(u) => u.element_to_py(py, idx),
        }
    }

    pub fn has_union(&self) -> bool {
        match self {
            Layout::Leaf(_) => false,
            Layout::ListOffset(lo) => lo.content.has_union(),
            Layout::Indexed(ix) => ix.content.has_union(),
            Layout::OffsetView(v) => v.content.has_union(),
            Layout::UnionScalarList(_) => true,
        }
    }

    pub fn is_pure_list_chain(&self) -> bool {
        !self.has_union()
    }

    pub fn uniquify_buffers(&mut self) {
        match self {
            Layout::Leaf(l) => l.uniquify_buffers(),
            Layout::ListOffset(lo) => {
                Arc::make_mut(&mut lo.offsets);
                lo.content.uniquify_buffers();
            }
            Layout::Indexed(ix) => {
                Arc::make_mut(&mut ix.index);
                ix.content.uniquify_buffers();
            }
            Layout::OffsetView(v) => {
                Arc::make_mut(&mut v.offsets);
                v.content.uniquify_buffers();
            }
            Layout::UnionScalarList(u) => u.uniquify_buffers(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Indexed {
    pub index: Arc<Vec<i64>>,
    pub content: Box<Layout>,
}

impl Indexed {
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn element_to_py(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        if idx >= self.len() {
            return Err(index_out_of_bounds(idx, self.len(), "on indexed view"));
        }
        // Negative indices are supported (Python-style) against the content length.
        let n = self.content.len() as i64;
        let mut ix = self.index[idx];
        if ix < 0 {
            ix += n;
        }
        if ix < 0 || ix >= n {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        self.content.element_to_py(py, ix as usize)
    }
}

#[derive(Clone, Debug)]
pub struct OffsetView {
    pub offsets: Arc<Vec<i64>>,
    pub start: usize,
    pub stop: usize, // exclusive, in list units (not offset units)
    pub content: Box<Layout>,
}

impl OffsetView {
    pub fn len(&self) -> usize {
        self.stop.saturating_sub(self.start)
    }

    pub fn list_element_to_py(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        if idx >= self.len() {
            return Err(index_out_of_bounds(idx, self.len(), "on indexed view"));
        }
        let abs = self.start + idx;
        if abs + 1 >= self.offsets.len() {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let start = self.offsets[abs] as usize;
        let end = self.offsets[abs + 1] as usize;
        let out = pyo3::types::PyList::empty_bound(py);
        for j in start..end {
            out.append(self.content.element_to_py(py, j)?)?;
        }
        Ok(out.into())
    }
}

/// Convert an `OffsetView` (view on a `ListOffset` outer axis) into a concrete `ListOffset`
/// with **relative offsets** and a content layout sliced to the visible scalar range.
///
/// This is a common normalization step for kernels that need a canonical, self-contained
/// offsets buffer (e.g. reductions over list chains) and is also critical for streams where
/// batches are typically `OffsetView`s.
pub fn offsetview_to_listoffset(v: &OffsetView) -> PyResult<ListOffset> {
    if v.start > v.stop || v.stop >= v.offsets.len() {
        return Err(invalid_slice_range(v.start, v.stop, v.offsets.len().saturating_sub(1)));
    }
    let base = v.offsets[v.start];
    let mut offs: Vec<i64> = Vec::with_capacity(v.stop - v.start + 1);
    for i in v.start..=v.stop {
        offs.push(v.offsets[i] - base);
    }
    let start = v.offsets[v.start] as usize;
    let end = v.offsets[v.stop] as usize;
    let content = take_range(v.content.as_ref(), start, end)?;
    Ok(ListOffset { offsets: Arc::new(offs), content: Box::new(content) })
}

#[derive(Clone, Debug)]
pub struct Leaf {
    pub dtype: DType,
    pub buffer: LeafBuffer,
    pub validity: Arc<BitVec<u8, Lsb0>>,
    pub len: usize,
    // Performance: track if any nulls exist without scanning the bitmap.
    // Conservative: once true, it may stay true even if nulls are later overwritten.
    pub has_nulls: bool,
}

#[derive(Clone, Debug)]
pub enum LeafBuffer {
    I8(Arc<Vec<i8>>),
    I16(Arc<Vec<i16>>),
    I32(Arc<Vec<i32>>),
    I64(Arc<Vec<i64>>),
    U8(Arc<Vec<u8>>),
    U16(Arc<Vec<u16>>),
    U32(Arc<Vec<u32>>),
    U64(Arc<Vec<u64>>),
    F16(Arc<Vec<u16>>), // store IEEE-754 half bits
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
    Bool(Arc<Vec<u8>>), // 0/1
    Char(Arc<Vec<u32>>),
    String(Arc<Vec<String>>),
}

impl LeafBuffer {
    pub fn new(dtype: DType) -> Self {
        match dtype {
            DType::Int8 => LeafBuffer::I8(Arc::new(Vec::new())),
            DType::Int16 => LeafBuffer::I16(Arc::new(Vec::new())),
            DType::Int32 => LeafBuffer::I32(Arc::new(Vec::new())),
            DType::Int64 => LeafBuffer::I64(Arc::new(Vec::new())),
            DType::UInt8 => LeafBuffer::U8(Arc::new(Vec::new())),
            DType::UInt16 => LeafBuffer::U16(Arc::new(Vec::new())),
            DType::UInt32 => LeafBuffer::U32(Arc::new(Vec::new())),
            DType::UInt64 => LeafBuffer::U64(Arc::new(Vec::new())),
            DType::Float16 => LeafBuffer::F16(Arc::new(Vec::new())),
            DType::Float32 => LeafBuffer::F32(Arc::new(Vec::new())),
            DType::Float64 => LeafBuffer::F64(Arc::new(Vec::new())),
            DType::Bool => LeafBuffer::Bool(Arc::new(Vec::new())),
            DType::Char => LeafBuffer::Char(Arc::new(Vec::new())),
            DType::String => LeafBuffer::String(Arc::new(Vec::new())),
        }
    }

    pub fn reserve(&mut self, n: usize) {
        match self {
            LeafBuffer::I8(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::I16(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::I32(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::I64(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::U8(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::U16(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::U32(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::U64(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::F16(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::F32(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::F64(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::Bool(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::Char(v) => Arc::make_mut(v).reserve(n),
            LeafBuffer::String(v) => Arc::make_mut(v).reserve(n),
        }
    }

    pub fn push_zero(&mut self) {
        match self {
            LeafBuffer::I8(v) => Arc::make_mut(v).push(0),
            LeafBuffer::I16(v) => Arc::make_mut(v).push(0),
            LeafBuffer::I32(v) => Arc::make_mut(v).push(0),
            LeafBuffer::I64(v) => Arc::make_mut(v).push(0),
            LeafBuffer::U8(v) => Arc::make_mut(v).push(0),
            LeafBuffer::U16(v) => Arc::make_mut(v).push(0),
            LeafBuffer::U32(v) => Arc::make_mut(v).push(0),
            LeafBuffer::U64(v) => Arc::make_mut(v).push(0),
            LeafBuffer::F16(v) => Arc::make_mut(v).push(0),
            LeafBuffer::F32(v) => Arc::make_mut(v).push(0.0),
            LeafBuffer::F64(v) => Arc::make_mut(v).push(0.0),
            LeafBuffer::Bool(v) => Arc::make_mut(v).push(0),
            LeafBuffer::Char(v) => Arc::make_mut(v).push(0),
            LeafBuffer::String(v) => Arc::make_mut(v).push(String::new()),
        }
    }

    pub fn copy_range(&self, start: usize, end: usize) -> Self {
        match self {
            LeafBuffer::I8(v) => LeafBuffer::I8(Arc::new(v[start..end].to_vec())),
            LeafBuffer::I16(v) => LeafBuffer::I16(Arc::new(v[start..end].to_vec())),
            LeafBuffer::I32(v) => LeafBuffer::I32(Arc::new(v[start..end].to_vec())),
            LeafBuffer::I64(v) => LeafBuffer::I64(Arc::new(v[start..end].to_vec())),
            LeafBuffer::U8(v) => LeafBuffer::U8(Arc::new(v[start..end].to_vec())),
            LeafBuffer::U16(v) => LeafBuffer::U16(Arc::new(v[start..end].to_vec())),
            LeafBuffer::U32(v) => LeafBuffer::U32(Arc::new(v[start..end].to_vec())),
            LeafBuffer::U64(v) => LeafBuffer::U64(Arc::new(v[start..end].to_vec())),
            LeafBuffer::F16(v) => LeafBuffer::F16(Arc::new(v[start..end].to_vec())),
            LeafBuffer::F32(v) => LeafBuffer::F32(Arc::new(v[start..end].to_vec())),
            LeafBuffer::F64(v) => LeafBuffer::F64(Arc::new(v[start..end].to_vec())),
            LeafBuffer::Bool(v) => LeafBuffer::Bool(Arc::new(v[start..end].to_vec())),
            LeafBuffer::Char(v) => LeafBuffer::Char(Arc::new(v[start..end].to_vec())),
            LeafBuffer::String(v) => LeafBuffer::String(Arc::new(v[start..end].to_vec())),
        }
    }

    pub fn concat(&self, other: &Self) -> PyResult<Self> {
        match (self, other) {
            (LeafBuffer::I8(a), LeafBuffer::I8(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::I8(Arc::new(out)))
            }
            (LeafBuffer::I16(a), LeafBuffer::I16(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::I16(Arc::new(out)))
            }
            (LeafBuffer::I32(a), LeafBuffer::I32(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::I32(Arc::new(out)))
            }
            (LeafBuffer::I64(a), LeafBuffer::I64(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::I64(Arc::new(out)))
            }
            (LeafBuffer::U8(a), LeafBuffer::U8(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::U8(Arc::new(out)))
            }
            (LeafBuffer::U16(a), LeafBuffer::U16(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::U16(Arc::new(out)))
            }
            (LeafBuffer::U32(a), LeafBuffer::U32(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::U32(Arc::new(out)))
            }
            (LeafBuffer::U64(a), LeafBuffer::U64(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::U64(Arc::new(out)))
            }
            (LeafBuffer::F16(a), LeafBuffer::F16(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::F16(Arc::new(out)))
            }
            (LeafBuffer::F32(a), LeafBuffer::F32(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::F32(Arc::new(out)))
            }
            (LeafBuffer::F64(a), LeafBuffer::F64(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::F64(Arc::new(out)))
            }
            (LeafBuffer::Bool(a), LeafBuffer::Bool(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::Bool(Arc::new(out)))
            }
            (LeafBuffer::Char(a), LeafBuffer::Char(b)) => {
                let mut out = a.to_vec();
                out.extend_from_slice(b);
                Ok(LeafBuffer::Char(Arc::new(out)))
            }
        (LeafBuffer::String(a), LeafBuffer::String(b)) => {
            let mut out = a.to_vec();
            out.extend(b.iter().cloned());
            Ok(LeafBuffer::String(Arc::new(out)))
        }
            _ => Err(internal_dtype_buffer_mismatch("leaf concat", DType::Int8)),
        }
    }

    pub fn push_from_index(&mut self, src: &LeafBuffer, idx: usize) -> PyResult<()> {
        match (self, src) {
            (LeafBuffer::I8(o), LeafBuffer::I8(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::I16(o), LeafBuffer::I16(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::I32(o), LeafBuffer::I32(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::I64(o), LeafBuffer::I64(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::U8(o), LeafBuffer::U8(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::U16(o), LeafBuffer::U16(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::U32(o), LeafBuffer::U32(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::U64(o), LeafBuffer::U64(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::F16(o), LeafBuffer::F16(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::F32(o), LeafBuffer::F32(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::F64(o), LeafBuffer::F64(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::Bool(o), LeafBuffer::Bool(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            (LeafBuffer::Char(o), LeafBuffer::Char(s)) => { Arc::make_mut(o).push(s[idx]); Ok(()) }
            _ => Err(internal("push_from_index", "dtype mismatch in leaf buffer")),
        }
    }

    pub fn extend_repeat_first(&mut self, src: &LeafBuffer, count: usize) -> PyResult<()> {
        if count == 0 {
            return Ok(());
        }
        match (self, src) {
            (LeafBuffer::I8(o), LeafBuffer::I8(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::I16(o), LeafBuffer::I16(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::I32(o), LeafBuffer::I32(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::I64(o), LeafBuffer::I64(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::U8(o), LeafBuffer::U8(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::U16(o), LeafBuffer::U16(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::U32(o), LeafBuffer::U32(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::U64(o), LeafBuffer::U64(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::F16(o), LeafBuffer::F16(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::F32(o), LeafBuffer::F32(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::F64(o), LeafBuffer::F64(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::Bool(o), LeafBuffer::Bool(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::Char(o), LeafBuffer::Char(s)) => { Arc::make_mut(o).extend(std::iter::repeat(s[0]).take(count)); Ok(()) }
            (LeafBuffer::String(o), LeafBuffer::String(s)) => {
                let val = s[0].clone();
                Arc::make_mut(o).extend(std::iter::repeat(val).take(count));
                Ok(())
            }
            _ => Err(internal("extend_repeat_first", "dtype mismatch in leaf buffer")),
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            match self {
                LeafBuffer::I8(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()),
                LeafBuffer::I16(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2),
                LeafBuffer::I32(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
                LeafBuffer::I64(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8),
                LeafBuffer::U8(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()),
                LeafBuffer::U16(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2),
                LeafBuffer::U32(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
                LeafBuffer::U64(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8),
                LeafBuffer::F16(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2),
                LeafBuffer::F32(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
                LeafBuffer::F64(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8),
                LeafBuffer::Bool(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()),
                LeafBuffer::Char(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
                LeafBuffer::String(_) => &[],
            }
        }
    }

    pub fn push_from_bytes(&mut self, bytes: &[u8], dtype: DType) -> PyResult<()> {
        match (self, dtype) {
            (LeafBuffer::I8(v), DType::Int8) => { Arc::make_mut(v).push(i8::from_ne_bytes([bytes[0]])); Ok(()) }
            (LeafBuffer::U8(v), DType::UInt8) => { Arc::make_mut(v).push(bytes[0]); Ok(()) }
            (LeafBuffer::Bool(v), DType::Bool) => { Arc::make_mut(v).push(if bytes[0] == 0 {0} else {1}); Ok(()) }
            (LeafBuffer::I16(v), DType::Int16) => { Arc::make_mut(v).push(i16::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::U16(v), DType::UInt16) => { Arc::make_mut(v).push(u16::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::F16(v), DType::Float16) => { Arc::make_mut(v).push(u16::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::I32(v), DType::Int32) => { Arc::make_mut(v).push(i32::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::U32(v), DType::UInt32) => { Arc::make_mut(v).push(u32::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::F32(v), DType::Float32) => { Arc::make_mut(v).push(f32::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::I64(v), DType::Int64) => { Arc::make_mut(v).push(i64::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::U64(v), DType::UInt64) => { Arc::make_mut(v).push(u64::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::F64(v), DType::Float64) => { Arc::make_mut(v).push(f64::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::Char(v), DType::Char) => { Arc::make_mut(v).push(u32::from_ne_bytes(bytes.try_into().unwrap())); Ok(()) }
            (LeafBuffer::String(_), DType::String) => Err(dtype_unsupported("push_from_bytes", DType::String)),
            _ => Err(internal("push_from_bytes", "dtype mismatch in leaf buffer")),
        }
    }

    pub fn set_from_bytes(&mut self, idx: usize, bytes: &[u8], dtype: DType) -> PyResult<()> {
        match (self, dtype) {
            (LeafBuffer::I8(v), DType::Int8) => { Arc::make_mut(v)[idx] = i8::from_ne_bytes([bytes[0]]); Ok(()) }
            (LeafBuffer::U8(v), DType::UInt8) => { Arc::make_mut(v)[idx] = bytes[0]; Ok(()) }
            (LeafBuffer::Bool(v), DType::Bool) => { Arc::make_mut(v)[idx] = if bytes[0] == 0 {0} else {1}; Ok(()) }
            (LeafBuffer::I16(v), DType::Int16) => { Arc::make_mut(v)[idx] = i16::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::U16(v), DType::UInt16) => { Arc::make_mut(v)[idx] = u16::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::F16(v), DType::Float16) => { Arc::make_mut(v)[idx] = u16::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::I32(v), DType::Int32) => { Arc::make_mut(v)[idx] = i32::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::U32(v), DType::UInt32) => { Arc::make_mut(v)[idx] = u32::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::F32(v), DType::Float32) => { Arc::make_mut(v)[idx] = f32::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::I64(v), DType::Int64) => { Arc::make_mut(v)[idx] = i64::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::U64(v), DType::UInt64) => { Arc::make_mut(v)[idx] = u64::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::F64(v), DType::Float64) => { Arc::make_mut(v)[idx] = f64::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::Char(v), DType::Char) => { Arc::make_mut(v)[idx] = u32::from_ne_bytes(bytes.try_into().unwrap()); Ok(()) }
            (LeafBuffer::String(_), DType::String) => Err(dtype_unsupported("set_from_bytes", DType::String)),
            _ => Err(internal("set_from_bytes", "dtype mismatch in leaf buffer")),
        }
    }
}

impl Leaf {
    pub fn new(dtype: DType) -> Self {
        Self {
            dtype,
            buffer: LeafBuffer::new(dtype),
            validity: Arc::new(BitVec::new()),
            len: 0,
            has_nulls: false,
        }
    }

    pub fn push_null(&mut self) {
        Arc::make_mut(&mut self.validity).push(false);
        self.len += 1;
        self.has_nulls = true;
        self.buffer.push_zero();
    }

    pub fn push_value(&mut self, bytes: &[u8]) -> PyResult<()> {
        debug_assert_eq!(bytes.len(), self.dtype.size_bytes());
        Arc::make_mut(&mut self.validity).push(true);
        self.len += 1;
        self.buffer.push_from_bytes(bytes, self.dtype)?;
        Ok(())
    }

    pub fn scalar_to_py(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        if idx >= self.len {
            return Err(index_out_of_bounds(idx, self.len, "on leaf"));
        }
        if !self.validity[idx] {
            return Ok(py.None());
        }
        match self.dtype {
            DType::Int8 => match &self.buffer { LeafBuffer::I8(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::UInt8 => match &self.buffer { LeafBuffer::U8(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Bool => match &self.buffer { LeafBuffer::Bool(v) => Ok((v[idx] != 0).into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Int16 => match &self.buffer { LeafBuffer::I16(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::UInt16 => match &self.buffer { LeafBuffer::U16(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Float16 => {
                let bits = match &self.buffer { LeafBuffer::F16(v) => v[idx], _ => return Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) };
                let v = f16::from_bits(bits);
                Ok(f32::from(v).into_py(py))
            }
            DType::Int32 => match &self.buffer { LeafBuffer::I32(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::UInt32 => match &self.buffer { LeafBuffer::U32(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Float32 => match &self.buffer { LeafBuffer::F32(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Int64 => match &self.buffer { LeafBuffer::I64(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::UInt64 => match &self.buffer { LeafBuffer::U64(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Float64 => match &self.buffer { LeafBuffer::F64(v) => Ok(v[idx].into_py(py)), _ => Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) },
            DType::Char => {
                let v = match &self.buffer { LeafBuffer::Char(v) => v[idx], _ => return Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) };
                let c = char::from_u32(v)
                    .ok_or_else(|| arg_invalid("char", "invalid Unicode scalar value", "use a valid UTF-32 code point."))?;
                Ok(c.to_string().into_py(py))
            }
            DType::String => {
                let s = match &self.buffer { LeafBuffer::String(v) => &v[idx], _ => return Err(internal_dtype_buffer_mismatch("leaf scalar_to_py", self.dtype)) };
                Ok(s.clone().into_py(py))
            }
        }
    }

    pub fn encode_scalar(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: DType) -> PyResult<(bool, Vec<u8>)> {
        if dtype == DType::String {
            return Err(dtype_unsupported("encode_scalar", DType::String));
        }
        let mut tmp = Leaf::new(dtype);
        push_scalar(py, obj, dtype, &mut tmp)?;
        debug_assert_eq!(tmp.len, 1);
        Ok((tmp.validity[0], tmp.buffer.as_bytes()[..dtype.size_bytes()].to_vec()))
    }

    pub fn set_encoded(&mut self, idx: usize, valid: bool, bytes: &[u8]) -> PyResult<()> {
        if idx >= self.len {
            return Err(index_out_of_bounds(idx, self.len, "on leaf"));
        }
        if bytes.len() != self.dtype.size_bytes() {
            return Err(internal("Leaf::set_encoded", "dtype byte width mismatch"));
        }
        if !valid {
            self.has_nulls = true;
        }
        Arc::make_mut(&mut self.validity).set(idx, valid);
        self.buffer.set_from_bytes(idx, bytes, self.dtype)?;
        Ok(())
    }

    pub fn set_i32(&mut self, idx: usize, value: i32) -> PyResult<()> {
        if idx >= self.len {
            return Err(index_out_of_bounds(idx, self.len, "on leaf"));
        }
        if self.dtype != DType::Int32 {
            return Err(internal("Leaf::set_i32", "requires dtype=int32"));
        }
        match &mut self.buffer {
            LeafBuffer::I32(v) => Arc::make_mut(v)[idx] = value,
            _ => return Err(internal("Leaf::set_i32", "dtype mismatch")),
        }
        Ok(())
    }

    pub fn uniquify_buffers(&mut self) {
        Arc::make_mut(&mut self.validity);
        match &mut self.buffer {
            LeafBuffer::I8(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::I16(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::I32(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::I64(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::U8(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::U16(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::U32(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::U64(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::F16(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::F32(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::F64(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::Bool(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::Char(v) => {
                Arc::make_mut(v);
            }
            LeafBuffer::String(v) => {
                Arc::make_mut(v);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ListOffset {
    pub offsets: Arc<Vec<i64>>,
    pub content: Box<Layout>,
}

impl ListOffset {
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn list_element_to_py(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        if idx >= self.len() {
            return Err(index_out_of_bounds(idx, self.len(), "on indexed view"));
        }
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        let out = pyo3::types::PyList::empty_bound(py);
        for j in start..end {
            out.append(self.content.element_to_py(py, j)?)?;
        }
        Ok(out.into())
    }

    pub fn child_len_total(&self, idx: usize) -> PyResult<usize> {
        if idx >= self.len() {
            return Err(index_out_of_bounds(idx, self.len(), "on indexed view"));
        }
        Ok((self.offsets[idx + 1] - self.offsets[idx]) as usize)
    }

    pub fn child_len_non_null_scalars(&self, idx: usize) -> PyResult<usize> {
        if idx >= self.len() {
            return Err(index_out_of_bounds(idx, self.len(), "on indexed view"));
        }
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        match self.content.as_ref() {
            Layout::Leaf(l) => Ok(l.validity[start..end].count_ones()),
            Layout::UnionScalarList(u) => u.count_non_null_scalars_in_range(start, end),
            _ => Ok(0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct UnionScalarList {
    // tag=0 => scalar leaf element, tag=1 => list element
    pub tags: Vec<u8>,
    pub index: Vec<i64>,
    pub scalars: Leaf,
    pub lists: ListOffset,
}

impl UnionScalarList {
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    pub fn element_to_py(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        if idx >= self.len() {
            return Err(index_out_of_bounds(idx, self.len(), "on indexed view"));
        }
        let tag = self.tags[idx];
        let ix = self.index[idx] as usize;
        match tag {
            0 => self.scalars.scalar_to_py(py, ix),
            1 => self.lists.list_element_to_py(py, ix),
            _ => Err(internal("union element", "invalid union tag")),
        }
    }

    pub fn count_non_null_scalars_in_range(&self, start: usize, end: usize) -> PyResult<usize> {
        if end > self.len() || start > end {
            return Err(invalid_slice_range(start, end, self.len()));
        }
        let mut c = 0usize;
        for i in start..end {
            if self.tags[i] == 0 {
                let ix = self.index[i] as usize;
                if self.scalars.validity[ix] {
                    c += 1;
                }
            }
        }
        Ok(c)
    }

    pub fn uniquify_buffers(&mut self) {
        self.scalars.uniquify_buffers();
        Arc::make_mut(&mut self.lists.offsets);
        self.lists.content.uniquify_buffers();
    }

    /// Slice ``[start, stop)`` along the union outer axis, compacting scalar/list pools.
    pub fn take_range(&self, start: usize, stop: usize) -> PyResult<Self> {
        take_union_scalar_list_range(self, start, stop)
    }
}

#[derive(Clone, Debug)]
pub struct GrumpyArray {
    pub dtype: DType,
    pub layout: Layout,
}

impl GrumpyArray {
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn to_py_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = pyo3::types::PyList::empty_bound(py);
        for i in 0..self.len() {
            out.append(self.layout.element_to_py(py, i)?)?;
        }
        Ok(out.into())
    }
}

pub fn build_array(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: DType) -> PyResult<GrumpyArray> {
    if let Some(layout) = try_build_from_numpy(py, obj, dtype)? {
        return Ok(GrumpyArray { dtype, layout });
    }
    let layout = build_layout(py, obj, dtype)?;
    Ok(GrumpyArray { dtype, layout })
}

impl GrumpyArray {
    pub fn is_pure_list_chain(&self) -> bool {
        self.layout.is_pure_list_chain()
    }

    /// Eagerly detach shared Arc buffers so subsequent in-place writes avoid copy-on-write.
    pub fn uniquify_buffers(&mut self) {
        self.layout.uniquify_buffers();
    }
}

pub fn list_chain_depth(layout: &Layout) -> Option<usize> {
    let mut d = 0usize;
    let mut cur = layout;
    loop {
        match cur {
            Layout::Leaf(_) => return Some(d),
            Layout::ListOffset(lo) => {
                d += 1;
                cur = lo.content.as_ref();
            }
            Layout::OffsetView(v) => {
                d += 1;
                cur = v.content.as_ref();
            }
            Layout::Indexed(ix) => {
                // Indexed preserves depth; it only changes indexing.
                cur = ix.content.as_ref();
            }
            Layout::UnionScalarList(_) => return None,
        }
    }
}

pub fn take_range(layout: &Layout, start: usize, end: usize) -> PyResult<Layout> {
    if start > end {
        return Err(invalid_slice_range(start, end, layout.len()));
    }
    match layout {
        Layout::Leaf(l) => {
            if end > l.len {
                return Err(index_out_of_bounds(end, l.len, "on leaf slice"));
            }
            let mut out = Leaf::new(l.dtype);
            out.len = end - start;
            out.validity = Arc::new(l.validity[start..end].to_bitvec());
            out.buffer = l.buffer.copy_range(start, end);
            out.has_nulls = l.has_nulls;
            Ok(Layout::Leaf(out))
        }
        Layout::ListOffset(lo) => {
            if end > lo.len() {
                return Err(index_out_of_bounds(end, lo.len(), "on list slice"));
            }
            let child_start = lo.offsets[start] as usize;
            let child_end = lo.offsets[end] as usize;
            let mut offsets = Vec::with_capacity(end - start + 1);
            offsets.push(0i64);
            let mut acc = 0i64;
            for i in start..end {
                let len_i = lo.offsets[i + 1] - lo.offsets[i];
                acc += len_i;
                offsets.push(acc);
            }
            let content = take_range(lo.content.as_ref(), child_start, child_end)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: Arc::new(offsets),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            if end > v.len() {
                return Err(index_out_of_bounds(end, v.len(), "on offset view slice"));
            }
            Ok(Layout::OffsetView(OffsetView {
                offsets: v.offsets.clone(),
                start: v.start + start,
                stop: v.start + end,
                content: v.content.clone(),
            }))
        }
        Layout::Indexed(ix) => {
            if end > ix.len() {
                return Err(index_out_of_bounds(end, ix.len(), "on indexed slice"));
            }
            let sub = ix.index[start..end].to_vec();
            Ok(Layout::Indexed(Indexed {
                index: Arc::new(sub),
                content: ix.content.clone(),
            }))
        }
        Layout::UnionScalarList(u) => u.take_range(start, end).map(Layout::UnionScalarList),
    }
}

/// Concatenate layout segments along the innermost leaf axis (used by union list compaction).
pub fn concat_layout_segments(segments: &[Layout]) -> PyResult<Layout> {
    if segments.is_empty() {
        return Err(concat_incompatible("cannot concat empty layout segments", "pass at least one layout segment."));
    }
    if segments.len() == 1 {
        return Ok(segments[0].clone());
    }
    let dt = match &segments[0] {
        Layout::Leaf(l) => l.dtype,
        Layout::ListOffset(lo) => leaf_dtype(lo.content.as_ref())?,
        _ => {
            return Err(layout_unsupported("concat_layout_segments", "unsupported layout kind for segment concat"))
        }
    };
    let mut cur = segments[0].clone();
    for seg in &segments[1..] {
        cur = concat_two_layout_segments(&cur, seg, dt)?;
    }
    Ok(cur)
}

fn leaf_dtype(layout: &Layout) -> PyResult<DType> {
    match layout {
        Layout::Leaf(l) => Ok(l.dtype),
        Layout::ListOffset(lo) => leaf_dtype(lo.content.as_ref()),
        Layout::OffsetView(v) => leaf_dtype(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_dtype(ix.content.as_ref()),
        Layout::UnionScalarList(u) => Ok(u.scalars.dtype),
    }
}

fn concat_two_layout_segments(a: &Layout, b: &Layout, dt: DType) -> PyResult<Layout> {
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => {
            if la.dtype != lb.dtype {
                return Err(dtype_mismatch(dt, lb.dtype, "in concat_layout_segments"));
            }
            let mut out = Leaf::new(dt);
            out.len = la.len + lb.len;
            out.has_nulls = la.has_nulls || lb.has_nulls;
            let mut validity = (*la.validity).clone();
            validity.extend_from_bitslice(lb.validity.as_bitslice());
            out.validity = Arc::new(validity);
            out.buffer = la.buffer.concat(&lb.buffer)?;
            Ok(Layout::Leaf(out))
        }
        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => {
            let mut offsets = oa.offsets.to_vec();
            let shift = *offsets.last().unwrap_or(&0);
            for &o in ob.offsets.iter().skip(1) {
                offsets.push(shift + o);
            }
            let content = concat_two_layout_segments(oa.content.as_ref(), ob.content.as_ref(), dt)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: Arc::new(offsets),
                content: Box::new(content),
            }))
        }
        _ => Err(concat_incompatible("incompatible layout kinds in concat_layout_segments", "ensure both segments share the same layout structure.")),
    }
}

fn take_union_scalar_list_range(u: &UnionScalarList, start: usize, end: usize) -> PyResult<UnionScalarList> {
    if start > end {
        return Err(invalid_slice_range(start, end, u.len()));
    }
    if end > u.len() {
        return Err(index_out_of_bounds(end, u.len(), "on union slice"));
    }
    let positions: Vec<usize> = (start..end).collect();
    pick_union_scalar_list_indices(u, &positions)
}

/// Remap union outer-axis tags/index for a pick/slice into compact scalar/list pool indices.
pub fn remap_union_pick(
    sub_tags: &[u8],
    sub_old_index: &[i64],
) -> PyResult<(Vec<i64>, Vec<usize>, Vec<usize>)> {
    use std::collections::HashMap;
    let mut scalar_remap: HashMap<usize, i64> = HashMap::new();
    let mut list_remap: HashMap<usize, i64> = HashMap::new();
    let mut scalar_old: Vec<usize> = Vec::new();
    let mut list_old: Vec<usize> = Vec::new();

    for i in 0..sub_tags.len() {
        let tag = sub_tags[i];
        let ix = sub_old_index[i] as usize;
        if tag == 0 {
            scalar_remap.entry(ix).or_insert_with(|| {
                let n = scalar_old.len() as i64;
                scalar_old.push(ix);
                n
            });
        } else if tag == 1 {
            list_remap.entry(ix).or_insert_with(|| {
                let n = list_old.len() as i64;
                list_old.push(ix);
                n
            });
        } else {
            return Err(internal("union remap", "invalid union tag"));
        }
    }

    let mut new_index = Vec::with_capacity(sub_tags.len());
    for i in 0..sub_tags.len() {
        let tag = sub_tags[i];
        let ix = sub_old_index[i] as usize;
        new_index.push(if tag == 0 {
            scalar_remap[&ix]
        } else {
            list_remap[&ix]
        });
    }
    Ok((new_index, scalar_old, list_old))
}

fn pick_union_scalar_list_indices(
    u: &UnionScalarList,
    outer_positions: &[usize],
) -> PyResult<UnionScalarList> {
    if outer_positions.is_empty() {
        let scalars = Leaf::new(u.scalars.dtype);
        let lists = ListOffset {
            offsets: Arc::new(vec![0i64]),
            content: Box::new(Layout::Leaf(Leaf::new(u.scalars.dtype))),
        };
        return Ok(UnionScalarList {
            tags: Vec::new(),
            index: Vec::new(),
            scalars,
            lists,
        });
    }
    let sub_tags: Vec<u8> = outer_positions.iter().map(|&i| u.tags[i]).collect();
    let sub_old_index: Vec<i64> = outer_positions.iter().map(|&i| u.index[i]).collect();

    let (new_index, scalar_old, list_old) = remap_union_pick(&sub_tags, &sub_old_index)?;

    let scalars = take_leaf_indices(&u.scalars, &scalar_old)?;

    let mut new_list_offsets = vec![0i64];
    let mut content_segs: Vec<Layout> = Vec::with_capacity(list_old.len());
    let mut acc = 0i64;
    for &li in &list_old {
        let s = u.lists.offsets[li] as usize;
        let e = u.lists.offsets[li + 1] as usize;
        let seg = take_range(u.lists.content.as_ref(), s, e)?;
        acc += (e - s) as i64;
        new_list_offsets.push(acc);
        content_segs.push(seg);
    }
    let list_content = if content_segs.is_empty() {
        u.lists.content.as_ref().clone()
    } else {
        concat_layout_segments(&content_segs)?
    };
    let lists = ListOffset {
        offsets: Arc::new(new_list_offsets),
        content: Box::new(list_content),
    };

    Ok(UnionScalarList {
        tags: sub_tags,
        index: new_index,
        scalars,
        lists,
    })
}

fn finalize_gathered_union(u: UnionScalarList) -> Layout {
    if u.tags.is_empty() {
        return Layout::Leaf(Leaf::new(u.scalars.dtype));
    }
    if u.tags.iter().all(|&t| t == 0) {
        return Layout::Leaf(u.scalars);
    }
    if u.tags.iter().all(|&t| t == 1) {
        return Layout::ListOffset(u.lists);
    }
    Layout::UnionScalarList(u)
}

/// Fancy axis-0 selection on a union array, compacting scalar/list pools.
pub fn gather_union_axis0_fancy(u: &UnionScalarList, indices: &[i64]) -> PyResult<Layout> {
    let root_len = u.len() as i64;
    let mut positions = Vec::with_capacity(indices.len());
    for &raw in indices {
        positions.push(normalize_index(raw, root_len)?);
    }
    let picked = pick_union_scalar_list_indices(u, &positions)?;
    Ok(finalize_gathered_union(picked))
}

fn concat_len1_leaves(elems: &[Layout]) -> PyResult<Layout> {
    let dt = match &elems[0] {
        Layout::Leaf(l) => l.dtype,
        _ => return Err(internal("concat_len1_leaves", "expected leaf layouts")),
    };
    let mut out = Leaf::new(dt);
    out.len = elems.len();
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; elems.len()]);
    out.has_nulls = false;
    out.buffer = LeafBuffer::new(dt);
    let out_valid = Arc::make_mut(&mut out.validity);
    for (i, e) in elems.iter().enumerate() {
        let l = match e {
            Layout::Leaf(l) => l,
            _ => return Err(internal("concat_len1_leaves", "expected leaf layouts")),
        };
        if l.len != 1 {
            return Err(internal("concat_len1_leaves", "expected length-1 leaves"));
        }
        if !l.validity[0] {
            out_valid.set(i, false);
            out.has_nulls = true;
        }
        out.buffer.push_from_index(&l.buffer, 0)?;
    }
    Ok(Layout::Leaf(out))
}

/// Stack axis-0 element selections into a single layout (list-chain path).
pub fn stack_axis0_selects(elems: &[Layout]) -> PyResult<Layout> {
    if elems.is_empty() {
        return Err(concat_incompatible("cannot stack empty selection", "pass at least one selected element."));
    }
    let all_scalar1 = elems
        .iter()
        .all(|e| matches!(e, Layout::Leaf(l) if l.len == 1));
    if all_scalar1 {
        return concat_len1_leaves(elems);
    }
    let all_leaves = elems.iter().all(|e| matches!(e, Layout::Leaf(_)));
    if all_leaves {
        let mut offsets = vec![0i64];
        let mut acc = 0i64;
        let mut segs = Vec::with_capacity(elems.len());
        for e in elems {
            let l = match e {
                Layout::Leaf(l) => l,
                _ => unreachable!(),
            };
            acc += l.len as i64;
            offsets.push(acc);
            segs.push(e.clone());
        }
        let content = concat_layout_segments(&segs)?;
        return Ok(Layout::ListOffset(ListOffset {
            offsets: Arc::new(offsets),
            content: Box::new(content),
        }));
    }
    Err(layout_unsupported("axis-0 gather", "unsupported stacked layout kinds"))
}

fn is_scalar_segment(seg: &Layout) -> bool {
    matches!(seg, Layout::Leaf(l) if l.len == 1)
}

fn segment_flat_len(seg: &Layout) -> PyResult<usize> {
    match seg {
        Layout::Leaf(l) => Ok(l.len),
        Layout::ListOffset(lo) => Ok(lo.offsets[lo.len()] as usize),
        Layout::UnionScalarList(u) => {
            let mut n = 0usize;
            for i in 0..u.len() {
                n += segment_flat_len(&drop_axis0_select_element(
                    &Layout::UnionScalarList(u.clone()),
                    i,
                )?)?;
            }
            Ok(n)
        }
        Layout::OffsetView(v) => {
            let start = v.offsets[v.start] as usize;
            let end = v.offsets[v.stop] as usize;
            segment_flat_len(&take_range(v.content.as_ref(), start, end)?)
        }
        Layout::Indexed(ix) => {
            let mut n = 0usize;
            for i in 0..ix.len() {
                n += segment_flat_len(&drop_axis0_select_element(
                    &Layout::Indexed(crate::layout::Indexed {
                        index: ix.index.clone(),
                        content: ix.content.clone(),
                    }),
                    i,
                )?)?;
            }
            Ok(n)
        }
    }
}

fn build_union_from_broadcast_segments(segs: &[Layout], dtype: DType) -> PyResult<Layout> {
    let mut tags = Vec::with_capacity(segs.len());
    let mut index = Vec::with_capacity(segs.len());
    let mut scalars = Leaf::new(dtype);
    let mut list_row_segs: Vec<Layout> = Vec::new();

    for seg in segs {
        if is_scalar_segment(seg) {
            tags.push(0);
            index.push(scalars.len as i64);
            let l = match seg {
                Layout::Leaf(l) => l,
                _ => unreachable!(),
            };
            append_leaf_into(&mut scalars, l)?;
        } else {
            tags.push(1);
            index.push(list_row_segs.len() as i64);
            list_row_segs.push(seg.clone());
        }
    }

    let mut list_offsets = vec![0i64];
    let mut acc = 0i64;
    let mut content_segs: Vec<Layout> = Vec::with_capacity(list_row_segs.len());
    for seg in &list_row_segs {
        content_segs.push(seg.clone());
        acc += segment_flat_len(seg)? as i64;
        list_offsets.push(acc);
    }
    let list_content = if content_segs.is_empty() {
        Layout::Leaf(Leaf::new(dtype))
    } else {
        concat_layout_segments(&content_segs)?
    };
    let lists = ListOffset {
        offsets: Arc::new(list_offsets),
        content: Box::new(list_content),
    };

    Ok(Layout::UnionScalarList(UnionScalarList {
        tags,
        index,
        scalars,
        lists,
    }))
}

/// Stack per-row broadcast segments; builds a union when rows mix scalars and lists.
pub fn stack_axis0_broadcast(segs: &[Layout], dtype: DType) -> PyResult<Layout> {
    if segs.is_empty() {
        return Err(concat_incompatible("cannot stack empty broadcast segments", "pass at least one broadcast segment."));
    }
    let all_scalar1 = segs.iter().all(is_scalar_segment);
    if all_scalar1 {
        return concat_len1_leaves(segs);
    }
    let has_scalar = segs.iter().any(is_scalar_segment);
    let has_listlike = segs.iter().any(|s| !is_scalar_segment(s));
    if has_scalar && has_listlike {
        return build_union_from_broadcast_segments(segs, dtype);
    }
    if has_listlike && !has_scalar {
        return build_union_from_broadcast_segments(segs, dtype);
    }
    stack_axis0_selects(segs)
}

/// Fancy axis-0 selection for any layout (union or list-chain).
pub fn gather_axis0_fancy(layout: &Layout, indices: &[i64]) -> PyResult<Layout> {
    match layout {
        Layout::UnionScalarList(u) => gather_union_axis0_fancy(u, indices),
        _ => {
            let root_len = layout.len() as i64;
            let elems: Vec<Layout> = indices
                .iter()
                .map(|&raw| {
                    let i = normalize_index(raw, root_len)?;
                    drop_axis0_select_element(layout, i)
                })
                .collect::<PyResult<_>>()?;
            stack_axis0_selects(&elems)
        }
    }
}

/// Coordinate indexing: apply ``coords`` axis-by-axis, returning the selected sub-layout.
pub fn index_by_coordinates(layout: &Layout, coords: &[i64]) -> PyResult<Layout> {
    if coords.is_empty() {
        return Err(arg_invalid("index", "empty coordinate index", "pass at least one coordinate."));
    }
    let ix = normalize_index(coords[0], layout.len() as i64)?;
    let elem = drop_axis0_select_element(layout, ix)?;
    if coords.len() == 1 {
        return Ok(elem);
    }
    if let Layout::Leaf(l) = &elem {
        if l.len == 1 {
            for &c in &coords[1..] {
                normalize_index(c, 1)?;
            }
            return Ok(elem);
        }
    }
    index_by_coordinates(&elem, &coords[1..])
}

/// 2D fancy coordinate gather; result is a flat leaf when every selection is a scalar.
pub fn gather_coordinate_fancy_2d(layout: &Layout, rows: &[i64], cols: &[i64]) -> PyResult<Layout> {
    if rows.len() != cols.len() {
        return Err(shape_mismatch("fancy coordinate indexing", "row and column index arrays must have the same length", "ensure len(rows) == len(cols)."));
    }
    let mut elems = Vec::with_capacity(rows.len());
    for k in 0..rows.len() {
        elems.push(index_by_coordinates(layout, &[rows[k], cols[k]])?);
    }
    let all_scalar1 = elems
        .iter()
        .all(|e| matches!(e, Layout::Leaf(l) if l.len == 1));
    if all_scalar1 {
        return concat_len1_leaves(&elems);
    }
    Err(layout_unsupported("fancy coordinate gather", "selection produced non-scalar elements"))
}

/// Peel one outer nesting axis (ListOffset content or union element concat).
pub fn peel_layout_axis(layout: &Layout) -> PyResult<Layout> {
    match layout {
        Layout::ListOffset(lo) => Ok((*lo.content).clone()),
        Layout::UnionScalarList(u) => {
            let mut elems = Vec::with_capacity(u.len());
            for i in 0..u.len() {
                elems.push(drop_axis0_select_element(layout, i)?);
            }
            stack_axis0_selects(&elems)
        }
        Layout::OffsetView(v) => {
            let start = v.offsets[v.start] as usize;
            let end = v.offsets[v.stop] as usize;
            take_range(v.content.as_ref(), start, end)
        }
        Layout::Indexed(ix) => Ok((*ix.content).clone()),
        Layout::Leaf(_) => Ok(layout.clone()),
    }
}

/// Drop the first ``drops`` schema nesting axes for dot-notation column views.
pub fn drop_layout_axes(layout: &Layout, drops: usize) -> PyResult<Layout> {
    let mut cur = layout.clone();
    for _ in 0..drops {
        cur = peel_layout_axis(&cur)?;
    }
    Ok(cur)
}

fn append_leaf_into(out: &mut Leaf, src: &Leaf) -> PyResult<()> {
    if out.dtype != src.dtype {
        return Err(dtype_mismatch(out.dtype, src.dtype, "while flattening layout to leaf"));
    }
    if src.has_nulls {
        out.has_nulls = true;
    }
    let out_valid = Arc::make_mut(&mut out.validity);
    for i in 0..src.len {
        out_valid.push(src.validity[i]);
        out.buffer.push_from_index(&src.buffer, i)?;
    }
    out.len += src.len;
    Ok(())
}

fn flatten_layout_to_leaf(layout: &Layout, dtype: DType) -> PyResult<Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l.clone()),
        Layout::ListOffset(lo) => {
            let mut out = Leaf::new(dtype);
            for i in 0..lo.len() {
                let sub = drop_axis0_select_element(layout, i)?;
                let leaf = flatten_layout_to_leaf(&sub, dtype)?;
                append_leaf_into(&mut out, &leaf)?;
            }
            Ok(out)
        }
        Layout::UnionScalarList(u) => {
            let mut out = Leaf::new(dtype);
            for i in 0..u.len() {
                let sub = drop_axis0_select_element(layout, i)?;
                let leaf = flatten_layout_to_leaf(&sub, dtype)?;
                append_leaf_into(&mut out, &leaf)?;
            }
            Ok(out)
        }
        Layout::OffsetView(v) => {
            let start = v.offsets[v.start] as usize;
            let end = v.offsets[v.stop] as usize;
            flatten_layout_to_leaf(&take_range(v.content.as_ref(), start, end)?, dtype)
        }
        Layout::Indexed(ix) => {
            let mut out = Leaf::new(dtype);
            for i in 0..ix.len() {
                let sub = drop_axis0_select_element(layout, i)?;
                let leaf = flatten_layout_to_leaf(&sub, dtype)?;
                append_leaf_into(&mut out, &leaf)?;
            }
            Ok(out)
        }
    }
}

/// Return a fully flattened leaf view (zero-copy for pure list-chains when possible).
pub fn leaf_view(layout: &Layout, dtype: DType) -> PyResult<Leaf> {
    if let Some(_) = list_chain_depth(layout) {
        let mut cur = layout;
        loop {
            match cur {
                Layout::Leaf(l) => return Ok(l.clone()),
                Layout::ListOffset(lo) => cur = lo.content.as_ref(),
                Layout::OffsetView(v) => cur = v.content.as_ref(),
                Layout::Indexed(ix) => cur = ix.content.as_ref(),
                Layout::UnionScalarList(_) => break,
            }
        }
    }
    flatten_layout_to_leaf(layout, dtype)
}

/// Number of nesting axes in a layout (union outer axis counts as one).
pub fn layout_ndim(layout: &Layout) -> PyResult<usize> {
    match layout {
        Layout::Indexed(ix) => layout_ndim(ix.content.as_ref()),
        Layout::OffsetView(v) => layout_ndim(v.content.as_ref()),
        Layout::UnionScalarList(u) => {
            let mut inner = 0usize;
            for i in 0..u.len() {
                if u.tags[i] == 1 {
                    let li = u.index[i] as usize;
                    let start = u.lists.offsets[li] as usize;
                    let end = u.lists.offsets[li + 1] as usize;
                    let seg = take_range(u.lists.content.as_ref(), start, end)?;
                    inner = inner.max(layout_ndim(&seg)?);
                }
            }
            Ok(1 + inner)
        }
        _ => {
            let depth = list_chain_depth(layout).ok_or_else(|| {
                internal("layout_ndim", "could not determine layout depth")
            })?;
            Ok(depth + 1)
        }
    }
}

pub fn drop_axis0_select_element(layout: &Layout, idx: usize) -> PyResult<Layout> {
    match layout {
        Layout::ListOffset(lo) => {
            if idx >= lo.len() {
            return Err(index_out_of_bounds(idx, lo.len(), "on list axis"));
            }
            let start = lo.offsets[idx] as usize;
            let end = lo.offsets[idx + 1] as usize;
            take_range(lo.content.as_ref(), start, end)
        }
        Layout::OffsetView(v) => {
            if idx >= v.len() {
            return Err(index_out_of_bounds(idx, v.len(), "on offset view"));
            }
            let abs = v.start + idx;
            let start = v.offsets[abs] as usize;
            let end = v.offsets[abs + 1] as usize;
            take_range(v.content.as_ref(), start, end)
        }
        Layout::Indexed(ix) => {
            if idx >= ix.len() {
                return Err(index_out_of_bounds_simple("on this axis"));
            }
            // Return a scalar/list element view by redirecting to content.
            let n = ix.content.len() as i64;
            let mut j = ix.index[idx];
            if j < 0 {
                j += n;
            }
            if j < 0 || j >= n {
                return Err(index_out_of_bounds_simple("on this axis"));
            }
            drop_axis0_select_element(ix.content.as_ref(), j as usize)
        }
        Layout::Leaf(l) => {
            if idx >= l.len {
                return Err(index_out_of_bounds_simple("on this axis"));
            }
            Ok(Layout::Leaf(take_leaf_indices(l, &[idx])?))
        }
        Layout::UnionScalarList(u) => {
            if idx >= u.len() {
            return Err(index_out_of_bounds(idx, u.len(), "on union axis"));
            }
            match u.tags[idx] {
                0 => {
                    let ix = u.index[idx] as usize;
                    Ok(Layout::Leaf(take_leaf_indices(&u.scalars, &[ix])?))
                }
                1 => {
                    let li = u.index[idx] as usize;
                    let start = u.lists.offsets[li] as usize;
                    let end = u.lists.offsets[li + 1] as usize;
                    take_range(u.lists.content.as_ref(), start, end)
                }
                _ => Err(internal("union element", "invalid union tag")),
            }
        }
    }
}

fn take_leaf_indices(l: &Leaf, indices: &[usize]) -> PyResult<Leaf> {
    let mut out = Leaf::new(l.dtype);
    out.len = indices.len();
    Arc::make_mut(&mut out.validity).reserve(indices.len());
    out.buffer.reserve(indices.len());
    for &ix in indices {
        if ix >= l.len {
            return Err(index_out_of_bounds(ix, l.len, "on leaf gather"));
        }
        Arc::make_mut(&mut out.validity).push(l.validity[ix]);
        out.buffer.push_from_index(&l.buffer, ix)?;
    }
    out.has_nulls = l.has_nulls;
    Ok(out)
}

pub fn coord_to_leaf_index(layout: &Layout, coords: &[i64]) -> PyResult<usize> {
    if let Layout::UnionScalarList(u) = layout {
        return coord_to_leaf_index_union(u, coords);
    }
    // Fast path: 2D list-chain (ListOffset -> Leaf), coords [row, col].
    if coords.len() == 2 {
        if let Layout::ListOffset(lo) = layout {
            if let Layout::Leaf(_) = lo.content.as_ref() {
                let row = normalize_index(coords[0], lo.len() as i64)?;
                let start = lo.offsets[row] as usize;
                let end = lo.offsets[row + 1] as usize;
                let col = normalize_index(coords[1], (end - start) as i64)?;
                return Ok(start + col);
            }
        }
    }

    // General path for pure list chains (ListOffset* -> Leaf).
    let mut list_offsets: Vec<&ListOffset> = Vec::new();
    let mut cur = layout;
    loop {
        match cur {
            Layout::ListOffset(lo) => {
                list_offsets.push(lo);
                cur = lo.content.as_ref();
            }
            Layout::OffsetView(_v) => {
                // Map to a synthetic ListOffset reference is messy; reject for now.
                return Err(unsupported("coordinate indexing", "OffsetView layouts are not supported yet", "materialize the view with gr.array(...) first."));
            }
            Layout::Indexed(_) => {
                return Err(unsupported("coordinate indexing", "Indexed views are not supported yet", "materialize the view with gr.array(...) first."));
            }
            Layout::Leaf(_) => break,
            Layout::UnionScalarList(_) => {
                return Err(internal("coord_to_leaf_index", "union should be handled above"))
            }
        }
    }
    if list_offsets.is_empty() {
        // 1D leaf
        if coords.len() != 1 {
            return Err(shape_mismatch("coordinate indexing", "coordinate length does not match array depth", "pass one coordinate per nesting axis plus the leaf axis."));
        }
        let leaf = match layout {
            Layout::Leaf(l) => l,
            _ => unreachable!(),
        };
        return normalize_index(coords[0], leaf.len as i64);
    }

    if coords.len() != list_offsets.len() + 1 {
        return Err(shape_mismatch("coordinate indexing", "coordinate length does not match array depth", "pass one coordinate per nesting axis plus the leaf axis."));
    }

    // Select list element at axis 0
    let mut sel = normalize_index(coords[0], list_offsets[0].len() as i64)?;

    // For intermediate axes, map local index to global list index for the next level.
    for axis in 1..list_offsets.len() {
        let prev = list_offsets[axis - 1];
        let start = prev.offsets[sel] as i64;
        let end = prev.offsets[sel + 1] as i64;
        let len = end - start;
        let local = normalize_index(coords[axis], len)?;
        sel = (start as usize) + local;
    }

    // Last axis selects within leaf range from the last ListOffset.
    let last = *list_offsets.last().unwrap();
    let start = last.offsets[sel] as i64;
    let end = last.offsets[sel + 1] as i64;
    let len = end - start;
    let local = normalize_index(*coords.last().unwrap(), len)?;
    Ok((start as usize) + local)
}

fn coord_to_leaf_index_union(u: &UnionScalarList, coords: &[i64]) -> PyResult<usize> {
    if coords.is_empty() {
        return Err(shape_mismatch("coordinate indexing", "coordinate length does not match array depth", "pass one coordinate per nesting axis plus the leaf axis."));
    }
    let outer = normalize_index(coords[0], u.len() as i64)?;
    match u.tags[outer] {
        0 => {
            if coords.len() == 1 {
                return Ok(u.index[outer] as usize);
            }
            normalize_index(coords[1], 1)?;
            if coords.len() > 2 {
                for c in &coords[2..] {
                    normalize_index(*c, 1)?;
                }
            }
            Ok(u.index[outer] as usize)
        }
        1 => {
            let li = u.index[outer] as usize;
            let start = u.lists.offsets[li] as usize;
            let end = u.lists.offsets[li + 1] as usize;
            if coords.len() == 1 {
                return Err(shape_mismatch("coordinate indexing", "coordinate length does not match array depth", "pass one coordinate per nesting axis plus the leaf axis."));
            }
            if coords.len() == 2 {
                let local = normalize_index(coords[1], (end - start) as i64)?;
                return Ok(start + local);
            }
            let sub = take_range(u.lists.content.as_ref(), start, end)?;
            coord_to_leaf_index(&sub, &coords[1..])
        }
        _ => Err(internal("union element", "invalid union tag")),
    }
}

/// Set a scalar value at coordinate indices (mutable coordinate assignment).
pub fn set_encoded_at_coord(
    layout: &mut Layout,
    coords: &[i64],
    valid: bool,
    bytes: &[u8],
) -> PyResult<()> {
    let ix = coord_to_leaf_index(layout, coords)?;
    let leaf = match layout {
        Layout::UnionScalarList(u) => {
            if coords.is_empty() {
                return Err(shape_mismatch(
                    "coordinate assignment",
                    "coordinate length does not match array depth",
                    "pass one coordinate per nesting axis plus the leaf axis.",
                ));
            }
            let outer = normalize_index(coords[0], u.len() as i64)?;
            match u.tags[outer] {
                0 => &mut u.scalars,
                1 => match u.lists.content.as_mut() {
                    Layout::Leaf(l) => l,
                    Layout::ListOffset(lo) => match lo.content.as_mut() {
                        Layout::Leaf(l) => l,
                        _ => {
                            return Err(layout_unsupported(
                                "assignment",
                                "coordinate assignment into nested union lists beyond list->leaf is not supported",
                            ));
                        }
                    },
                    _ => {
                        return Err(layout_unsupported(
                            "assignment",
                            "union list branch has unsupported content layout",
                        ));
                    }
                },
                _ => return Err(internal("union element", "invalid union tag")),
            }
        }
        Layout::ListOffset(lo) => match lo.content.as_mut() {
            Layout::Leaf(l) => l,
            _ => {
                return Err(layout_unsupported(
                    "assignment",
                    "coordinate assignment on nested list chains beyond list->leaf is not supported",
                ));
            }
        },
        Layout::Leaf(l) => l,
        _ => {
            return Err(layout_unsupported(
                "assignment",
                "mutable coordinate indexing is not supported on this layout/view",
            ));
        }
    };
    leaf.set_encoded(ix, valid, bytes)?;
    Ok(())
}

fn normalize_index(i: i64, len: i64) -> PyResult<usize> {
    if len < 0 {
        return Err(internal("normalize_index", "negative length"));
    }
    let mut j = i;
    if j < 0 {
        j += len;
    }
    if j < 0 || j >= len {
        return Err(index_out_of_bounds(
            if i >= 0 { i as usize } else { 0 },
            len as usize,
            "on this axis",
        ));
    }
    Ok(j as usize)
}

pub fn gather_2d_fancy_leaf(layout: &Layout, rows: &[i64], cols: &[i64]) -> PyResult<Leaf> {
    // Only supports depth=2 (ListOffset -> Leaf)
    let depth = list_chain_depth(layout).ok_or_else(|| layout_unsupported("fancy gather", "array is not a pure list chain"))?;
    if depth != 1 {
        return Err(unsupported("fancy gather", "only 2D arrays are supported for now", "reduce dimensionality or use general indexing."));
    }
    if rows.len() != cols.len() {
        return Err(shape_mismatch("fancy gather", "row and column index arrays must have the same length", "ensure len(rows) == len(cols)."));
    }
    let (lo_offsets, lo_content, lo_len, row_base): (&[i64], &Layout, usize, usize) = match layout {
        Layout::ListOffset(lo) => (lo.offsets.as_slice(), lo.content.as_ref(), lo.len(), 0),
        Layout::OffsetView(v) => (&v.offsets, v.content.as_ref(), v.len(), v.start),
        _ => return Err(PyValueError::new_err("Expected list array.")),
    };
    let leaf = match lo_content {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf at depth 2.")),
    };
    let mut out = Leaf::new(leaf.dtype);
    out.len = rows.len();
    Arc::make_mut(&mut out.validity).reserve(rows.len());
    out.buffer.reserve(rows.len());
    out.has_nulls = leaf.has_nulls;

    for k in 0..rows.len() {
        let mut r = rows[k];
        if r < 0 {
            r += lo_len as i64;
        }
        if r < 0 || r >= lo_len as i64 {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let rr = row_base + (r as usize);
        let start = lo_offsets[rr] as usize;
        let end = lo_offsets[rr + 1] as usize;
        let mut c = cols[k];
        let len = (end - start) as i64;
        if c < 0 {
            c += len;
        }
        if c < 0 || c >= len {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let ix = start + c as usize;
        Arc::make_mut(&mut out.validity).push(leaf.validity[ix]);
        out.buffer.push_from_index(&leaf.buffer, ix)?;
    }
    Ok(out)
}

pub fn gather_2d_fancy_sum_i64(layout: &Layout, rows: &[i64], cols: &[i64], dtype: DType) -> PyResult<i64> {
    // Only supports depth=2 (ListOffset -> Leaf)
    let depth = list_chain_depth(layout).ok_or_else(|| layout_unsupported("fancy gather", "array is not a pure list chain"))?;
    if depth != 1 {
        return Err(unsupported("fancy gather", "only 2D arrays are supported for now", "reduce dimensionality or use general indexing."));
    }
    if rows.len() != cols.len() {
        return Err(shape_mismatch("fancy gather", "row and column index arrays must have the same length", "ensure len(rows) == len(cols)."));
    }
    let (lo_offsets, lo_content, lo_len, row_base): (&[i64], &Layout, usize, usize) = match layout {
        Layout::ListOffset(lo) => (lo.offsets.as_slice(), lo.content.as_ref(), lo.len(), 0),
        Layout::OffsetView(v) => (&v.offsets, v.content.as_ref(), v.len(), v.start),
        _ => return Err(PyValueError::new_err("Expected list array.")),
    };
    let leaf = match lo_content {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf at depth 2.")),
    };
    if leaf.dtype != dtype {
        return Err(internal_dtype_buffer_mismatch("gather checksum", dtype));
    }
    if leaf.has_nulls {
        return Err(arg_invalid("gather checksum", "leaf contains nulls", "filter nulls or use a path that skips null entries."));
    }
    if dtype != DType::Int32 && dtype != DType::Int64 {
        return Err(dtype_unsupported("gather checksum", dtype));
    }

    let mut acc: i64 = 0;
    for k in 0..rows.len() {
        let mut r = rows[k];
        if r < 0 {
            r += lo_len as i64;
        }
        if r < 0 || r >= lo_len as i64 {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let rr = row_base + (r as usize);
        let start = lo_offsets[rr] as usize;
        let end = lo_offsets[rr + 1] as usize;
        let mut c = cols[k];
        let len = (end - start) as i64;
        if c < 0 {
            c += len;
        }
        if c < 0 || c >= len {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let ix = start + c as usize;
        match &leaf.buffer {
            LeafBuffer::I32(v) if dtype == DType::Int32 => acc += v[ix] as i64,
            LeafBuffer::I64(v) if dtype == DType::Int64 => acc += v[ix],
            _ => return Err(PyValueError::new_err("Internal error: dtype mismatch in gather checksum.")),
        }
    }
    Ok(acc)
}

pub fn scatter_2d_fancy_numeric(
    py: Python<'_>,
    layout: &mut Layout,
    rows: &[i64],
    cols: &[i64],
    values: &Bound<'_, PyAny>,
    dtype: DType,
) -> PyResult<()> {
    // Only supports depth=2 (ListOffset -> Leaf)
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Expected list array.")),
    };
    let nrows = lo.len() as i64;
    let leaf = match lo.content.as_mut() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf at depth 2.")),
    };
    if rows.len() != cols.len() {
        return Err(shape_mismatch("fancy gather", "row and column index arrays must have the same length", "ensure len(rows) == len(cols)."));
    }
    let n = rows.len();

    // Prepare values: accept NumPy 1D arrays directly (fast), otherwise fall back to Python sequence/scalar.
    // (We only optimize the dtypes we support here; other cases still work via scalar extraction path.)
    enum Values<'a, 'py> {
        Scalar(&'a Bound<'py, PyAny>),
        Seq(Bound<'py, PySequence>),
        NpI32(Vec<i32>),
        NpI64(Vec<i64>),
        NpF64(Vec<f64>),
        NpBool(Vec<bool>),
    }

    let values_parsed: Values<'_, '_> = if let Ok(ro) = values.extract::<numpy::PyReadonlyArray1<'_, i32>>() {
        let s = ro.as_slice()?;
        if s.len() != n {
            return Err(shape_mismatch("fancy assignment", "value length must match number of selected coordinates", "pass a value array with one element per (row, col) pair."));
        }
        Values::NpI32(s.to_vec())
    } else if let Ok(ro) = values.extract::<numpy::PyReadonlyArray1<'_, i64>>() {
        let s = ro.as_slice()?;
        if s.len() != n {
            return Err(shape_mismatch("fancy assignment", "value length must match number of selected coordinates", "pass a value array with one element per (row, col) pair."));
        }
        Values::NpI64(s.to_vec())
    } else if let Ok(ro) = values.extract::<numpy::PyReadonlyArray1<'_, f64>>() {
        let s = ro.as_slice()?;
        if s.len() != n {
            return Err(shape_mismatch("fancy assignment", "value length must match number of selected coordinates", "pass a value array with one element per (row, col) pair."));
        }
        Values::NpF64(s.to_vec())
    } else if let Ok(ro) = values.extract::<numpy::PyReadonlyArray1<'_, bool>>() {
        let s = ro.as_slice()?;
        if s.len() != n {
            return Err(shape_mismatch("fancy assignment", "value length must match number of selected coordinates", "pass a value array with one element per (row, col) pair."));
        }
        Values::NpBool(s.to_vec())
    } else if is_sequence_like(py, values)? {
        let s = values.downcast::<PySequence>()?;
        if s.len()? as usize != n {
            return Err(shape_mismatch("fancy assignment", "value length must match number of selected coordinates", "pass a value array with one element per (row, col) pair."));
        }
        Values::Seq(s.clone())
    } else {
        Values::Scalar(values)
    };

    let sz = dtype.size_bytes();
    for k in 0..n {
        let mut r = rows[k];
        if r < 0 {
            r += nrows;
        }
        if r < 0 || r >= nrows {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let start = lo.offsets[r as usize] as usize;
        let end = lo.offsets[r as usize + 1] as usize;
        let mut c = cols[k];
        let len = (end - start) as i64;
        if c < 0 {
            c += len;
        }
        if c < 0 || c >= len {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let ix = start + c as usize;

        let v_obj;
        let v = match &values_parsed {
            Values::Scalar(s) => *s,
            Values::Seq(s) => {
                v_obj = s.get_item(k as usize)?;
                &v_obj
            }
            Values::NpI32(vs) => {
                v_obj = (vs[k] as i64).into_py(py).into_bound(py);
                &v_obj
            }
            Values::NpI64(vs) => {
                v_obj = vs[k].into_py(py).into_bound(py);
                &v_obj
            }
            Values::NpF64(vs) => {
                v_obj = vs[k].into_py(py).into_bound(py);
                &v_obj
            }
            Values::NpBool(vs) => {
                v_obj = vs[k].into_py(py).into_bound(py);
                &v_obj
            }
        };

        if v.is_none() {
            Arc::make_mut(&mut leaf.validity).set(ix, false);
            leaf.has_nulls = true;
            // Set stored value to zero by overwriting existing slot.
            let zeros = vec![0u8; sz];
            leaf.buffer.set_from_bytes(ix, &zeros, dtype)?;
            continue;
        }

        let bytes = match dtype {
            DType::Int8 => (v.extract::<i64>()? as i8).to_ne_bytes().to_vec(),
            DType::Int16 => (v.extract::<i64>()? as i16).to_ne_bytes().to_vec(),
            DType::Int32 => (v.extract::<i64>()? as i32).to_ne_bytes().to_vec(),
            DType::Int64 => (v.extract::<i64>()?).to_ne_bytes().to_vec(),
            DType::UInt8 => (v.extract::<i64>()? as u8).to_ne_bytes().to_vec(),
            DType::UInt16 => (v.extract::<i64>()? as u16).to_ne_bytes().to_vec(),
            DType::UInt32 => (v.extract::<i64>()? as u32).to_ne_bytes().to_vec(),
            DType::UInt64 => (v.extract::<u64>()?).to_ne_bytes().to_vec(),
            DType::Bool => vec![u8::from(v.extract::<bool>()?)],
            DType::Float16 => {
                let f = v.extract::<f64>()? as f32;
                let h = f16::from_f32(f);
                h.to_bits().to_ne_bytes().to_vec()
            }
            DType::Float32 => (v.extract::<f64>()? as f32).to_ne_bytes().to_vec(),
            DType::Float64 => (v.extract::<f64>()?).to_ne_bytes().to_vec(),
            DType::Char => {
                return Err(dtype_unsupported("scatter", DType::Char))
            }
            DType::String => {
                return Err(PyValueError::new_err(
                    "Vectorized scatter does not support dtype=string.",
                ))
            }
        };

        Arc::make_mut(&mut leaf.validity).set(ix, true);
        leaf.buffer.set_from_bytes(ix, &bytes[..sz], dtype)?;
    }
    Ok(())
}

pub fn scatter_2d_fancy_i32(
    layout: &mut Layout,
    rows: &[i64],
    cols: &[i64],
    values: &[i32],
) -> PyResult<()> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        _ => return Err(PyValueError::new_err("Expected list array.")),
    };
    let nrows = lo.len() as i64;
    let offsets = lo.offsets.as_slice();
    let leaf = match lo.content.as_mut() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf at depth 2.")),
    };
    if leaf.dtype != DType::Int32 {
        return Err(internal_dtype_buffer_mismatch("scatter_2d_fancy_i32", DType::Int32));
    }
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err(shape_mismatch("fancy scatter", "index and value lengths must match", "pass one value per (row, col) pair."));
    }
    let v = match &mut leaf.buffer {
        LeafBuffer::I32(buf) => Arc::make_mut(buf),
        _ => return Err(PyValueError::new_err("Internal error: expected int32 buffer.")),
    };
    for k in 0..rows.len() {
        let mut r = rows[k];
        if r < 0 {
            r += nrows;
        }
        if r < 0 || r >= nrows {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let start = offsets[r as usize] as usize;
        let end = offsets[r as usize + 1] as usize;
        let mut c = cols[k];
        let len = (end - start) as i64;
        if c < 0 {
            c += len;
        }
        if c < 0 || c >= len {
            return Err(index_out_of_bounds_simple("on this axis"));
        }
        let ix = start + c as usize;
        v[ix] = values[k];
    }
    Ok(())
}


pub fn fill_layout_like(
    py: Python<'_>,
    layout: &Layout,
    dtype: DType,
    fill_value: &Bound<'_, PyAny>,
) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => {
            let mut out = Leaf::new(dtype);
            out.len = l.len;
            out.validity = Arc::new(bitvec::bitvec![u8, bitvec::order::Lsb0; 1; l.len]);
            out.has_nulls = false;
            let mut tmp = Leaf::new(dtype);
            push_scalar(py, fill_value, dtype, &mut tmp)?;
            if tmp.len != 1 {
                return Err(internal("fill_layout_like", "fill scalar length mismatch"));
            }
            out.buffer.extend_repeat_first(&tmp.buffer, l.len)?;
            Ok(Layout::Leaf(out))
        }
        Layout::ListOffset(lo) => {
            let content = fill_layout_like(py, lo.content.as_ref(), dtype, fill_value)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = fill_layout_like(py, ix.content.as_ref(), dtype, fill_value)?;
            Ok(Layout::Indexed(Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            let content = fill_layout_like(py, v.content.as_ref(), dtype, fill_value)?;
            Ok(Layout::OffsetView(OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(u) => {
            let scalars_layout = fill_layout_like(py, &Layout::Leaf(u.scalars.clone()), dtype, fill_value)?;
            let scalars = match scalars_layout {
                Layout::Leaf(l) => l,
                _ => unreachable!(),
            };
            let lists_content = fill_layout_like(py, u.lists.content.as_ref(), dtype, fill_value)?;
            let lists = ListOffset {
                offsets: u.lists.offsets.clone(),
                content: Box::new(lists_content),
            };
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars,
                lists,
            }))
        }
    }
}

pub fn concat_to_py_list(py: Python<'_>, arrays: &[GrumpyArray], dim: usize) -> PyResult<PyObject> {
    if arrays.is_empty() {
        return Err(concat_incompatible("cat() requires at least one array", "pass one or more arrays to concatenate."));
    }
    let mut cur = arrays[0].to_py_list(py)?;
    for a in &arrays[1..] {
        let rhs = a.to_py_list(py)?;
        cur = concat_py_any(py, &cur.bind(py), &rhs.bind(py), dim)?;
    }
    Ok(cur)
}

fn concat_py_any(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, dim: usize) -> PyResult<PyObject> {
    if dim == 0 {
        let out = pyo3::types::PyList::empty_bound(py);
        let la = a.downcast::<pyo3::types::PySequence>()?;
        for i in 0..la.len()? {
            out.append(la.get_item(i)?)?;
        }
        let lb = b.downcast::<pyo3::types::PySequence>()?;
        for i in 0..lb.len()? {
            out.append(lb.get_item(i)?)?;
        }
        return Ok(out.into());
    }
    let la = a.downcast::<pyo3::types::PySequence>()?;
    let lb = b.downcast::<pyo3::types::PySequence>()?;
    let na = la.len()?;
    let nb = lb.len()?;
    if na != nb {
        return Err(PyValueError::new_err(format!(
            "cat(dim={}) requires same length along outer axes; got {} and {}.",
            dim, na, nb
        )));
    }
    let out = pyo3::types::PyList::empty_bound(py);
    for i in 0..na {
        let ai = la.get_item(i)?;
        let bi = lb.get_item(i)?;
        // At deeper dims, we require both to be sequences; if not, error.
        if !ai.downcast::<pyo3::types::PySequence>().is_ok()
            || !bi.downcast::<pyo3::types::PySequence>().is_ok()
        {
            return Err(PyValueError::new_err(format!(
                "cat(dim={}) requires list-like elements at axis {}; found non-list at index {}.",
                dim, dim - 1, i
            )));
        }
        let joined = concat_py_any(py, &ai, &bi, dim - 1)?;
        out.append(joined)?;
    }
    Ok(out.into())
}

#[inline]
unsafe fn extract_pyint_i32(ob: *mut pyo3::ffi::PyObject) -> Option<i32> {
    if pyo3::ffi::PyLong_Check(ob) == 0 {
        return None;
    }
    let err_before = pyo3::ffi::PyErr_Occurred();
    let v = pyo3::ffi::PyLong_AsLong(ob);
    if v == -1 {
        let err_after = pyo3::ffi::PyErr_Occurred();
        if !err_after.is_null() {
            if err_before.is_null() {
                pyo3::ffi::PyErr_Clear();
            }
            return None;
        }
    }
    if v < i32::MIN as i64 || v > i32::MAX as i64 {
        return None;
    }
    Some(v as i32)
}

fn try_build_leaf_i32_from_pylist_fast(list: &Bound<'_, PyList>) -> PyResult<Option<Leaf>> {
    let n = list.len();
    if n == 0 {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let item = match list.get_item(i) {
            Ok(x) => x,
            Err(_) => return Ok(None),
        };
        let Some(v) = (unsafe { extract_pyint_i32(item.as_ptr()) }) else {
            return Ok(None);
        };
        values.push(v);
    }
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = values.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; values.len()]);
    leaf.buffer = LeafBuffer::I32(Arc::new(values));
    Ok(Some(leaf))
}

fn try_build_listoffset_i32_from_pylist_fast(
    list: &Bound<'_, PyList>,
    require_uniform_cols: bool,
) -> PyResult<Option<ListOffset>> {
    let nrows = list.len();
    if nrows == 0 {
        return Ok(None);
    }

    let mut row_lens = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let row = match list.get_item(i) {
            Ok(x) => x,
            Err(_) => return Ok(None),
        };
        let row_list = match row.downcast::<PyList>() {
            Ok(x) => x,
            Err(_) => return Ok(None),
        };
        row_lens.push(row_list.len());
    }

    if require_uniform_cols {
        let ncols = row_lens[0];
        if ncols == 0 {
            return Ok(None);
        }
        if row_lens.iter().any(|&len| len != ncols) {
            return Ok(None);
        }
    }

    let total: usize = row_lens.iter().sum();
    let mut offsets = Vec::with_capacity(nrows + 1);
    offsets.push(0);
    let mut values = Vec::with_capacity(total);

    for i in 0..nrows {
        let row = list.get_item(i)?;
        let row_list = row.downcast::<PyList>().map_err(|_| {
            internal("build_layout", "expected list row element")
        })?;
        let row_len = row_lens[i];
        for j in 0..row_len {
            let item = row_list.get_item(j)?;
            let Some(v) = (unsafe { extract_pyint_i32(item.as_ptr()) }) else {
                return Ok(None);
            };
            values.push(v);
        }
        offsets.push(offsets[i] + row_len as i64);
    }

    let n = values.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(values));

    Ok(Some(ListOffset {
        offsets: Arc::new(offsets),
        content: Box::new(Layout::Leaf(leaf)),
    }))
}

fn try_build_rect2d_listoffset_i32_fast(list: &Bound<'_, PyList>) -> PyResult<Option<ListOffset>> {
    try_build_listoffset_i32_from_pylist_fast(list, true)
}

fn try_build_ragged_listoffset_i32_fast(list: &Bound<'_, PyList>) -> PyResult<Option<ListOffset>> {
    try_build_listoffset_i32_from_pylist_fast(list, false)
}

fn try_build_from_numpy(_py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Option<Layout>> {
    if dtype != DType::Int32 {
        return Ok(None);
    }
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray2<'_, i32>>() {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Ok(None);
        }
        let nrows = shape[0];
        let ncols = shape[1];
        if ncols == 0 {
            return Ok(None);
        }
        let slice = arr.as_slice()?;
        let mut offsets = Vec::with_capacity(nrows + 1);
        offsets.push(0);
        for r in 0..nrows {
            offsets.push(offsets[r] + ncols as i64);
        }
        let n = nrows * ncols;
        let mut leaf = Leaf::new(DType::Int32);
        leaf.len = n;
        leaf.has_nulls = false;
        leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
        leaf.buffer = LeafBuffer::I32(Arc::new(slice.to_vec()));
        return Ok(Some(Layout::ListOffset(ListOffset {
            offsets: Arc::new(offsets),
            content: Box::new(Layout::Leaf(leaf)),
        })));
    }
    if let Ok(arr) = obj.extract::<numpy::PyReadonlyArray1<'_, i32>>() {
        let n = arr.len();
        let slice = arr.as_slice()?;
        let mut leaf = Leaf::new(DType::Int32);
        leaf.len = n;
        leaf.has_nulls = false;
        leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
        leaf.buffer = LeafBuffer::I32(Arc::new(slice.to_vec()));
        return Ok(Some(Layout::Leaf(leaf)));
    }
    Ok(None)
}

fn try_build_leaf_i32_from_pylist(_py: Python<'_>, seq: &Bound<'_, PySequence>) -> PyResult<Option<Leaf>> {
    if let Ok(list) = seq.downcast::<PyList>() {
        return try_build_leaf_i32_from_pylist_fast(list);
    }
    let n = seq.len()?;
    if n == 0 {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(n as usize);
    for i in 0..n {
        let item = seq.get_item(i)?;
        if item.is_none() {
            return Ok(None);
        }
        match extract_int::<i32>(&item) {
            Ok(v) => values.push(v),
            Err(_) => return Ok(None),
        }
    }
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = values.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; values.len()]);
    leaf.buffer = LeafBuffer::I32(Arc::new(values));
    Ok(Some(leaf))
}

fn try_build_rect2d_listoffset_i32(_py: Python<'_>, seq: &Bound<'_, PySequence>) -> PyResult<Option<ListOffset>> {
    if let Ok(list) = seq.downcast::<PyList>() {
        if let Some(lo) = try_build_rect2d_listoffset_i32_fast(list)? {
            return Ok(Some(lo));
        }
        return try_build_ragged_listoffset_i32_fast(list);
    }
    let nrows = seq.len()?;
    if nrows == 0 {
        return Ok(None);
    }
    let row0 = seq.get_item(0)?;
    let row0_seq = row0.downcast::<PySequence>().ok();
    let ncols = row0_seq.as_ref().map(|r| r.len()).transpose()?;
    let ncols = match ncols {
        Some(0) | None => return Ok(None),
        Some(n) => n as usize,
    };

    let mut offsets = Vec::with_capacity(nrows as usize + 1);
    offsets.push(0);
    let mut values = Vec::with_capacity(nrows as usize * ncols);

    for i in 0..nrows {
        let row = seq.get_item(i)?;
        let row_seq = row.downcast::<PySequence>().map_err(|_| {
            internal("build_layout", "expected sequence element")
        })?;
        if row_seq.len()? as usize != ncols {
            return Ok(None);
        }
        for j in 0..ncols {
            let item = row_seq.get_item(j)?;
            if item.is_none() {
                return Ok(None);
            }
            match extract_int::<i32>(&item) {
                Ok(v) => values.push(v),
                Err(_) => return Ok(None),
            }
        }
        offsets.push(offsets[i as usize] + ncols as i64);
    }

    let n = values.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(values));

    Ok(Some(ListOffset {
        offsets: Arc::new(offsets),
        content: Box::new(Layout::Leaf(leaf)),
    }))
}

fn build_layout(py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Layout> {
    if is_sequence_like(py, obj)? {
        // Fast path: rectangular nested Python lists of ints (int32) without per-element type probes.
        if dtype == DType::Int32 {
            if let Ok(list) = obj.downcast::<PyList>() {
                if list.len() > 0 {
                    let row0 = list.get_item(0)?;
                    if row0.downcast::<PyList>().is_ok() {
                        if let Some(lo) = try_build_rect2d_listoffset_i32_fast(list)? {
                            return Ok(Layout::ListOffset(lo));
                        }
                        if let Some(lo) = try_build_ragged_listoffset_i32_fast(list)? {
                            return Ok(Layout::ListOffset(lo));
                        }
                    } else if let Some(leaf) = try_build_leaf_i32_from_pylist_fast(list)? {
                        return Ok(Layout::Leaf(leaf));
                    }
                }
            }
        }

        let seq = obj.downcast::<PySequence>()?;
        let n = seq.len()?;
        if n == 0 {
            // Empty: represent as an empty leaf for 1D empties, or empty list if used as a nested element.
            return Ok(Layout::Leaf(Leaf::new(dtype)));
        }

        let mut flags = Vec::with_capacity(n);
        for i in 0..n {
            let item = seq.get_item(i)?;
            flags.push(is_sequence_like(py, &item)?);
        }

        if flags.iter().all(|b| !*b) {
            if dtype == DType::Int32 {
                if let Some(leaf) = try_build_leaf_i32_from_pylist(py, seq)? {
                    return Ok(Layout::Leaf(leaf));
                }
            }
            return Ok(Layout::Leaf(build_leaf_from_scalars(py, seq, dtype)?));
        }
        if flags.iter().all(|b| *b) {
            if dtype == DType::Int32 {
                if let Some(lo) = try_build_rect2d_listoffset_i32(py, seq)? {
                    return Ok(Layout::ListOffset(lo));
                }
            }
            return Ok(Layout::ListOffset(build_listoffset_from_lists(py, seq, dtype)?));
        }

        Ok(Layout::UnionScalarList(build_union_scalar_list(py, seq, dtype, &flags)?))
    } else {
        // Scalar: construct a length-1 leaf array.
        let mut leaf = Leaf::new(dtype);
        push_scalar(py, obj, dtype, &mut leaf)?;
        Ok(Layout::Leaf(leaf))
    }
}

fn build_leaf_from_scalars(
    py: Python<'_>,
    seq: &Bound<'_, PySequence>,
    dtype: DType,
) -> PyResult<Leaf> {
    let n = seq.len()?;
    let mut leaf = Leaf::new(dtype);
    leaf.buffer.reserve(n);
    Arc::make_mut(&mut leaf.validity).reserve(n);
    for i in 0..n {
        let item = seq.get_item(i)?;
        push_scalar(py, &item, dtype, &mut leaf)?;
    }
    Ok(leaf)
}

fn build_listoffset_from_lists(
    py: Python<'_>,
    seq: &Bound<'_, PySequence>,
    dtype: DType,
) -> PyResult<ListOffset> {
    let n = seq.len()?;
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0);

    // Flatten all child elements into a Vec<PyObject> so we can recursively build content.
    let mut flat_children: Vec<PyObject> = Vec::new();

    for i in 0..n {
        let item = seq.get_item(i)?;
        let child_seq = item.downcast::<PySequence>().map_err(|_| {
            internal("build_layout", "expected sequence element")
        })?;
        let m = child_seq.len()?;
        for j in 0..m {
            flat_children.push(child_seq.get_item(j)?.into());
        }
        let last = *offsets.last().unwrap();
        offsets.push(last + m as i64);
    }

    let content_obj = pyo3::types::PyList::new_bound(py, flat_children);
    let content = build_layout(py, &content_obj, dtype)?;

    Ok(ListOffset {
        offsets: Arc::new(offsets),
        content: Box::new(content),
    })
}

fn build_union_scalar_list(
    py: Python<'_>,
    seq: &Bound<'_, PySequence>,
    dtype: DType,
    flags_is_list: &[bool],
) -> PyResult<UnionScalarList> {
    let n = seq.len()?;
    debug_assert_eq!(n, flags_is_list.len());

    let mut tags = Vec::with_capacity(n);
    let mut index = Vec::with_capacity(n);

    let mut scalar_items: Vec<PyObject> = Vec::new();
    let mut list_items: Vec<PyObject> = Vec::new();

    let mut scalar_ix = 0i64;
    let mut list_ix = 0i64;
    for i in 0..n {
        let item = seq.get_item(i)?;
        if flags_is_list[i] {
            tags.push(1);
            index.push(list_ix);
            list_items.push(item.into());
            list_ix += 1;
        } else {
            tags.push(0);
            index.push(scalar_ix);
            scalar_items.push(item.into());
            scalar_ix += 1;
        }
    }

    let scalar_list = pyo3::types::PyList::new_bound(py, scalar_items);
    let scalar_layout = build_layout(py, &scalar_list, dtype)?;
    let scalars = match scalar_layout {
        Layout::Leaf(l) => l,
        _ => {
            return Err(PyValueError::new_err(
                "Internal error: scalar branch did not build into a leaf.",
            ))
        }
    };

    let list_list = pyo3::types::PyList::new_bound(py, list_items);
    let list_layout = build_layout(py, &list_list, dtype)?;
    let lists = match list_layout {
        Layout::ListOffset(lo) => lo,
        Layout::OffsetView(_v) => {
            return Err(PyValueError::new_err(
                "Internal error: list branch built into an OffsetView.",
            ))
        }
        Layout::Indexed(_) => {
            return Err(PyValueError::new_err(
                "Internal error: list branch built into an Indexed view.",
            ))
        }
        Layout::Leaf(_) => {
            return Err(PyValueError::new_err(
                "Internal error: list branch built into a leaf.",
            ))
        }
        Layout::UnionScalarList(_) => {
            return Err(PyValueError::new_err(
                "Internal error: list branch built into a scalar/list union.",
            ))
        }
    };

    Ok(UnionScalarList {
        tags,
        index,
        scalars,
        lists,
    })
}

fn push_scalar(_py: Python<'_>, obj: &Bound<'_, PyAny>, dtype: DType, leaf: &mut Leaf) -> PyResult<()> {
    if obj.is_none() {
        leaf.push_null();
        return Ok(());
    }

    if dtype == DType::String {
        let s = obj
            .extract::<String>()
            .map_err(|_| arg_invalid("value", "expected a Python string for dtype=string", "pass str values for string arrays."))?;
        Arc::make_mut(&mut leaf.validity).push(true);
        leaf.len += 1;
        match &mut leaf.buffer {
            LeafBuffer::String(v) => Arc::make_mut(v).push(s),
            _ => return Err(PyValueError::new_err("Internal dtype mismatch (string).")),
        }
        return Ok(());
    }

    let bytes = match dtype {
        DType::Bool => {
            let v = obj.extract::<bool>()?;
            vec![u8::from(v)]
        }
        DType::Int8 => {
            let v = extract_int::<i8>(obj)?;
            v.to_ne_bytes().to_vec()
        }
        DType::Int16 => extract_int::<i16>(obj)?.to_ne_bytes().to_vec(),
        DType::Int32 => extract_int::<i32>(obj)?.to_ne_bytes().to_vec(),
        DType::Int64 => extract_int::<i64>(obj)?.to_ne_bytes().to_vec(),
        DType::UInt8 => extract_uint::<u8>(obj)?.to_ne_bytes().to_vec(),
        DType::UInt16 => extract_uint::<u16>(obj)?.to_ne_bytes().to_vec(),
        DType::UInt32 => extract_uint::<u32>(obj)?.to_ne_bytes().to_vec(),
        DType::UInt64 => extract_uint::<u64>(obj)?.to_ne_bytes().to_vec(),
        DType::Float16 => {
            let v = extract_float64(obj)? as f32;
            let h = f16::from_f32(v);
            h.to_bits().to_ne_bytes().to_vec()
        }
        DType::Float32 => (extract_float64(obj)? as f32).to_ne_bytes().to_vec(),
        DType::Float64 => extract_float64(obj)?.to_ne_bytes().to_vec(),
        DType::Char => {
            let s = obj.extract::<String>()?;
            let mut it = s.chars();
            let c = it
                .next()
                .ok_or_else(|| arg_invalid("char", "empty string is not a valid char", "pass a single-character string."))?;
            if it.next().is_some() {
                return Err(PyValueError::new_err(
                    "Only single-character strings are allowed for dtype=char.",
                ));
            }
            (c as u32).to_ne_bytes().to_vec()
        }
        DType::String => unreachable!(),
    };

    leaf.push_value(&bytes)?;
    Ok(())
}

fn extract_int<T>(obj: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: TryFrom<i128>,
{
    if obj.is_instance_of::<PyBool>() {
        let v = obj.extract::<bool>()?;
        let i = if v { 1i128 } else { 0i128 };
        return T::try_from(i).map_err(|_| PyValueError::new_err("Integer overflow."));
    }
    if obj.is_instance_of::<PyInt>() {
        let i = obj.extract::<i128>()?;
        return T::try_from(i).map_err(|_| PyValueError::new_err("Integer overflow."));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        if !f.is_finite() || f.fract() != 0.0 {
            return Err(PyValueError::new_err(
                "Cannot cast non-integer float to integer dtype.",
            ));
        }
        return T::try_from(f as i128).map_err(|_| PyValueError::new_err("Integer overflow."));
    }
    Err(PyValueError::new_err("Expected an int-like value."))
}

fn extract_uint<T>(obj: &Bound<'_, PyAny>) -> PyResult<T>
where
    T: TryFrom<u128>,
{
    if obj.is_instance_of::<PyBool>() {
        let v = obj.extract::<bool>()?;
        let u = if v { 1u128 } else { 0u128 };
        return T::try_from(u).map_err(|_| PyValueError::new_err("Integer overflow."));
    }
    if obj.is_instance_of::<PyInt>() {
        let i = obj.extract::<i128>()?;
        if i < 0 {
            return Err(PyValueError::new_err(
                "Cannot cast negative value to unsigned dtype.",
            ));
        }
        return T::try_from(i as u128).map_err(|_| PyValueError::new_err("Integer overflow."));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        if !f.is_finite() || f.fract() != 0.0 || f < 0.0 {
            return Err(PyValueError::new_err(
                "Cannot cast float to unsigned integer dtype.",
            ));
        }
        return T::try_from(f as u128).map_err(|_| PyValueError::new_err("Integer overflow."));
    }
    Err(PyValueError::new_err("Expected an int-like value."))
}

fn extract_float64(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    if obj.is_instance_of::<PyBool>() {
        let v = obj.extract::<bool>()?;
        return Ok(if v { 1.0 } else { 0.0 });
    }
    if obj.is_instance_of::<PyInt>() {
        let i = obj.extract::<i128>()?;
        return Ok(i as f64);
    }
    if obj.is_instance_of::<PyFloat>() {
        let f = obj.extract::<f64>()?;
        return Ok(f);
    }
    Err(PyValueError::new_err("Expected a float-like value."))
}

