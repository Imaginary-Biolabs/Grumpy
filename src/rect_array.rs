//! Rectangular tensor layout analysis and conversion for framework interop.
//!
//! Only pure list-chains with uniform row lengths at every axis (no unions, no nulls)
//! can be exported as dense C-contiguous tensors.

use crate::dtype::DType;
use crate::error::{arg_invalid, dtype_unsupported, layout_unsupported, shape_mismatch, unsupported};
use crate::layout::{take_range, Leaf, LeafBuffer, Layout, ListOffset};
use bitvec::prelude::*;
use numpy::{dtype_bound, PyArrayDescr, PyArrayDescrMethods};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use std::sync::Arc;

/// Dense rectangular tensor metadata referencing a contiguous leaf slice.
pub struct RectTensor<'a> {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub leaf: &'a Leaf,
    /// Element offset into `leaf` (not bytes).
    pub start: usize,
    pub count: usize,
}

/// Result of validating and preparing a rectangular tensor for export.
pub enum RectExport<'a> {
    /// Zero-copy view into an existing leaf buffer.
    View(RectTensor<'a>),
    /// Gathered row-major bytes when the layout is rectangular but not contiguous.
    Gathered {
        shape: Vec<usize>,
        dtype: DType,
        bytes: Vec<u8>,
    },
}

pub fn interop_dtype_supported(dt: DType) -> bool {
    !matches!(dt, DType::Char | DType::String)
}

pub fn rectangular_shape(layout: &Layout) -> PyResult<Vec<usize>> {
    if layout.has_union() {
        return Err(shape_mismatch(
            "interop",
            "union layouts are not rectangular tensors",
            "use a pure list-chain without UnionScalarList, or materialize rows explicitly.",
        ));
    }
    if has_nulls(layout) {
        return Err(shape_mismatch(
            "interop",
            "null values are not supported in rectangular tensor conversion",
            "filter or fill nulls before converting to a dense tensor.",
        ));
    }
    shape_of(layout)
}

pub fn export_rectangular(layout: &Layout) -> PyResult<RectExport<'_>> {
    let shape = rectangular_shape(layout)?;
    let dtype = leaf_dtype(layout)?;
    if !interop_dtype_supported(dtype) {
        return Err(dtype_unsupported("interop", dtype));
    }
    let expected: usize = shape.iter().product();
    if let Some(view) = try_contiguous_leaf_view(layout)? {
        if view.count != expected {
            return Err(shape_mismatch(
                "interop",
                "rectangular tensor leaf range does not match shape",
                "ensure the layout is a dense C-contiguous list-chain.",
            ));
        }
        return Ok(RectExport::View(RectTensor {
            shape,
            dtype,
            leaf: view.leaf,
            start: view.start,
            count: view.count,
        }));
    }
    let bytes = gather_bytes(layout, &shape, dtype)?;
    Ok(RectExport::Gathered {
        shape,
        dtype,
        bytes,
    })
}

struct ContiguousView<'a> {
    leaf: &'a Leaf,
    start: usize,
    count: usize,
}

fn has_nulls(layout: &Layout) -> bool {
    match layout {
        Layout::Leaf(l) => l.has_nulls,
        Layout::ListOffset(lo) => has_nulls(lo.content.as_ref()),
        Layout::Indexed(ix) => has_nulls(ix.content.as_ref()),
        Layout::OffsetView(v) => has_nulls(v.content.as_ref()),
        Layout::UnionScalarList(_) => true,
    }
}

fn leaf_dtype(layout: &Layout) -> PyResult<DType> {
    match layout {
        Layout::Leaf(l) => Ok(l.dtype),
        Layout::ListOffset(lo) => leaf_dtype(lo.content.as_ref()),
        Layout::Indexed(ix) => leaf_dtype(ix.content.as_ref()),
        Layout::OffsetView(v) => leaf_dtype(v.content.as_ref()),
        Layout::UnionScalarList(_) => Err(layout_unsupported(
            "interop",
            "UnionScalarList layout is not a rectangular tensor",
        )),
    }
}

fn shape_of(layout: &Layout) -> PyResult<Vec<usize>> {
    match layout {
        Layout::Leaf(l) => Ok(vec![l.len]),
        Layout::ListOffset(_) | Layout::OffsetView(_) | Layout::Indexed(_) => {
            let n = outer_len(layout);
            if n == 0 {
                return Ok(vec![0]);
            }
            let child0 = element_layout(layout, 0)?;
            let mut child_shape = shape_of(&child0)?;
            for i in 1..n {
                let child = element_layout(layout, i)?;
                let s = shape_of(&child)?;
                if s != child_shape {
                    return Err(shape_mismatch(
                        "interop",
                        "ragged nesting: inner dimensions differ across outer indices",
                        "ensure every row/sub-array has the same shape at each list level.",
                    ));
                }
            }
            let mut shape = vec![n];
            shape.append(&mut child_shape);
            Ok(shape)
        }
        Layout::UnionScalarList(_) => Err(layout_unsupported(
            "interop",
            "UnionScalarList layout is not a rectangular tensor",
        )),
    }
}

fn outer_len(layout: &Layout) -> usize {
    match layout {
        Layout::ListOffset(lo) => lo.len(),
        Layout::OffsetView(v) => v.len(),
        Layout::Indexed(ix) => ix.len(),
        Layout::Leaf(l) => l.len,
        Layout::UnionScalarList(u) => u.len(),
    }
}

fn element_layout(layout: &Layout, idx: usize) -> PyResult<Layout> {
    match layout {
        Layout::ListOffset(lo) => {
            if idx >= lo.len() {
                return Err(shape_mismatch(
                    "interop",
                    "internal error while analyzing rectangular shape",
                    "report this as a bug.",
                ));
            }
            take_range(
                lo.content.as_ref(),
                lo.offsets[idx] as usize,
                lo.offsets[idx + 1] as usize,
            )
        }
        Layout::OffsetView(v) => {
            if idx >= v.len() {
                return Err(shape_mismatch(
                    "interop",
                    "internal error while analyzing rectangular shape",
                    "report this as a bug.",
                ));
            }
            let abs = v.start + idx;
            let start = v.offsets[abs] as usize;
            let end = v.offsets[abs + 1] as usize;
            take_range(v.content.as_ref(), start, end)
        }
        Layout::Indexed(ix) => {
            if idx >= ix.len() {
                return Err(shape_mismatch(
                    "interop",
                    "internal error while analyzing rectangular shape",
                    "report this as a bug.",
                ));
            }
            let n = ix.content.len() as i64;
            let mut j = ix.index[idx];
            if j < 0 {
                j += n;
            }
            if j < 0 || j >= n {
                return Err(shape_mismatch(
                    "interop",
                    "indexed view is out of bounds while analyzing shape",
                    "ensure indices are valid before tensor conversion.",
                ));
            }
            element_layout(ix.content.as_ref(), j as usize)
        }
        Layout::Leaf(l) => take_range(layout, idx, idx + 1.min(l.len)),
        Layout::UnionScalarList(_) => Err(layout_unsupported(
            "interop",
            "UnionScalarList layout is not a rectangular tensor",
        )),
    }
}

fn try_contiguous_leaf_view(layout: &Layout) -> PyResult<Option<ContiguousView<'_>>> {
    if has_indexed(layout) {
        return Ok(None);
    }
    match layout {
        Layout::Leaf(l) => Ok(Some(ContiguousView {
            leaf: l,
            start: 0,
            count: l.len,
        })),
        Layout::OffsetView(v) => match v.content.as_ref() {
            Layout::Leaf(l) => {
                let start = v.offsets[v.start] as usize;
                let end = v.offsets[v.stop] as usize;
                Ok(Some(ContiguousView {
                    leaf: l,
                    start,
                    count: end.saturating_sub(start),
                }))
            }
            Layout::ListOffset(lo) => {
                if !is_regular_listoffset_in_range(lo, v.start, v.stop)? {
                    return Ok(None);
                }
                let child_start = v.offsets[v.start] as usize;
                let child_end = v.offsets[v.stop] as usize;
                match lo.content.as_ref() {
                    Layout::Leaf(l) => Ok(Some(ContiguousView {
                        leaf: l,
                        start: child_start,
                        count: child_end.saturating_sub(child_start),
                    })),
                    Layout::ListOffset(inner) => {
                        if !is_regular_listoffset(inner) {
                            return Ok(None);
                        }
                        let leaf = match inner.content.as_ref() {
                            Layout::Leaf(l) => l,
                            _ => return Ok(None),
                        };
                        Ok(Some(ContiguousView {
                            leaf,
                            start: child_start,
                            count: child_end.saturating_sub(child_start),
                        }))
                    }
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        },
        Layout::ListOffset(lo) => {
            if !is_regular_listoffset(lo) {
                return Ok(None);
            }
            try_contiguous_leaf_view(lo.content.as_ref())
        }
        Layout::Indexed(_) | Layout::UnionScalarList(_) => Ok(None),
    }
}

fn is_regular_listoffset_in_range(lo: &ListOffset, start: usize, stop: usize) -> PyResult<bool> {
    if start > stop || stop > lo.len() {
        return Ok(false);
    }
    if stop == start {
        return Ok(true);
    }
    let step = lo.offsets[start + 1] - lo.offsets[start];
    if step <= 0 {
        return Ok(false);
    }
    for r in start..stop {
        if lo.offsets[r + 1] - lo.offsets[r] != step {
            return Ok(false);
        }
    }
    if lo.offsets[start] != lo.offsets[stop] - step * (stop - start) as i64 {
        return Ok(false);
    }
    Ok(is_regular_listoffset_content(lo.content.as_ref(), step))
}

fn has_indexed(layout: &Layout) -> bool {
    match layout {
        Layout::Indexed(_) => true,
        Layout::ListOffset(lo) => has_indexed(lo.content.as_ref()),
        Layout::OffsetView(v) => has_indexed(v.content.as_ref()),
        Layout::Leaf(_) => false,
        Layout::UnionScalarList(_) => true,
    }
}

fn is_regular_listoffset(lo: &ListOffset) -> bool {
    let n = lo.len();
    if n == 0 {
        return true;
    }
    let step = lo.offsets[1] - lo.offsets[0];
    if step <= 0 {
        return false;
    }
    for r in 0..n {
        if lo.offsets[r + 1] - lo.offsets[r] != step {
            return false;
        }
    }
    for r in 0..=n {
        if lo.offsets[r] != step * r as i64 {
            return false;
        }
    }
    is_regular_listoffset_content(lo.content.as_ref(), step)
}

fn is_regular_listoffset_content(content: &Layout, expected_step: i64) -> bool {
    match content {
        Layout::ListOffset(lo) => {
            if !is_regular_listoffset(lo) {
                return false;
            }
            let inner_step = lo.offsets[1] - lo.offsets[0];
            is_regular_listoffset_content(lo.content.as_ref(), inner_step)
        }
        Layout::Leaf(_) => true,
        Layout::OffsetView(v) => match v.content.as_ref() {
            Layout::Leaf(_) => {
                let start = v.offsets[v.start] as usize;
                let end = v.offsets[v.stop] as usize;
                (end - start) as i64 == expected_step
            }
            _ => false,
        },
        Layout::Indexed(_) | Layout::UnionScalarList(_) => false,
    }
}

fn gather_bytes(layout: &Layout, shape: &[usize], dtype: DType) -> PyResult<Vec<u8>> {
    let elem_size = dtype.size_bytes();
    let total: usize = shape.iter().product();
    let mut out = vec![0u8; total * elem_size];
    flatten_to_bytes(layout, shape, dtype, &mut out, 0)?;
    Ok(out)
}

fn flatten_to_bytes(
    layout: &Layout,
    shape: &[usize],
    dtype: DType,
    out: &mut [u8],
    out_offset_elems: usize,
) -> PyResult<()> {
    let elem_size = dtype.size_bytes();
    if shape.is_empty() {
        return Err(shape_mismatch(
            "interop",
            "empty shape while gathering tensor bytes",
            "pass a non-empty rectangular array.",
        ));
    }
    if shape.len() == 1 {
        let n = shape[0];
        match layout {
            Layout::Leaf(l) => {
                copy_leaf_range(l, 0, n, dtype, out, out_offset_elems)?;
            }
            Layout::OffsetView(v) => match v.content.as_ref() {
                Layout::Leaf(l) => {
                    let start = v.offsets[v.start] as usize;
                    copy_leaf_range(l, start, n, dtype, out, out_offset_elems)?;
                }
                _ => {
                    for i in 0..n {
                        let child = element_layout(layout, i)?;
                        flatten_to_bytes(&child, &[], dtype, out, out_offset_elems + i)?;
                    }
                }
            },
            _ => {
                for i in 0..n {
                    let child = element_layout(layout, i)?;
                    flatten_to_bytes(&child, &[], dtype, out, out_offset_elems + i)?;
                }
            }
        }
        return Ok(());
    }
    let (head, tail) = shape.split_first().expect("shape len checked");
    let mut elem_offset = out_offset_elems;
    for i in 0..*head {
        let child = element_layout(layout, i)?;
        let tail_elems: usize = tail.iter().product();
        flatten_to_bytes(&child, tail, dtype, out, elem_offset)?;
        elem_offset += tail_elems;
    }
    Ok(())
}

fn copy_leaf_range(
    leaf: &Leaf,
    start: usize,
    count: usize,
    dtype: DType,
    out: &mut [u8],
    out_offset_elems: usize,
) -> PyResult<()> {
    let elem_size = dtype.size_bytes();
    let bytes = leaf.buffer.as_bytes();
    let byte_start = start * elem_size;
    let byte_end = byte_start + count * elem_size;
    if byte_end > bytes.len() || out_offset_elems * elem_size + count * elem_size > out.len() {
        return Err(shape_mismatch(
            "interop",
            "leaf byte range does not match requested tensor shape",
            "ensure the layout is rectangular and all-valid.",
        ));
    }
    let dst = out_offset_elems * elem_size;
    out[dst..dst + count * elem_size].copy_from_slice(&bytes[byte_start..byte_end]);
    Ok(())
}

pub fn layout_from_shape_and_bytes(shape: &[usize], dtype: DType, bytes: &[u8]) -> PyResult<Layout> {
    if !interop_dtype_supported(dtype) {
        return Err(dtype_unsupported("interop", dtype));
    }
    let expected: usize = shape.iter().product();
    let expected_bytes = expected * dtype.size_bytes();
    if bytes.len() != expected_bytes {
        return Err(arg_invalid(
            "data",
            format!(
                "byte length {} does not match shape {:?} and dtype {:?} (expected {} bytes)",
                bytes.len(),
                shape,
                dtype,
                expected_bytes
            ),
            "pass a C-contiguous buffer matching the array shape and dtype.",
        ));
    }
    let leaf = leaf_from_bytes(dtype, bytes)?;
    Ok(wrap_list_chain(shape, leaf))
}

fn wrap_list_chain(shape: &[usize], leaf: Leaf) -> Layout {
    if shape.len() == 1 {
        return Layout::Leaf(leaf);
    }
    let mut content = Layout::Leaf(leaf);
    for i in (0..shape.len() - 1).rev() {
        let nlists: usize = shape[0..=i].iter().product();
        let step: usize = if i == shape.len() - 2 {
            shape[shape.len() - 1]
        } else {
            shape[i + 1..shape.len() - 1].iter().product()
        };
        let mut offsets = Vec::with_capacity(nlists + 1);
        for j in 0..=nlists {
            offsets.push((j as i64) * step as i64);
        }
        content = Layout::ListOffset(ListOffset {
            offsets: Arc::new(offsets),
            content: Box::new(content),
        });
    }
    content
}

pub fn leaf_from_bytes(dtype: DType, bytes: &[u8]) -> PyResult<Leaf> {
    let n = bytes.len() / dtype.size_bytes();
    let mut leaf = Leaf::new(dtype);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = buffer_from_bytes(dtype, bytes)?;
    Ok(leaf)
}

fn buffer_from_bytes(dtype: DType, bytes: &[u8]) -> PyResult<LeafBuffer> {
    let n = bytes.len() / dtype.size_bytes();
    macro_rules! from_typed {
        ($t:ty, $variant:ident) => {{
            let slice: &[$t] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const $t, n)
            };
            Ok(LeafBuffer::$variant(Arc::new(slice.to_vec())))
        }};
    }
    match dtype {
        DType::Int8 => from_typed!(i8, I8),
        DType::Int16 => from_typed!(i16, I16),
        DType::Int32 => from_typed!(i32, I32),
        DType::Int64 => from_typed!(i64, I64),
        DType::UInt8 => from_typed!(u8, U8),
        DType::UInt16 => from_typed!(u16, U16),
        DType::UInt32 => from_typed!(u32, U32),
        DType::UInt64 => from_typed!(u64, U64),
        DType::Float16 => from_typed!(u16, F16),
        DType::Float32 => from_typed!(f32, F32),
        DType::Float64 => from_typed!(f64, F64),
        DType::Bool => from_typed!(u8, Bool),
        DType::Char | DType::String => Err(dtype_unsupported("interop", dtype)),
    }
}

pub fn numpy_dtype_to_grumpy(py: Python<'_>, descr: &Bound<'_, PyArrayDescr>) -> PyResult<DType> {
    macro_rules! check {
        ($t:ty, $dt:expr) => {
            if descr.is_equiv_to(&dtype_bound::<$t>(py)) {
                return Ok($dt);
            }
        };
    }
    check!(i8, DType::Int8);
    check!(i16, DType::Int16);
    check!(i32, DType::Int32);
    check!(i64, DType::Int64);
    check!(u8, DType::UInt8);
    check!(u16, DType::UInt16);
    check!(u32, DType::UInt32);
    check!(u64, DType::UInt64);
    // half not in Element by default - check float16 via name
    let name: String = descr.getattr("name")?.extract()?;
    if name == "float16" {
        return Ok(DType::Float16);
    }
    check!(f32, DType::Float32);
    check!(f64, DType::Float64);
    check!(bool, DType::Bool);
    Err(unsupported(
        "from_numpy",
        format!("unsupported NumPy dtype '{name}' for Grumpy tensor import"),
        "use a numeric or bool NumPy array with C-contiguous storage.",
    ))
}
