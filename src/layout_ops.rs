//! Shared layout navigation, axis mapping, and concatenation helpers.

use crate::dtype::DType;
use crate::error::{
    concat_incompatible, dtype_mismatch, internal, internal_dtype_buffer_mismatch, layout_unsupported,
    shape_mismatch,
};
use crate::layout::{
    drop_axis0_select_element, leaf_view, stack_axis0_broadcast, GrumpyArray, Indexed, Layout,
    Leaf, LeafBuffer, ListOffset, OffsetView, UnionScalarList,
};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Peel views until a 1D leaf is reached.
pub fn leaf_1d<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::OffsetView(v) => leaf_1d(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_1d(ix.content.as_ref()),
        Layout::ListOffset(_) => Err(layout_unsupported("leaf_1d", "expected a 1D leaf array")),
        Layout::UnionScalarList(_) => Err(layout_unsupported("leaf_1d", "union arrays are not supported here")),
    }
}

/// Peel views until a 2D list→leaf structure is reached.
pub fn listoffset_leaf2d<'a>(layout: &'a Layout) -> PyResult<(&'a ListOffset, &'a Leaf)> {
    match layout {
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(l) => Ok((lo, l)),
            _ => Err(layout_unsupported("listoffset_leaf2d", "expected 2D list->leaf array")),
        },
        Layout::OffsetView(v) => listoffset_leaf2d(v.content.as_ref()),
        Layout::Indexed(ix) => listoffset_leaf2d(ix.content.as_ref()),
        _ => Err(layout_unsupported("listoffset_leaf2d", "expected 2D list->leaf array")),
    }
}

/// Return a 1D leaf for an array, flattening short views when needed.
pub fn array_as_leaf_1d(x: &GrumpyArray) -> PyResult<Leaf> {
    if let Ok(l) = leaf_1d(&x.layout) {
        Ok(l.clone())
    } else {
        leaf_view(&x.layout, x.dtype)
    }
}

pub fn new_leaf_i64_from(v: Vec<i64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I64(Arc::new(v));
    leaf
}

pub fn new_leaf_i32_from(v: Vec<i32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I32(Arc::new(v));
    leaf
}

pub fn new_leaf_u32_from(v: Vec<u32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U32(Arc::new(v));
    leaf
}

pub fn new_leaf_u64_from(v: Vec<u64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::UInt64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::U64(Arc::new(v));
    leaf
}

pub fn new_leaf_f32_from(v: Vec<f32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float32);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F32(Arc::new(v));
    leaf
}

pub fn new_leaf_f64_from(v: Vec<f64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::F64(Arc::new(v));
    leaf
}

pub fn new_leaf_bool_from(v: Vec<u8>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Bool);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::Bool(Arc::new(v));
    leaf
}

pub fn new_leaf_char_from(v: Vec<u32>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Char);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::Char(Arc::new(v));
    leaf
}

/// How to treat a bare leaf encountered at the innermost axis during recursive mapping.
pub enum LastAxisLeafMode {
    /// len <= 1: unchanged; len > 1: wrap as single-row ListOffset then process.
    PromoteShortLeaf,
    /// Bare leaf at this level is an error (partition expects list structure).
    RequireListOffset,
}

/// Recursively map the innermost list→leaf axis, rebuilding wrapper nodes.
pub fn map_last_axis(
    layout: &Layout,
    leaf_mode: LastAxisLeafMode,
    on_listoffset_leaf: &dyn Fn(&ListOffset, &Leaf) -> PyResult<Layout>,
) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => match leaf_mode {
            LastAxisLeafMode::PromoteShortLeaf => {
                if l.len <= 1 {
                    return Ok(layout.clone());
                }
                let lo = ListOffset {
                    offsets: Arc::new(vec![0i64, l.len as i64]),
                    content: Box::new(Layout::Leaf(l.clone())),
                };
                on_listoffset_leaf(&lo, l)
            }
            LastAxisLeafMode::RequireListOffset => Err(internal(
                "map_last_axis",
                "expected list structure at the innermost axis",
            )),
        },
        Layout::ListOffset(lo) => match lo.content.as_ref() {
            Layout::Leaf(leaf) => on_listoffset_leaf(lo, leaf),
            _ => Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(map_last_axis(
                    lo.content.as_ref(),
                    leaf_mode,
                    on_listoffset_leaf,
                )?),
            })),
        },
        Layout::UnionScalarList(u) => {
            let list_content = map_last_axis(u.lists.content.as_ref(), leaf_mode, on_listoffset_leaf)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: u.tags.clone(),
                index: u.index.clone(),
                scalars: u.scalars.clone(),
                lists: ListOffset {
                    offsets: u.lists.offsets.clone(),
                    content: Box::new(list_content),
                },
            }))
        }
        Layout::OffsetView(v) => {
            let content = map_last_axis(v.content.as_ref(), leaf_mode, on_listoffset_leaf)?;
            Ok(Layout::OffsetView(OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = map_last_axis(ix.content.as_ref(), leaf_mode, on_listoffset_leaf)?;
            Ok(Layout::Indexed(Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
    }
}

/// Map a function over each outer element of a union array and stack results.
pub fn map_union_axis0(
    layout: &Layout,
    dtype: DType,
    mut f: impl FnMut(Layout) -> PyResult<Layout>,
) -> PyResult<Layout> {
    let n = layout.len();
    let mut segs: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let sub = drop_axis0_select_element(layout, i)?;
        segs.push(f(sub)?);
    }
    stack_axis0_broadcast(&segs, dtype)
}

fn union_scalar_dtype_from_list(lo: &ListOffset) -> PyResult<DType> {
    match lo.content.as_ref() {
        Layout::Leaf(l) => Ok(l.dtype),
        Layout::ListOffset(inner) => union_scalar_dtype_from_list(inner),
        _ => Err(concat_incompatible(
            "cannot infer dtype for list layout during axis-0 concat",
            "ensure list content is a leaf or nested list with a known dtype.",
        )),
    }
}

fn lift_listoffset_to_union(lo: &ListOffset, dt: DType) -> UnionScalarList {
    let n = lo.len();
    UnionScalarList {
        tags: vec![1u8; n],
        index: (0..n as i64).collect(),
        scalars: Leaf::new(dt),
        lists: lo.clone(),
    }
}

fn concat_union_scalar_lists_axis0(layouts: &[Layout]) -> PyResult<Layout> {
    let mut unions: Vec<&UnionScalarList> = Vec::with_capacity(layouts.len());
    for l in layouts {
        match l {
            Layout::UnionScalarList(u) => unions.push(u),
            _ => {
                return Err(internal("concat_axis0", "concat mixed layout kinds in union path"));
            }
        }
    }
    if unions.is_empty() {
        return Err(internal("concat_axis0", "cannot concat empty layouts"));
    }
    let mut all_tags: Vec<u8> = Vec::new();
    let mut all_index: Vec<i64> = Vec::new();
    let mut scalar_segs: Vec<Layout> = Vec::with_capacity(unions.len());
    let mut list_segs: Vec<Layout> = Vec::with_capacity(unions.len());
    let mut scalar_base = 0i64;
    let mut list_base = 0i64;
    for u in &unions {
        for i in 0..u.len() {
            all_tags.push(u.tags[i]);
            all_index.push(match u.tags[i] {
                0 => scalar_base + u.index[i],
                1 => list_base + u.index[i],
                _ => return Err(PyValueError::new_err("Invalid union tag.")),
            });
        }
        scalar_base += u.scalars.len as i64;
        list_base += u.lists.len() as i64;
        scalar_segs.push(Layout::Leaf(u.scalars.clone()));
        list_segs.push(Layout::ListOffset(u.lists.clone()));
    }
    let scalars = match concat_axis0_layouts(&scalar_segs)? {
        Layout::Leaf(l) => l,
        _ => {
            return Err(internal(
                "concat_axis0",
                "union scalar concat did not produce a leaf",
            ))
        }
    };
    let lists = match concat_axis0_layouts(&list_segs)? {
        Layout::ListOffset(lo) => lo,
        _ => {
            return Err(internal(
                "concat_axis0",
                "union list concat did not produce ListOffset",
            ))
        }
    };
    Ok(Layout::UnionScalarList(UnionScalarList {
        tags: all_tags,
        index: all_index,
        scalars,
        lists,
    }))
}

/// Concatenate layouts along axis 0 (union-aware).
pub fn concat_axis0_layouts(layouts: &[Layout]) -> PyResult<Layout> {
    if layouts.is_empty() {
        return Err(internal("concat_axis0", "cannot concat empty layouts"));
    }
    let has_union = layouts.iter().any(|l| matches!(l, Layout::UnionScalarList(_)));
    if has_union {
        let normalized: Vec<Layout> = layouts
            .iter()
            .map(|l| match l {
                Layout::UnionScalarList(_) => Ok(l.clone()),
                Layout::ListOffset(lo) => {
                    let dt = union_scalar_dtype_from_list(lo)?;
                    Ok(Layout::UnionScalarList(lift_listoffset_to_union(lo, dt)))
                }
                Layout::Leaf(leaf) => {
                    let lo = ListOffset {
                        offsets: Arc::new(vec![0i64, leaf.len as i64]),
                        content: Box::new(Layout::Leaf(leaf.clone())),
                    };
                    Ok(Layout::UnionScalarList(lift_listoffset_to_union(&lo, leaf.dtype)))
                }
                _ => Err(concat_incompatible(
                    "cannot mix union arrays with this layout kind on axis 0",
                    "use gr.array(...) to build list-chain or UnionScalarList layouts before concatenating.",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        return concat_union_scalar_lists_axis0(&normalized);
    }
    match &layouts[0] {
        Layout::Leaf(first) => {
            let dt = first.dtype;
            let total: usize = layouts.iter().map(|l| l.len()).sum();
            let mut out = Leaf::new(dt);
            out.len = total;
            out.validity = Arc::new(bitvec![u8, Lsb0; 1; total]);
            out.has_nulls = false;
            out.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(Arc::new(Vec::with_capacity(total))),
                DType::Int16 => LeafBuffer::I16(Arc::new(Vec::with_capacity(total))),
                DType::Int32 => LeafBuffer::I32(Arc::new(Vec::with_capacity(total))),
                DType::Int64 => LeafBuffer::I64(Arc::new(Vec::with_capacity(total))),
                DType::UInt8 => LeafBuffer::U8(Arc::new(Vec::with_capacity(total))),
                DType::UInt16 => LeafBuffer::U16(Arc::new(Vec::with_capacity(total))),
                DType::UInt32 => LeafBuffer::U32(Arc::new(Vec::with_capacity(total))),
                DType::UInt64 => LeafBuffer::U64(Arc::new(Vec::with_capacity(total))),
                DType::Float16 => LeafBuffer::F16(Arc::new(Vec::with_capacity(total))),
                DType::Float32 => LeafBuffer::F32(Arc::new(Vec::with_capacity(total))),
                DType::Float64 => LeafBuffer::F64(Arc::new(Vec::with_capacity(total))),
                DType::Bool => LeafBuffer::Bool(Arc::new(Vec::with_capacity(total))),
                DType::Char => LeafBuffer::Char(Arc::new(Vec::with_capacity(total))),
                DType::String => LeafBuffer::String(Arc::new(Vec::with_capacity(total))),
            };
            let mut pos = 0usize;
            for l in layouts {
                let leaf = match l {
                    Layout::Leaf(x) => x,
                    _ => return Err(internal("concat_axis0", "concat mixed layout kinds")),
                };
                if leaf.dtype != dt {
                    return Err(dtype_mismatch(dt, leaf.dtype, "during axis-0 leaf concat"));
                }
                if leaf.has_nulls {
                    out.has_nulls = true;
                }
                for i in 0..leaf.len {
                    if !leaf.validity[i] {
                        Arc::make_mut(&mut out.validity).set(pos + i, false);
                    }
                }
                match (&leaf.buffer, &mut out.buffer) {
                    (LeafBuffer::I8(v), LeafBuffer::I8(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::I16(v), LeafBuffer::I16(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::I32(v), LeafBuffer::I32(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::I64(v), LeafBuffer::I64(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U8(v), LeafBuffer::U8(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U16(v), LeafBuffer::U16(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U32(v), LeafBuffer::U32(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::U64(v), LeafBuffer::U64(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::F16(v), LeafBuffer::F16(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::F32(v), LeafBuffer::F32(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::F64(v), LeafBuffer::F64(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::Bool(v), LeafBuffer::Bool(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::Char(v), LeafBuffer::Char(o)) => Arc::make_mut(o).extend_from_slice(&v[..]),
                    (LeafBuffer::String(v), LeafBuffer::String(o)) => Arc::make_mut(o).extend(v.iter().cloned()),
                    _ => return Err(internal_dtype_buffer_mismatch("concat_axis0", dt)),
                }
                pos += leaf.len;
            }
            Ok(Layout::Leaf(out))
        }
        Layout::ListOffset(_) => {
            let mut all_offsets: Vec<i64> = Vec::new();
            all_offsets.push(0);
            let mut content_segs: Vec<Layout> = Vec::with_capacity(layouts.len());
            let mut acc: i64 = 0;
            for l in layouts {
                let lo = match l {
                    Layout::ListOffset(lo) => lo,
                    _ => return Err(internal("concat_axis0", "concat mixed layout kinds")),
                };
                let offs = lo.offsets.as_slice();
                if offs.is_empty() {
                    return Err(internal("concat_axis0", "invalid empty offsets"));
                }
                for &o in &offs[1..] {
                    all_offsets.push(acc + o);
                }
                acc += *offs.last().unwrap();
                content_segs.push(lo.content.as_ref().clone());
            }
            let content = concat_axis0_layouts(&content_segs)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: Arc::new(all_offsets),
                content: Box::new(content),
            }))
        }
        _ => Err(layout_unsupported("concat_axis0", "unsupported layout kind")),
    }
}

/// Concatenate arrays along the given axis (layout-native, no Python round-trip).
pub fn concat_arrays(arrays: &[GrumpyArray], dim: usize) -> PyResult<GrumpyArray> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("cat() requires at least one array."));
    }
    let dtype = arrays[0].dtype;
    for a in &arrays[1..] {
        if a.dtype != dtype {
            return Err(dtype_mismatch(dtype, a.dtype, "in cat()"));
        }
    }
    if dim == 0 {
        let layouts: Vec<Layout> = arrays.iter().map(|a| a.layout.clone()).collect();
        let layout = concat_axis0_layouts(&layouts)?;
        return Ok(GrumpyArray { dtype, layout });
    }
    let n = arrays[0].len();
    for a in &arrays[1..] {
        if a.len() != n {
            return Err(shape_mismatch(
                "cat",
                format!("dim={dim} requires same length along outer axes; got {n} and {}", a.len()),
                "ensure all inputs share the same outer shape before concatenating on this axis.",
            ));
        }
    }
    let mut segs: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let sub_arrays: Vec<GrumpyArray> = arrays
            .iter()
            .map(|a| {
                Ok(GrumpyArray {
                    dtype: a.dtype,
                    layout: drop_axis0_select_element(&a.layout, i)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let sub = concat_arrays(&sub_arrays, dim - 1)?;
        segs.push(sub.layout);
    }
    let layout = concat_axis0_layouts(&segs)?;
    Ok(GrumpyArray { dtype, layout })
}
