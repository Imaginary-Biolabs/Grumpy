//! Shared recursive layout transforms (union-aware).

use crate::layout::{Layout, Leaf, ListOffset, UnionScalarList};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn map_layout_unary<F>(layout: &Layout, f: &mut F) -> PyResult<Layout>
where
    F: FnMut(&Leaf) -> PyResult<Leaf>,
{
    match layout {
        Layout::Leaf(l) => Ok(Layout::Leaf(f(l)?)),
        Layout::ListOffset(lo) => {
            let content = map_layout_unary(lo.content.as_ref(), f)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            let content = map_layout_unary(v.content.as_ref(), f)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: v.offsets.clone(),
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            }))
        }
        Layout::Indexed(ix) => {
            let content = map_layout_unary(ix.content.as_ref(), f)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(u) => {
            let scalars = f(&u.scalars)?;
            let list_content = map_layout_unary(u.lists.content.as_ref(), f)?;
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

pub fn map_layout_binary<F>(a: &Layout, b: &Layout, f: &mut F) -> PyResult<Layout>
where
    F: FnMut(&Leaf, &Leaf) -> PyResult<Leaf>,
{
    match (a, b) {
        (Layout::Leaf(la), Layout::Leaf(lb)) => Ok(Layout::Leaf(f(la, lb)?)),
        (Layout::ListOffset(oa), Layout::ListOffset(ob)) => {
            if oa.offsets != ob.offsets {
                return Err(PyValueError::new_err(
                    "Binary layout op requires identical offsets.",
                ));
            }
            let content = map_layout_binary(oa.content.as_ref(), ob.content.as_ref(), f)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: oa.offsets.clone(),
                content: Box::new(content),
            }))
        }
        (Layout::OffsetView(va), Layout::OffsetView(vb)) => {
            if va.start != vb.start || va.stop != vb.stop || va.offsets != vb.offsets {
                return Err(PyValueError::new_err(
                    "Binary layout op requires identical offset views.",
                ));
            }
            let content = map_layout_binary(va.content.as_ref(), vb.content.as_ref(), f)?;
            Ok(Layout::OffsetView(crate::layout::OffsetView {
                offsets: va.offsets.clone(),
                start: va.start,
                stop: vb.stop,
                content: Box::new(content),
            }))
        }
        (Layout::Indexed(ia), Layout::Indexed(ib)) => {
            if ia.index != ib.index {
                return Err(PyValueError::new_err(
                    "Binary layout op requires identical index vectors.",
                ));
            }
            let content = map_layout_binary(ia.content.as_ref(), ib.content.as_ref(), f)?;
            Ok(Layout::Indexed(crate::layout::Indexed {
                index: ia.index.clone(),
                content: Box::new(content),
            }))
        }
        (Layout::UnionScalarList(ua), Layout::UnionScalarList(ub)) => {
            if ua.tags != ub.tags || ua.index != ub.index || ua.lists.offsets != ub.lists.offsets {
                return Err(PyValueError::new_err(
                    "Binary layout op requires identical union structure.",
                ));
            }
            let scalars = f(&ua.scalars, &ub.scalars)?;
            let list_content =
                map_layout_binary(ua.lists.content.as_ref(), ub.lists.content.as_ref(), f)?;
            Ok(Layout::UnionScalarList(UnionScalarList {
                tags: ua.tags.clone(),
                index: ua.index.clone(),
                scalars,
                lists: ListOffset {
                    offsets: ua.lists.offsets.clone(),
                    content: Box::new(list_content),
                },
            }))
        }
        _ => Err(PyValueError::new_err(
            "Binary layout op requires matching layout kinds.",
        )),
    }
}

pub fn find_leaf_any(layout: &Layout) -> PyResult<&Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::ListOffset(lo) => find_leaf_any(lo.content.as_ref()),
        Layout::OffsetView(v) => find_leaf_any(v.content.as_ref()),
        Layout::Indexed(ix) => find_leaf_any(ix.content.as_ref()),
        Layout::UnionScalarList(u) => {
            if u.scalars.len > 0 {
                Ok(&u.scalars)
            } else {
                find_leaf_any(u.lists.content.as_ref())
            }
        }
    }
}
