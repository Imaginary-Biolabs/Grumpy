//! Schema-level indexing: single-level subset only (int, slice, list, bool mask).

use crate::dataframe::{GrumpyDataFrame, Schema};
use crate::error::{arg_invalid, index_out_of_bounds, schema_violation};
use crate::layout::{drop_axis0_select_element, Layout, ListOffset};
use crate::layout::{Indexed, OffsetView};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyEllipsis, PySequence, PySlice, PyTuple};
use std::sync::Arc;

/// Selection at one schema nesting level.
#[derive(Clone, Debug)]
pub enum LevelSelector {
    One(i64),
    Slice { start: i64, stop: i64, step: i64 },
    Fancy(Vec<i64>),
    BoolMask(Vec<bool>),
}

fn normalize_index(raw: i64, len: i64) -> PyResult<usize> {
    let mut i = raw;
    if i < 0 {
        i += len;
    }
    if i < 0 || i >= len {
        return Err(index_out_of_bounds(i as usize, len as usize, "on schema index"));
    }
    Ok(i as usize)
}

fn is_ellipsis(py: Python<'_>, obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyEllipsis>()
}

fn is_full_slice(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(slc) = obj.downcast::<PySlice>() {
        let start = slc.getattr(intern!(slc.py(), "start"))?;
        let stop = slc.getattr(intern!(slc.py(), "stop"))?;
        let step = slc.getattr(intern!(slc.py(), "step"))?;
        return Ok(start.is_none() && stop.is_none() && step.is_none());
    }
    Ok(false)
}

fn parse_slice_indices(py: Python<'_>, slc: &Bound<'_, PySlice>, n: i64) -> PyResult<Vec<i64>> {
    let indices = slc.call_method1("indices", (n,))?;
    let t = indices.downcast::<PyTuple>()?;
    let start = t.get_item(0)?.extract::<i64>()?;
    let stop = t.get_item(1)?.extract::<i64>()?;
    let step = t.get_item(2)?.extract::<i64>()?;
    if step == 0 {
        return Err(arg_invalid(
            "slice step",
            "step cannot be zero",
            "use a non-zero step for schema slicing.",
        ));
    }
    let mut out = Vec::new();
    let mut i = start;
    if step > 0 {
        while i < stop {
            out.push(i);
            i += step;
        }
    } else {
        while i > stop {
            out.push(i);
            i += step;
        }
    }
    Ok(out)
}

fn parse_int_list(py: Python<'_>, seq: &Bound<'_, PySequence>) -> PyResult<Vec<i64>> {
    let mut out = Vec::with_capacity(seq.len()? as usize);
    for i in 0..seq.len()? {
        let it = seq.get_item(i)?;
        out.push(it.extract::<i64>().map_err(|_| {
            arg_invalid(
                "index",
                "fancy schema index must contain integers",
                "pass a list of int indices at this schema level.",
            )
        })?);
    }
    Ok(out)
}

fn parse_bool_mask(py: Python<'_>, seq: &Bound<'_, PySequence>, n: usize) -> PyResult<Vec<bool>> {
    let m = seq.len()? as usize;
    if m != n {
        return Err(schema_violation(
            "boolean schema index requires mask length to match entity count at this level",
            format!("mask has length {m} but level has {n} entities."),
            "pass a boolean mask with one entry per entity at the active schema level.",
        ));
    }
    let mut mask = Vec::with_capacity(m);
    for i in 0..m {
        let it = seq.get_item(i)?;
        mask.push(it.extract::<bool>().map_err(|_| {
            arg_invalid(
                "index",
                "boolean mask must contain only bool values",
                "pass True/False for each entity at this schema level.",
            )
        })?);
    }
    Ok(mask)
}

fn is_nested_sequence(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !crate::dtype::is_sequence_like(py, obj)? {
        return Ok(false);
    }
    let seq = obj.downcast::<PySequence>()?;
    if seq.len()? == 0 {
        return Ok(false);
    }
    let first = seq.get_item(0)?;
    Ok(crate::dtype::is_sequence_like(py, &first)?)
}

/// Parse one schema-level selector (single level only).
pub fn parse_level_selector(py: Python<'_>, obj: &Bound<'_, PyAny>, n: usize) -> PyResult<LevelSelector> {
    if is_ellipsis(py, obj) {
        return Err(arg_invalid(
            "index",
            "ellipsis is not supported in schema indexing",
            "pass an explicit int, slice, list, or boolean mask at one schema level.",
        ));
    }
    if is_full_slice(obj)? || obj.is_none() {
        return Err(arg_invalid(
            "index",
            "full slice or omitted level is not supported in schema indexing",
            "pass an explicit int, slice, list, or boolean mask at one schema level.",
        ));
    }
    if let Ok(slc) = obj.downcast::<PySlice>() {
        let fancy = parse_slice_indices(py, slc, n as i64)?;
        if fancy.len() == n {
            return Err(arg_invalid(
                "index",
                "full-range slice is not supported in schema indexing",
                "pass a strict subset slice at this schema level.",
            ));
        }
        return Ok(LevelSelector::Fancy(fancy));
    }
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(LevelSelector::One(obj.extract()?));
    }
    if crate::dtype::is_sequence_like(py, obj)? {
        let seq = obj.downcast::<PySequence>()?;
        if seq.len()? == 0 {
            return Err(arg_invalid(
                "index",
                "empty index sequence",
                "pass at least one index at this schema level.",
            ));
        }
        let first = seq.get_item(0)?;
        if first.is_instance_of::<pyo3::types::PyBool>() {
            return Ok(LevelSelector::BoolMask(parse_bool_mask(py, seq, n)?));
        }
        if is_nested_sequence(py, obj)? {
            return Err(arg_invalid(
                "index",
                "nested list indexing is not supported",
                "index one schema level at a time (e.g. df.scene[i].molecule[j]); for pairs, index twice.",
            ));
        }
        return Ok(LevelSelector::Fancy(parse_int_list(py, seq)?));
    }
    Err(arg_invalid(
        "index",
        "schema index must be int, slice, list, or boolean mask",
        "use df.level[i], df.level[[i,j]], or df.level[i:j] at one schema level.",
    ))
}

fn select_one(layout: &Layout, idx: i64) -> PyResult<Layout> {
    let i = normalize_index(idx, layout.len() as i64)?;
    drop_axis0_select_element(layout, i)
}

fn should_wrap_deep_leaf_segment(
    col_level: Option<usize>,
    index_depth: usize,
    max_schema_level: Option<usize>,
    col_layout: &Layout,
    elem: &Layout,
) -> bool {
    let Some(lvl) = col_level else {
        return false;
    };
    let Some(max_lvl) = max_schema_level else {
        return false;
    };
    matches!(elem, Layout::Leaf(l) if l.len > 1)
        && lvl == max_lvl
        && lvl > index_depth
        && matches!(col_layout, Layout::ListOffset(_))
}

/// Keep one outer dataframe row when a point selection yields a multi-value leaf segment.
fn wrap_multi_leaf_row(layout: Layout) -> Layout {
    if let Layout::Leaf(l) = &layout {
        if l.len > 1 {
            return Layout::ListOffset(ListOffset {
                offsets: Arc::new(vec![0, l.len as i64]),
                content: Box::new(layout),
            });
        }
    }
    layout
}

fn is_scalar_terminal(layout: &Layout) -> bool {
    matches!(layout, Layout::Leaf(l) if l.len <= 1)
}

/// Axis-0 subset via ``Indexed`` / ``OffsetView`` (preserves nesting without materializing stacks).
fn select_axis0_indexed(layout: &Layout, indices: &[i64]) -> PyResult<Layout> {
    let n = layout.len() as i64;
    let mut sub: Vec<i64> = Vec::new();
    for &raw in indices {
        let mut j = raw;
        if j < 0 {
            j += n;
        }
        if j >= 0 && j < n {
            sub.push(j);
        }
    }
    if sub.is_empty() {
        return Err(index_out_of_bounds(0, layout.len(), "on schema index"));
    }
    if sub.len() > 1
        && sub.windows(2).all(|w| w[1] == w[0] + 1)
        && matches!(layout, Layout::ListOffset(_))
    {
        let start = sub[0] as usize;
        let stop = (sub[sub.len() - 1] + 1) as usize;
        let lo = match layout {
            Layout::ListOffset(lo) => lo,
            _ => unreachable!(),
        };
        return Ok(Layout::OffsetView(OffsetView {
            offsets: lo.offsets.clone(),
            start,
            stop,
            content: lo.content.clone(),
        }));
    }
    Ok(Layout::Indexed(Indexed {
        index: Arc::new(sub),
        content: Box::new(layout.clone()),
    }))
}

fn selector_indices(sel: &LevelSelector) -> PyResult<Vec<i64>> {
    match sel {
        LevelSelector::Fancy(idxs) => Ok(idxs.clone()),
        LevelSelector::BoolMask(mask) => {
            let mut idxs = Vec::new();
            for (i, &b) in mask.iter().enumerate() {
                if b {
                    idxs.push(i as i64);
                }
            }
            Ok(idxs)
        }
        LevelSelector::One(i) => Ok(vec![*i]),
        LevelSelector::Slice { start, stop, step } => {
            let mut idxs = Vec::new();
            let mut i = *start;
            if *step > 0 {
                while i < *stop {
                    idxs.push(i);
                    i += *step;
                }
            } else {
                while i > *stop {
                    idxs.push(i);
                    i += *step;
                }
            }
            Ok(idxs)
        }
    }
}

/// Apply a single axis-0 selector on the current schema level, preserving inner nesting.
pub fn select_axis0_relative(df: &GrumpyDataFrame, sel: LevelSelector) -> PyResult<GrumpyDataFrame> {
    let max_schema_level = df
        .schema
        .as_ref()
        .map(|s| s.levels.len().saturating_sub(1));
    let mut out_cols = Vec::with_capacity(df.cols.len());
    for (name, col) in df.names.iter().zip(df.cols.iter()) {
        let col_level = df
            .schema
            .as_ref()
            .and_then(|s| s.level_for_column(name).ok());
        let layout = match &sel {
            LevelSelector::One(i) => {
                if col.layout.len() == 1 {
                    if is_scalar_terminal(&col.layout) {
                        col.layout.clone()
                    } else {
                        normalize_index(*i, 1)?;
                        col.layout.clone()
                    }
                } else {
                    let elem = select_one(&col.layout, *i)?;
                    if should_wrap_deep_leaf_segment(
                        col_level,
                        df.index_depth,
                        max_schema_level,
                        &col.layout,
                        &elem,
                    ) {
                        wrap_multi_leaf_row(elem)
                    } else {
                        elem
                    }
                }
            }
            LevelSelector::Fancy(_)
            | LevelSelector::BoolMask(_)
            | LevelSelector::Slice { .. } => {
                select_axis0_indexed(&col.layout, &selector_indices(&sel)?)?
            }
        };
        out_cols.push(crate::layout::GrumpyArray {
            dtype: col.dtype,
            layout,
        });
    }
    GrumpyDataFrame::from_schema_index_step(
        df.names.clone(),
        out_cols,
        df.schema.clone(),
        df.index_depth + 1,
    )
}

fn effective_outer_len(layout: &Layout) -> usize {
    let mut cur = layout;
    for _ in 0..32 {
        let n = cur.len();
        if n != 1 {
            return n;
        }
        cur = match cur {
            Layout::ListOffset(lo) => lo.content.as_ref(),
            Layout::Indexed(ix) if ix.index.len() == 1 => ix.content.as_ref(),
            Layout::OffsetView(v) if v.len() == 1 => v.content.as_ref(),
            _ => return 1,
        };
    }
    1
}

pub fn entity_count_at_current_level(df: &GrumpyDataFrame) -> usize {
    if let Some(schema) = &df.schema {
        let level = df.index_depth;
        let mut fallback = 0usize;
        for (name, col) in df.names.iter().zip(df.cols.iter()) {
            if let Ok(lvl) = schema.level_for_column(name) {
                let n = effective_outer_len(&col.layout);
                if lvl == level {
                    return n;
                }
                if lvl > level {
                    fallback = fallback.max(n);
                }
            }
        }
        if fallback > 0 {
            return fallback;
        }
    }
    df.nrows()
}

pub fn parse_dataframe_index_key(
    py: Python<'_>,
    key: &Bound<'_, PyAny>,
    df: &GrumpyDataFrame,
) -> PyResult<LevelSelector> {
    if let Ok(tup) = key.downcast::<PyTuple>() {
        if tup.is_empty() {
            return Err(arg_invalid("index", "empty index tuple", "pass a single index."));
        }
        let first = tup.get_item(0)?;
        if first.extract::<String>().is_ok() {
            return Err(arg_invalid(
                "key",
                "column selection must be strings",
                "pass column names as str or tuple[str, ...].",
            ));
        }
        if tup.len() > 1 {
            return Err(arg_invalid(
                "index",
                "multi-level tuple indexing is not supported",
                "index one schema level at a time via df.scene[i].molecule[j]; for pairs, call indexing twice on separate dataframes.",
            ));
        }
        return parse_level_selector(py, &first, entity_count_at_current_level(df));
    }
    parse_level_selector(py, key, entity_count_at_current_level(df))
}

pub fn accessor_target_level(schema: &Schema, path: &[String]) -> PyResult<usize> {
    if path.is_empty() {
        return Err(arg_invalid(
            "accessor",
            "empty schema accessor path",
            "use df.<level>[index] drill-down.",
        ));
    }
    let name = path.last().expect("non-empty path");
    schema.name_to_level.get(name).copied().ok_or_else(|| {
        schema_violation(
            format!("'{}' is not a declared schema level", name),
            "accessor path must use declared schema level names.",
            "use names from schema= in drill-down accessors.",
        )
    })
}

/// Drill-down ``accessor[level_key]`` — single-level subset on the parent dataframe.
pub fn accessor_getitem(
    py: Python<'_>,
    df: &GrumpyDataFrame,
    level: usize,
    key: &Bound<'_, PyAny>,
) -> PyResult<GrumpyDataFrame> {
    if df.index_depth != level {
        return Err(schema_violation(
            format!(
                "accessor level {level} does not match dataframe index_depth {}",
                df.index_depth
            ),
            "narrow outer schema levels before indexing deeper levels.",
            "use df.scene[i].molecule[j] instead of skipping levels on the root dataframe.",
        ));
    }
    let sel = parse_level_selector(py, key, entity_count_at_current_level(df))?;
    select_axis0_relative(df, sel)
}

pub fn dataframe_getitem(df: &GrumpyDataFrame, sel: LevelSelector) -> PyResult<GrumpyDataFrame> {
    if df.schema.is_some() {
        return select_axis0_relative(df, sel);
    }
    df.row_select_indexed(selector_to_indices(&sel)?)
}

fn selector_to_indices(sel: &LevelSelector) -> PyResult<Arc<Vec<i64>>> {
    Ok(Arc::new(selector_indices(sel)?))
}
