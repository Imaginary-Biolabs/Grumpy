use crate::dtype::DType;
use crate::layout::{
    build_array, coord_to_leaf_index, drop_axis0_select_element, gather_2d_fancy_leaf,
    gather_axis0_fancy, gather_coordinate_fancy_2d, index_by_coordinates, scatter_2d_fancy_i32,
    scatter_2d_fancy_numeric, take_range, GrumpyArray, Layout, LeafBuffer,
};
use crate::py_api::convert::{wrap_index_layout_result, wrap_result};
use crate::py_api::types::PyGrumpyArray;
use numpy::PyReadonlyArray1;
use crate::error::{arg_invalid, index_out_of_bounds, index_out_of_bounds_simple, internal, layout_unsupported, shape_mismatch, unsupported};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyInt, PySlice, PyTuple};
use std::sync::Arc;

pub(crate) fn fast_getitem(py: Python<'_>, arr: &GrumpyArray, index: &Bound<'_, PyAny>) -> PyResult<Option<PyObject>> {
    let is_list_chain = arr.is_pure_list_chain();

    // Coordinate tuple indexing
    if let Ok(tup) = index.downcast::<PyTuple>() {
        // Hot path: x[int, int] on 2D int32 arrays (skip fancy-index probes).
        if tup.len() == 2 {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.is_instance_of::<PyInt>() && a1.is_instance_of::<PyInt>() {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    if !is_list_chain {
                        let result = index_by_coordinates(&arr.layout, &[r, c])?;
                        return Ok(Some(wrap_index_layout_result(py, arr.dtype, result)?));
                    }
                    if arr.dtype == DType::Int32 {
                        if let Layout::ListOffset(lo) = &arr.layout {
                            if let Layout::Leaf(leaf) = lo.content.as_ref() {
                                if !leaf.has_nulls {
                                    if let LeafBuffer::I32(buf) = &leaf.buffer {
                                        let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                                        return Ok(Some((buf[leaf_ix] as i64).into_py(py)));
                                    }
                                }
                            }
                        }
                    }
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_fast(&arr.layout)?;
                    let out = leaf.scalar_to_py(py, leaf_ix)?;
                    return Ok(Some(out));
                }
            }
        }

        // Fancy coordinate indexing if any part is sequence-like (and no slices).
        let mut has_seq = false;
        let mut has_slice = false;
        for i in 0..tup.len() {
            let p = tup.get_item(i)?;
            if p.downcast::<PySlice>().is_ok() {
                has_slice = true;
            } else if is_index_vec_like(py, &p)? {
                has_seq = true;
            }
        }
        if has_slice {
            return Ok(None);
        }
        if tup.len() == 2 && !has_seq {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.downcast::<PySlice>().is_err() && a1.downcast::<PySlice>().is_err() {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    if !is_list_chain {
                        let result = index_by_coordinates(&arr.layout, &[r, c])?;
                        return Ok(Some(wrap_index_layout_result(py, arr.dtype, result)?));
                    }
                    if arr.dtype == DType::Int32 {
                        if let Layout::ListOffset(lo) = &arr.layout {
                            if let Layout::Leaf(leaf) = lo.content.as_ref() {
                                if !leaf.has_nulls {
                                    if let LeafBuffer::I32(buf) = &leaf.buffer {
                                        let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                                        return Ok(Some((buf[leaf_ix] as i64).into_py(py)));
                                    }
                                }
                            }
                        }
                    }
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_fast(&arr.layout)?;
                    let out = leaf.scalar_to_py(py, leaf_ix)?;
                    return Ok(Some(out));
                }
            }
        }
        if has_seq {
            // Support 2D fancy: (rows, cols) with optional scalar broadcast.
            if tup.len() != 2 {
                return Ok(None);
            }
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            let rows_d = extract_index_data(py, &a0)?;
            let cols_d = extract_index_data(py, &a1)?;
            let rows: &[i64] = match &rows_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };
            let cols: &[i64] = match &cols_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };

            if !is_list_chain {
                if !rows.is_empty() && !cols.is_empty() {
                    let layout = gather_coordinate_fancy_2d(&arr.layout, rows, cols)?;
                    return Ok(Some(wrap_index_layout_result(py, arr.dtype, layout)?));
                }
                if !rows.is_empty() && cols.is_empty() {
                    let col = a1.extract::<i64>()?;
                    let cols2: Vec<i64> = vec![col; rows.len()];
                    let layout = gather_coordinate_fancy_2d(&arr.layout, rows, &cols2)?;
                    return Ok(Some(wrap_index_layout_result(py, arr.dtype, layout)?));
                }
                if rows.is_empty() && !cols.is_empty() {
                    let row = a0.extract::<i64>()?;
                    let rows2: Vec<i64> = vec![row; cols.len()];
                    let layout = gather_coordinate_fancy_2d(&arr.layout, &rows2, cols)?;
                    return Ok(Some(wrap_index_layout_result(py, arr.dtype, layout)?));
                }
                return Ok(None);
            }

            if !rows.is_empty() && !cols.is_empty() {
                let leaf = gather_2d_fancy_leaf(&arr.layout, &rows, &cols)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            if !rows.is_empty() && cols.is_empty() {
                let col = a1.extract::<i64>()?;
                let cols2 = vec![col; rows.len()];
                let leaf = gather_2d_fancy_leaf(&arr.layout, &rows, &cols2)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            if rows.is_empty() && !cols.is_empty() {
                let row = a0.extract::<i64>()?;
                let rows2 = vec![row; cols.len()];
                let leaf = gather_2d_fancy_leaf(&arr.layout, &rows2, &cols)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            return Ok(None);
        }

        // Pure coordinate (ints only)
        if tup.len() == 1 {
            let i0 = tup.get_item(0)?.extract::<i64>()?;
            return fast_getitem(py, arr, &i0.into_py(py).into_bound(py));
        }
        // Scalar coordinate: return scalar
        let mut coords: Vec<i64> = Vec::with_capacity(tup.len());
        for i in 0..tup.len() {
            coords.push(tup.get_item(i)?.extract::<i64>()?);
        }
        if !is_list_chain {
            let result = index_by_coordinates(&arr.layout, &coords)?;
            return Ok(Some(wrap_index_layout_result(py, arr.dtype, result)?));
        }
        // Only support scalar selection ending in leaf scalar.
        let leaf_ix = coord_to_leaf_index(&arr.layout, &coords)?;
        let leaf = find_leaf_fast(&arr.layout)?;
        let out = leaf.scalar_to_py(py, leaf_ix)?;
        return Ok(Some(out));
    }

    // Axis-0 int selection (drops axis)
    if let Ok(i0) = index.extract::<i64>() {
        let root_len = arr.len() as i64;
        let mut i = i0;
        if i < 0 {
            i += root_len;
        }
        if i < 0 || i >= root_len {
            return Err(index_out_of_bounds(i as usize, root_len as usize, "on axis 0"));
        }
        let layout = drop_axis0_select_element(&arr.layout, i as usize)?;
        let out = PyGrumpyArray {
            inner: GrumpyArray {
                dtype: arr.dtype,
                layout,
            },
        };
        return Ok(Some(out.into_py(py)));
    }

    // Axis-0 slice: zero-copy view for list-chains; compact take for unions.
    if let Ok(slc) = index.downcast::<PySlice>() {
        match &arr.layout {
            Layout::UnionScalarList(u) => {
                let n: isize = u
                    .len()
                    .try_into()
                    .map_err(|_| arg_invalid("slice", "slice length too large", "use a smaller slice range."))?;
                let indices = slc.indices(n)?;
                let (start, stop, step) = (indices.start, indices.stop, indices.step);
                let layout = if step == 1 {
                    let start_u = start as usize;
                    let stop_u = stop as usize;
                    if start_u > stop_u || stop_u > u.len() {
                        return Err(index_out_of_bounds_simple("on slice axis"));
                    }
                    take_range(&arr.layout, start_u, stop_u)?
                } else {
                    let mut idxs: Vec<i64> = Vec::new();
                    let mut i = start;
                    while (step > 0 && i < stop) || (step < 0 && i > stop) {
                        idxs.push(i as i64);
                        i += step;
                    }
                    gather_axis0_fancy(&arr.layout, &idxs)?
                };
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout,
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            Layout::ListOffset(root_lo) if is_list_chain => {
                // Only step=1 for now (fast and common). Other steps fall back.
                let n: isize = root_lo
                    .len()
                    .try_into()
                    .map_err(|_| arg_invalid("slice", "slice length too large", "use a smaller slice range."))?;
                let indices = slc.indices(n)?;
                let (start, stop, step) = (indices.start, indices.stop, indices.step);
                if step != 1 {
                    return Ok(None);
                }
                let start_u = start as usize;
                let stop_u = stop as usize;
                if start_u > stop_u || stop_u > root_lo.len() {
                    return Err(index_out_of_bounds_simple("on slice axis"));
                }
                let layout = Layout::OffsetView(crate::layout::OffsetView {
                    offsets: root_lo.offsets.clone(),
                    start: start_u,
                    stop: stop_u,
                    content: root_lo.content.clone(),
                });
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout,
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
            _ => return Ok(None),
        }
    }

    // Axis-0 fancy / boolean selection (union arrays; list-chains use per-row rules via fallback).
    if !is_list_chain && is_index_vec_like(py, index)? {
        if let Ok(seq) = index.downcast::<pyo3::types::PySequence>() {
            let m = seq.len()? as usize;
            let n = arr.len();
            let mut bools: Vec<bool> = Vec::new();
            let mut all_bool = m > 0;
            for i in 0..m {
                let it = seq.get_item(i)?;
                if it.is_instance_of::<pyo3::types::PyBool>() {
                    bools.push(it.extract::<bool>()?);
                } else {
                    all_bool = false;
                    break;
                }
            }
            if all_bool && m == n {
                let picked: Vec<i64> = bools
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| **b)
                    .map(|(i, _)| i as i64)
                    .collect();
                let layout = gather_axis0_fancy(&arr.layout, &picked)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: arr.dtype,
                        layout,
                    },
                };
                return Ok(Some(out.into_py(py)));
            }
        }
        let rows_d = extract_index_data(py, index)?;
        let rows: Vec<i64> = match rows_d {
            IndexData::NpI64(ro) => ro.as_slice()?.to_vec(),
            IndexData::Owned(v) => v,
            IndexData::Empty => return Ok(None),
        };
        if !rows.is_empty() {
            let layout = gather_axis0_fancy(&arr.layout, &rows)?;
            let out = PyGrumpyArray {
                inner: GrumpyArray {
                    dtype: arr.dtype,
                    layout,
                },
            };
            return Ok(Some(out.into_py(py)));
        }
    }

    Ok(None)
}

pub(crate) fn find_leaf_fast<'a>(layout: &'a Layout) -> PyResult<&'a crate::layout::Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::ListOffset(lo) => find_leaf_fast(lo.content.as_ref()),
        Layout::Indexed(ix) => find_leaf_fast(ix.content.as_ref()),
        Layout::OffsetView(v) => find_leaf_fast(v.content.as_ref()),
        Layout::UnionScalarList(_) => Err(layout_unsupported("indexing", "UnionScalarList layouts are not supported yet")),
    }
}

pub(crate) fn fast_setitem(
    py: Python<'_>,
    arr: &mut GrumpyArray,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !arr.is_pure_list_chain() {
        return Ok(false);
    }

    // Only support leaf-mutation assignments (no structural changes).
    // Supported:
    // - x[i,j] = scalar
    // - x[[i...],[j...]] = list/scalar
    // - x[[i...], j] scalar broadcast
    if let Ok(tup) = index.downcast::<PyTuple>() {
        if tup.len() == 0 {
            return Ok(false);
        }

        // Hot path: x[int, int] = int on 2D int32 arrays (skip fancy-index probes).
        if tup.len() == 2 {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.is_instance_of::<PyInt>() && a1.is_instance_of::<PyInt>() {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    if arr.dtype == DType::Int32 {
                        if let Ok(v) = value.extract::<i32>() {
                            let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                            if let Layout::ListOffset(lo) = &mut arr.layout {
                                if let Layout::Leaf(leaf) = lo.content.as_mut() {
                                    if !leaf.has_nulls {
                                        if let LeafBuffer::I32(buf) = &mut leaf.buffer {
                                            Arc::make_mut(buf)[leaf_ix] = v;
                                            return Ok(true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_mut_fast(&mut arr.layout)?;
                    if arr.dtype == DType::Int32 {
                        if let Ok(v) = value.extract::<i32>() {
                            leaf.set_i32(leaf_ix, v)?;
                            return Ok(true);
                        }
                    }
                }
            }
        }

        let mut has_seq = false;
        let mut has_slice = false;
        for i in 0..tup.len() {
            let p = tup.get_item(i)?;
            if p.downcast::<PySlice>().is_ok() {
                has_slice = true;
            } else if is_index_vec_like(py, &p)? {
                has_seq = true;
            }
        }
        if has_slice {
            return Ok(false);
        }
        if has_seq {
            if tup.len() != 2 {
                return Ok(false);
            }
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            let rows_d = extract_index_data(py, &a0)?;
            let cols_d = extract_index_data(py, &a1)?;
            let rows: &[i64] = match &rows_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };
            let cols: &[i64] = match &cols_d {
                IndexData::NpI64(ro) => ro.as_slice()?,
                IndexData::Owned(v) => v.as_slice(),
                IndexData::Empty => &[],
            };
            let n = if !rows.is_empty() { rows.len() } else { cols.len() };
            if n == 0 {
                return Ok(false);
            }
            let rows2_owned;
            let cols2_owned;
            let rows2: &[i64] = if rows.is_empty() {
                rows2_owned = vec![a0.extract::<i64>()?; n];
                rows2_owned.as_slice()
            } else {
                rows
            };
            let cols2: &[i64] = if cols.is_empty() {
                cols2_owned = vec![a1.extract::<i64>()?; n];
                cols2_owned.as_slice()
            } else {
                cols
            };

            // If dtype=int32 and values is a NumPy i32 array, use fully typed scatter (no per-element Python extraction).
            if arr.dtype == DType::Int32 {
                if let Ok(vro) = value.extract::<PyReadonlyArray1<'_, i32>>() {
                    let vs = vro.as_slice()?;
                    if vs.len() != n {
                        return Err(shape_mismatch(
                            "fancy assignment",
                            "value length must match number of selected coordinates",
                            "pass a value array with one element per (row, col) pair.",
                        ));
                    }
                    scatter_2d_fancy_i32(&mut arr.layout, rows2, cols2, vs)?;
                    return Ok(true);
                }
            }

            // Otherwise fall back to generic vectorized scatter (still avoids rebuilding structures).
            scatter_2d_fancy_numeric(py, &mut arr.layout, rows2, cols2, value, arr.dtype)?;
            return Ok(true);
        }

        // Pure scalar coordinate assignment (2D+ supported if it ends in leaf)
        if tup.len() == 2 {
            let a0 = tup.get_item(0)?;
            let a1 = tup.get_item(1)?;
            if a0.downcast::<PySlice>().is_err()
                && a1.downcast::<PySlice>().is_err()
                && !is_index_vec_like(py, &a0)?
                && !is_index_vec_like(py, &a1)?
            {
                if let (Ok(r), Ok(c)) = (a0.extract::<i64>(), a1.extract::<i64>()) {
                    let leaf_ix = coord_to_leaf_index(&arr.layout, &[r, c])?;
                    let leaf = find_leaf_mut_fast(&mut arr.layout)?;
                    if arr.dtype == DType::Int32 {
                        if let Ok(v) = value.extract::<i32>() {
                            leaf.set_i32(leaf_ix, v)?;
                            return Ok(true);
                        }
                    }
                }
            }
        }
        let mut coords: Vec<i64> = Vec::with_capacity(tup.len());
        for i in 0..tup.len() {
            coords.push(tup.get_item(i)?.extract::<i64>()?);
        }
        let leaf_ix = coord_to_leaf_index(&arr.layout, &coords)?;
        let (valid, bytes) = crate::layout::Leaf::encode_scalar(py, value, arr.dtype)?;
        let leaf = find_leaf_mut_fast(&mut arr.layout)?;
        leaf.set_encoded(leaf_ix, valid, &bytes)?;
        return Ok(true);
    }

    Ok(false)
}

fn find_leaf_mut_fast<'a>(layout: &'a mut Layout) -> PyResult<&'a mut crate::layout::Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::ListOffset(lo) => find_leaf_mut_fast(lo.content.as_mut()),
        Layout::Indexed(ix) => find_leaf_mut_fast(ix.content.as_mut()),
        Layout::OffsetView(v) => find_leaf_mut_fast(v.content.as_mut()),
        Layout::UnionScalarList(_) => Err(layout_unsupported("indexing", "UnionScalarList layouts are not supported yet")),
    }
}

enum IndexData<'py> {
    Empty,
    NpI64(PyReadonlyArray1<'py, i64>),
    Owned(Vec<i64>),
}

fn extract_index_data<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<IndexData<'py>> {
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<'py, i64>>() {
        return Ok(IndexData::NpI64(ro));
    }
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<'py, i32>>() {
        let slice = ro.as_slice()?;
        return Ok(IndexData::Owned(slice.iter().map(|&x| x as i64).collect()));
    }
    if crate::dtype::is_sequence_like(py, obj)? {
        let s = obj.downcast::<pyo3::types::PySequence>()?;
        let mut out = Vec::with_capacity(s.len()? as usize);
        for i in 0..s.len()? {
            out.push(s.get_item(i as usize)?.extract::<i64>()?);
        }
        return Ok(IndexData::Owned(out));
    }
    Ok(IndexData::Empty)
}

fn is_index_vec_like(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.extract::<PyReadonlyArray1<'_, i64>>().is_ok() {
        return Ok(true);
    }
    if obj.extract::<PyReadonlyArray1<'_, i32>>().is_ok() {
        return Ok(true);
    }
    crate::dtype::is_sequence_like(py, obj)
}

pub(crate) fn getitem_coordinate(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    _dtype: DType,
) -> PyResult<PyObject> {
    // Coordinate indexing:
    // - int/slice: applied on axis 0
    // - tuple: multiple axes, with fancy coordinates if any axis is a sequence of ints.
    if let Ok(tup) = index.downcast::<PyTuple>() {
        if tup.len() == 0 {
            return Err(arg_invalid("index", "empty index tuple is not allowed", "pass at least one coordinate."));
        }
        let parts: Vec<Bound<'_, PyAny>> = (0..tup.len()).map(|i| tup.get_item(i).unwrap()).collect();
        return getitem_coordinate_tuple(py, base, &parts);
    }

    // Single-axis coordinate indexing
    getitem_axis(py, base, index)
}

enum CoordPart {
    Int(i64),
    Fancy(Vec<i64>),
}

fn getitem_coordinate_tuple(py: Python<'_>, base: &Bound<'_, PyAny>, parts: &[Bound<'_, PyAny>]) -> PyResult<PyObject> {
    // If there are no fancy parts (sequences), just apply axis-by-axis with int/slice.
    let mut has_fancy = false;
    for p in parts {
        if crate::dtype::is_sequence_like(py, p)? {
            has_fancy = true;
            break;
        }
    }
    if !has_fancy {
        let mut cur = base.clone().unbind();
        let mut cur_b = cur.bind(py);
        for p in parts {
            cur = getitem_axis(py, &cur_b, p)?;
            cur_b = cur.bind(py);
        }
        return Ok(cur);
    }

    // Fancy coordinate indexing: allow scalars and 1D int sequences; disallow slices.
    let mut parsed: Vec<CoordPart> = Vec::with_capacity(parts.len());
    let mut fancy_lens: Vec<usize> = Vec::new();
    for p in parts {
        if p.downcast::<PySlice>().is_ok() {
            return Err(unsupported("coordinate assignment", "slice combined with fancy arrays is not supported yet", "use integer or fancy indices consistently."));
        }
        if let Ok(i) = p.extract::<i64>() {
            parsed.push(CoordPart::Int(i));
        } else if crate::dtype::is_sequence_like(py, p)? {
            let seq = p.downcast::<pyo3::types::PySequence>()?;
            let mut v = Vec::with_capacity(seq.len()? as usize);
            for j in 0..seq.len()? {
                v.push(seq.get_item(j as usize)?.extract::<i64>()?);
            }
            fancy_lens.push(v.len());
            parsed.push(CoordPart::Fancy(v));
        } else {
            return Err(unsupported(
                "coordinate indexing",
                "unsupported index component",
                "use int, slice, or 1D int sequence per axis.",
            ));
        }
    }

    let n = *fancy_lens
        .iter()
        .max()
        .ok_or_else(|| internal("indexing", "unexpected internal state"))?;
    for l in &fancy_lens {
        if *l != n {
            return Err(shape_mismatch(
                "coordinate indexing",
                "multiple index arrays require the same length",
                "align fancy index lengths or use scalar broadcast indices.",
            ));
        }
    }

    let out = pyo3::types::PyList::empty_bound(py);
    for k in 0..n {
        let mut cur = base.clone().unbind();
        let mut cur_b = cur.bind(py);
        for p in &parsed {
            let ix = match p {
                CoordPart::Int(i) => *i,
                CoordPart::Fancy(v) => v[k],
            };
            cur = getitem_axis(py, &cur_b, &ix.into_py(py).into_bound(py))?;
            cur_b = cur.bind(py);
        }
        out.append(cur)?;
    }
    Ok(out.into())
}

fn getitem_axis(py: Python<'_>, base: &Bound<'_, PyAny>, index: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    // base must be sequence-like for int/slice selection
    let seq = base.downcast::<pyo3::types::PySequence>().map_err(|_| {
        arg_invalid("index", "attempted to index into a scalar value", "index only into array/list dimensions.")
    })?;
    let len = seq.len()? as i64;

    if let Ok(slc) = index.downcast::<PySlice>() {
        let (start, stop, step) = parse_slice(py, slc, len)?;
        let out = pyo3::types::PyList::empty_bound(py);
        let mut i = start;
        if step > 0 {
            while i < stop {
                out.append(seq.get_item(i as usize)?)?;
                i += step;
            }
        } else {
            while i > stop {
                out.append(seq.get_item(i as usize)?)?;
                i += step;
            }
        }
        return Ok(out.into());
    }

    let mut i = index.extract::<i64>().map_err(|_| arg_invalid("index", "must be int or slice", "use integer indices or slices for this axis."))?;
    if i < 0 {
        i += len;
    }
    if i < 0 || i >= len {
        return Err(index_out_of_bounds(i as usize, len as usize, "on this axis"));
    }
    Ok(seq.get_item(i as usize)?.into())
}

fn parse_slice(
    _py: Python<'_>,
    slc: &Bound<'_, PySlice>,
    len: i64,
) -> PyResult<(i64, i64, i64)> {
    let indices = slc.call_method1("indices", (len,))?;
    let t = indices.downcast::<PyTuple>()?;
    let start = t.get_item(0)?.extract::<i64>()?;
    let stop = t.get_item(1)?.extract::<i64>()?;
    let step = t.get_item(2)?.extract::<i64>()?;
    Ok((start, stop, step))
}

pub(crate) fn getitem_array_indexing(py: Python<'_>, base: &Bound<'_, PyAny>, index: &Bound<'_, PyAny>, _dtype: DType) -> PyResult<PyObject> {
    let base_seq = base.downcast::<pyo3::types::PySequence>()?;
    let n = base_seq.len()? as usize;
    let idx_seq = index.downcast::<pyo3::types::PySequence>()?;
    let m = idx_seq.len()? as usize;

    // Determine if this is a boolean mask on axis 0.
    let mut is_all_bool = true;
    let mut bools: Vec<bool> = Vec::new();
    for i in 0..m {
        let it = idx_seq.get_item(i)?;
        if it.is_instance_of::<pyo3::types::PyBool>() {
            bools.push(it.extract::<bool>()?);
        } else {
            is_all_bool = false;
            break;
        }
    }
    if is_all_bool {
        if m != n {
            return Err(shape_mismatch("boolean indexing", "mask length must match outer dimension", "pass a mask with one entry per row."));
        }
        let out = pyo3::types::PyList::empty_bound(py);
        for i in 0..n {
            if bools[i] {
                out.append(base_seq.get_item(i as usize)?)?;
            }
        }
        return Ok(out.into());
    }

    // If index length != outer length, treat as outer fancy selection on axis 0.
    if m != n {
        let out = pyo3::types::PyList::empty_bound(py);
        for i in 0..m {
            let it = idx_seq.get_item(i)?;
            let mut ix = it.extract::<i64>()?;
            if ix < 0 {
                ix += n as i64;
            }
            if ix < 0 || ix >= n as i64 {
                return Err(index_out_of_bounds(ix as usize, n, "on axis 0"));
            }
            out.append(base_seq.get_item(ix as usize)?)?;
        }
        return Ok(out.into());
    }

    // Per-row indexing: apply each index element to corresponding row.
    let out = pyo3::types::PyList::empty_bound(py);
    for i in 0..n {
        let row = base_seq.get_item(i as usize)?;
        let sub = idx_seq.get_item(i as usize)?;
        // For int -> wrap as single-element list (matches example [[1],[5]])
        if let Ok(slc) = sub.downcast::<PySlice>() {
            let got = getitem_axis(py, &row, &slc.clone().into_any())?;
            out.append(got)?;
        } else if sub.extract::<i64>().is_ok() {
            let v = getitem_axis(py, &row, &sub)?;
            let wrap = pyo3::types::PyList::empty_bound(py);
            wrap.append(v)?;
            out.append(wrap)?;
        } else if crate::dtype::is_sequence_like(py, &sub)? {
            // sequence of ints for this row
            let sseq = sub.downcast::<pyo3::types::PySequence>()?;
            let wrap = pyo3::types::PyList::empty_bound(py);
            for j in 0..sseq.len()? {
                let jx = sseq.get_item(j as usize)?;
                let v = getitem_axis(py, &row, &jx)?;
                wrap.append(v)?;
            }
            out.append(wrap)?;
        } else {
            return Err(unsupported("per-row indexing", "unsupported index element type", "use int, slice, or sequence of ints per row."));
        }
    }
    Ok(out.into())
}

pub(crate) fn setitem_coordinate(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(tup) = index.downcast::<PyTuple>() {
        let parts: Vec<Bound<'_, PyAny>> = (0..tup.len()).map(|i| tup.get_item(i).unwrap()).collect();
        return setitem_coordinate_tuple(py, base, &parts, value);
    }
    setitem_axis(py, base, index, value)
}

fn setitem_coordinate_tuple(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    parts: &[Bound<'_, PyAny>],
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if parts.is_empty() {
        return Err(arg_invalid("index", "empty index tuple is not allowed", "pass at least one coordinate."));
    }

    let mut has_fancy = false;
    for p in parts {
        if crate::dtype::is_sequence_like(py, p)? {
            has_fancy = true;
            break;
        }
    }

    if !has_fancy {
        // Walk down to the last axis, mutating at the end.
        let mut cur_obj = base.clone().unbind();
        for ax in 0..parts.len() - 1 {
            let p = &parts[ax];
            if p.downcast::<PySlice>().is_ok() {
                return Err(unsupported("coordinate assignment", "slice in coordinate tuple is not supported yet", "use integer indices for assignment."));
            }
            let child = getitem_axis(py, &cur_obj.bind(py), p)?;
            cur_obj = child;
        }
        let last = &parts[parts.len() - 1];
        return setitem_axis(py, &cur_obj.bind(py), last, value);
    }

    // Fancy coordinate assignment: allow scalar ints and 1D int sequences; disallow slices.
    // All index arrays must have same length; scalar ints are broadcast.
    let mut fancy_lens: Vec<usize> = Vec::new();
    let mut parsed_ints: Vec<Option<Vec<i64>>> = Vec::with_capacity(parts.len());
    let mut parsed_scalars: Vec<Option<i64>> = Vec::with_capacity(parts.len());
    for p in parts {
        if p.downcast::<PySlice>().is_ok() {
            return Err(unsupported("coordinate assignment", "slice combined with fancy arrays is not supported yet", "use integer or fancy indices consistently."));
        }
        if let Ok(i) = p.extract::<i64>() {
            parsed_scalars.push(Some(i));
            parsed_ints.push(None);
        } else if crate::dtype::is_sequence_like(py, p)? {
            let seq = p.downcast::<pyo3::types::PySequence>()?;
            let mut v = Vec::with_capacity(seq.len()? as usize);
            for j in 0..seq.len()? {
                v.push(seq.get_item(j as usize)?.extract::<i64>()?);
            }
            fancy_lens.push(v.len());
            parsed_scalars.push(None);
            parsed_ints.push(Some(v));
        } else {
            return Err(unsupported("coordinate assignment", "unsupported index component", "use int or 1D int sequence per axis."));
        }
    }
    let n = *fancy_lens
        .iter()
        .max()
        .ok_or_else(|| internal("indexing", "unexpected internal state"))?;
    for l in &fancy_lens {
        if *l != n {
            return Err(shape_mismatch(
                "coordinate assignment",
                "multiple index arrays require the same length",
                "align fancy index lengths or use scalar broadcast indices.",
            ));
        }
    }

    // Values: scalar broadcast or sequence length n
    let (values_is_scalar, values_seq) = if crate::dtype::is_sequence_like(py, value)? {
        let vseq = value.downcast::<pyo3::types::PySequence>()?;
        if vseq.len()? as usize != n {
            return Err(shape_mismatch(
                "fancy assignment",
                "value length must match number of selected coordinates",
                "pass one value per selected coordinate.",
            ));
        }
        (false, Some(vseq))
    } else {
        (true, None)
    };

    for k in 0..n {
        // Walk to parent of last axis
        let mut cur_obj = base.clone().unbind();
        for ax in 0..parts.len() - 1 {
            let ix = if let Some(s) = parsed_scalars[ax] {
                s
            } else {
                parsed_ints[ax].as_ref().unwrap()[k]
            };
            let child = getitem_axis(py, &cur_obj.bind(py), &ix.into_py(py).into_bound(py))?;
            cur_obj = child;
        }
        let last_ax = parts.len() - 1;
        let last_ix = if let Some(s) = parsed_scalars[last_ax] {
            s
        } else {
            parsed_ints[last_ax].as_ref().unwrap()[k]
        };
        let v_k = if values_is_scalar {
            value.clone()
        } else {
            values_seq
                .as_ref()
                .unwrap()
                .get_item(k as usize)?
        };
        setitem_axis(py, &cur_obj.bind(py), &last_ix.into_py(py).into_bound(py), &v_k)?;
    }

    Ok(())
}

fn setitem_axis(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let seq = base.downcast::<pyo3::types::PySequence>().map_err(|_| {
        arg_invalid("index", "attempted to assign into a scalar value", "assign only into array/list dimensions.")
    })?;
    let len = seq.len()? as i64;

    if let Ok(slc) = index.downcast::<PySlice>() {
        let (start, stop, step) = parse_slice(py, slc, len)?;
        if step == 0 {
            return Err(arg_invalid("slice step", "step cannot be zero", "use a non-zero slice step."));
        }
        // Collect target indices
        let mut idxs: Vec<usize> = Vec::new();
        let mut i = start;
        if step > 0 {
            while i < stop {
                idxs.push(i as usize);
                i += step;
            }
        } else {
            while i > stop {
                idxs.push(i as usize);
                i += step;
            }
        }
        if crate::dtype::is_sequence_like(py, value)? {
            let vseq = value.downcast::<pyo3::types::PySequence>()?;
            if vseq.len()? as usize != idxs.len() {
                return Err(shape_mismatch(
                    "slice assignment",
                    "value length must match slice length",
                    "pass one value per slice element.",
                ));
            }
            for (k, ix) in idxs.iter().enumerate() {
                let v = vseq.get_item(k)?;
                seq.set_item(*ix, v)?;
            }
        } else {
            for ix in idxs {
                seq.set_item(ix, value.clone())?;
            }
        }
        return Ok(());
    }

    let mut i = index
        .extract::<i64>()
        .map_err(|_| arg_invalid("index", "must be int or slice", "use integer indices or slices for this axis."))?;
    if i < 0 {
        i += len;
    }
    if i < 0 || i >= len {
        return Err(index_out_of_bounds(i as usize, len as usize, "on this axis"));
    }
    seq.set_item(i as usize, value.clone())?;
    Ok(())
}

pub(crate) fn setitem_array(
    py: Python<'_>,
    base: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let base_seq = base.downcast::<pyo3::types::PySequence>()?;
    let n = base_seq.len()? as usize;
    let idx_seq = index.downcast::<pyo3::types::PySequence>()?;
    let m = idx_seq.len()? as usize;

    // boolean mask on axis 0?
    let mut is_all_bool = true;
    let mut bools: Vec<bool> = Vec::new();
    for i in 0..m {
        let it = idx_seq.get_item(i)?;
        if it.is_instance_of::<pyo3::types::PyBool>() {
            bools.push(it.extract::<bool>()?);
        } else {
            is_all_bool = false;
            break;
        }
    }
    if is_all_bool {
        if m != n {
            return Err(shape_mismatch(
                "boolean assignment",
                "mask length must match outer dimension",
                "pass a boolean mask with one entry per row.",
            ));
        }
        let targets: Vec<usize> = (0..n).filter(|i| bools[*i]).collect();
        return assign_outer_positions(py, &base_seq, &targets, value);
    }

    // Outer fancy if len != outer dim.
    if m != n {
        let mut targets: Vec<usize> = Vec::with_capacity(m);
        for i in 0..m {
            let it = idx_seq.get_item(i)?;
            let mut ix = it.extract::<i64>()?;
            if ix < 0 {
                ix += n as i64;
            }
            if ix < 0 || ix >= n as i64 {
                return Err(index_out_of_bounds(ix as usize, n, "on axis 0"));
            }
            targets.push(ix as usize);
        }
        return assign_outer_positions(py, &base_seq, &targets, value);
    }

    // Per-row assignment: m == n
    let values_per_row: Option<Vec<PyObject>> = if crate::dtype::is_sequence_like(py, value)? {
        let vseq = value.downcast::<pyo3::types::PySequence>()?;
        if vseq.len()? as usize == n {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(vseq.get_item(i)?.into());
            }
            Some(out)
        } else {
            None
        }
    } else {
        None
    };

    for i in 0..n {
        let row = base_seq.get_item(i)?;
        let sub = idx_seq.get_item(i)?;
        let apply = |v: &Bound<'_, PyAny>| -> PyResult<()> {
            if let Ok(slc) = sub.downcast::<PySlice>() {
                setitem_axis(py, &row, &slc.clone().into_any(), v)?;
            } else if sub.extract::<i64>().is_ok() {
                // Assign into single position
                setitem_axis(py, &row, &sub, v)?;
            } else if crate::dtype::is_sequence_like(py, &sub)? {
                // Sequence of indices for this row: allow scalar broadcast or sequence matching index count.
                let sseq = sub.downcast::<pyo3::types::PySequence>()?;
                if crate::dtype::is_sequence_like(py, v)? {
                    let vseqq = v.downcast::<pyo3::types::PySequence>()?;
                    if vseqq.len()? != sseq.len()? {
                        return Err(shape_mismatch(
                            "per-row assignment",
                            "value length must match selected row indices",
                            "pass one value per selected index in the row.",
                        ));
                    }
                    for j in 0..sseq.len()? {
                        let jx = sseq.get_item(j as usize)?;
                        let vv = vseqq.get_item(j as usize)?;
                        setitem_axis(py, &row, &jx, &vv)?;
                    }
                } else {
                    for j in 0..sseq.len()? {
                        let jx = sseq.get_item(j as usize)?;
                        setitem_axis(py, &row, &jx, v)?;
                    }
                }
            } else {
                return Err(unsupported("per-row indexing", "unsupported index element type", "use int, slice, or sequence of ints per row."));
            }
            Ok(())
        };

        if let Some(vs) = &values_per_row {
            let v_i = vs[i].bind(py);
            apply(&v_i)?;
        } else {
            apply(value)?;
        }
    }
    Ok(())
}

fn assign_outer_positions(
    py: Python<'_>,
    base_seq: &Bound<'_, pyo3::types::PySequence>,
    targets: &[usize],
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if crate::dtype::is_sequence_like(py, value)? {
        let vseq = value.downcast::<pyo3::types::PySequence>()?;
        if vseq.len()? as usize != targets.len() {
            return Err(shape_mismatch(
                "fancy assignment",
                "value length must match number of selected elements",
                "pass one value per selected index.",
            ));
        }
        for (k, ix) in targets.iter().enumerate() {
            base_seq.set_item(*ix, vseq.get_item(k)?)?;
        }
    } else {
        for ix in targets {
            base_seq.set_item(*ix, value.clone())?;
        }
    }
    Ok(())
}
