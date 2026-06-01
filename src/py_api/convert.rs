use crate::dtype::DType;
use crate::layout::{build_array, GrumpyArray, Layout};
use crate::py_api::types::PyGrumpyArray;
use numpy::{PyArray1, PyArrayMethods, Element};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyModule};


pub(crate) fn sizes_to_vec_usize_fast(_py: Python<'_>, sizes: &Bound<'_, PyAny>) -> PyResult<Option<Vec<usize>>> {
    if let Ok(seq) = sizes.downcast::<pyo3::types::PySequence>() {
        let n = seq.len()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let v = seq.get_item(i)?;
            let x: i64 = v.extract()?;
            if x < 0 {
                return Err(PyValueError::new_err("sizes must be non-negative."));
            }
            out.push(x as usize);
        }
        return Ok(Some(out));
    }
    if let Ok(gr_sizes) = sizes.extract::<PyRef<'_, PyGrumpyArray>>() {
        if gr_sizes.inner.dtype != DType::Int64 {
            return Ok(None);
        }
        if let Layout::Leaf(leaf) = &gr_sizes.inner.layout {
            if leaf.has_nulls {
                return Err(PyValueError::new_err("sizes array must not contain nulls."));
            }
            if let crate::layout::LeafBuffer::I64(v) = &leaf.buffer {
                let mut out = Vec::with_capacity(leaf.len);
                for &x in v.as_slice().iter().take(leaf.len) {
                    if x < 0 {
                        return Err(PyValueError::new_err("sizes must be non-negative."));
                    }
                    out.push(x as usize);
                }
                return Ok(Some(out));
            }
        }
    }
    Ok(None)
}

pub(crate) fn leaf_to_numpy_1d_typed(
    py: Python<'_>,
    leaf: &crate::layout::Leaf,
    dt: DType,
) -> PyResult<Option<PyObject>> {
    let n = leaf.len;
    match dt {
        DType::Int32 => Ok(Some(bytes_to_numpy_1d::<i32>(py, leaf.buffer.as_bytes(), n)?)),
        DType::Int64 => Ok(Some(bytes_to_numpy_1d::<i64>(py, leaf.buffer.as_bytes(), n)?)),
        DType::Float32 => Ok(Some(bytes_to_numpy_1d::<f32>(py, leaf.buffer.as_bytes(), n)?)),
        DType::Float64 => Ok(Some(bytes_to_numpy_1d::<f64>(py, leaf.buffer.as_bytes(), n)?)),
        // extend later (uint/bool/float16) once benchmarks show it's needed
        _ => Ok(None),
    }
}

fn bytes_to_numpy_1d<T: numpy::Element>(py: Python<'_>, bytes: &[u8], n: usize) -> PyResult<PyObject> {
    let expected = n * std::mem::size_of::<T>();
    if bytes.len() != expected {
        return Err(PyValueError::new_err("Internal error: leaf byte size mismatch."));
    }

    let arr = PyArray1::<T>::zeros_bound(py, n, false);
    unsafe {
        let dst = arr.data() as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    Ok(arr.into_py(py))
}

pub(crate) fn sizes_to_list_any(py: Python<'_>, sizes: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    // If sizes is a GrumpyArray, use its to_list.
    if let Ok(arr) = sizes.extract::<PyRef<'_, PyGrumpyArray>>() {
        return arr.inner.to_py_list(py);
    }
    Ok(sizes.clone().unbind())
}

pub(crate) fn max_list_depth(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    if crate::dtype::is_sequence_like(py, obj)? {
        let seq = obj.downcast::<pyo3::types::PySequence>()?;
        let mut m = 0usize;
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            m = m.max(max_list_depth(py, &it)?);
        }
        Ok(m + 1)
    } else {
        Ok(0)
    }
}

pub(crate) fn normalize_dim(dim: isize, max_depth: isize) -> PyResult<usize> {
    let mut d = dim;
    if d < 0 {
        d += max_depth;
    }
    if d < 0 || d > max_depth {
        return Err(PyValueError::new_err("dim out of range."));
    }
    Ok(d as usize)
}

pub(crate) fn parse_dims(py: Python<'_>, obj: &Bound<'_, PyAny>, max_depth: usize) -> PyResult<Vec<isize>> {
    let md = max_depth as isize;
    if let Ok(i) = obj.extract::<isize>() {
        let d = normalize_dim(i, md)? as isize;
        return Ok(vec![d]);
    }
    if crate::dtype::is_sequence_like(py, obj)? {
        let seq = obj.downcast::<pyo3::types::PySequence>()?;
        let mut out = Vec::with_capacity(seq.len()? as usize);
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            let v = it.extract::<isize>()?;
            out.push(normalize_dim(v, md)? as isize);
        }
        return Ok(out);
    }
    Err(PyValueError::new_err("dim/but must be an int or a sequence of ints."))
}

pub(crate) fn flatten_collect(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    level: usize,
    axes_to_remove: &std::collections::BTreeSet<usize>,
) -> PyResult<Vec<PyObject>> {
    if !crate::dtype::is_sequence_like(py, obj)? {
        return Ok(vec![obj.clone().unbind()]);
    }
    let seq = obj.downcast::<pyo3::types::PySequence>()?;
    if axes_to_remove.contains(&level) {
        // Remove this list level: splice children into parent.
        let mut out: Vec<PyObject> = Vec::new();
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            out.extend(flatten_collect(py, &it, level + 1, axes_to_remove)?);
        }
        Ok(out)
    } else {
        // Keep this list level, but allow deeper levels to be removed (splicing into this list).
        let out_list = pyo3::types::PyList::empty_bound(py);
        for i in 0..seq.len()? {
            let it = seq.get_item(i as usize)?;
            for e in flatten_collect(py, &it, level + 1, axes_to_remove)? {
                out_list.append(e)?;
            }
        }
        Ok(vec![out_list.into_py(py)])
    }
}

pub(crate) fn unflatten_rec(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    dim: usize,
) -> PyResult<PyObject> {
    if dim == 0 {
        let data_seq = data.downcast::<pyo3::types::PySequence>().map_err(|_| {
            PyValueError::new_err("unflatten(dim=0) requires a sequence of values.")
        })?;
        let sizes_seq = sizes.downcast::<pyo3::types::PySequence>().map_err(|_| {
            PyValueError::new_err("sizes must be a sequence of integers.")
        })?;
        let total = data_seq.len()? as usize;
        let out = pyo3::types::PyList::empty_bound(py);
        let mut pos = 0usize;
        for i in 0..sizes_seq.len()? {
            let s = sizes_seq.get_item(i as usize)?.extract::<isize>()?;
            if s < 0 {
                return Err(PyValueError::new_err("sizes entries must be non-negative."));
            }
            let s = s as usize;
            if pos + s > total {
                return Err(PyValueError::new_err(
                    "sizes do not sum to the number of elements being unflattened.",
                ));
            }
            let chunk = pyo3::types::PyList::empty_bound(py);
            for j in 0..s {
                chunk.append(data_seq.get_item((pos + j) as usize)?)?;
            }
            pos += s;
            out.append(chunk)?;
        }
        if pos != total {
            return Err(PyValueError::new_err(
                "sizes do not sum to the number of elements being unflattened.",
            ));
        }
        return Ok(out.into());
    }

    let data_seq = data.downcast::<pyo3::types::PySequence>().map_err(|_| {
        PyValueError::new_err("unflatten requires list-like structure at the target axis.")
    })?;
    let sizes_seq = sizes.downcast::<pyo3::types::PySequence>().map_err(|_| {
        PyValueError::new_err("sizes must match the structure of the array along outer axes.")
    })?;
    if data_seq.len()? != sizes_seq.len()? {
        return Err(PyValueError::new_err(
            "sizes must have the same outer length as the array along axes above dim.",
        ));
    }
    let out = pyo3::types::PyList::empty_bound(py);
    for i in 0..data_seq.len()? {
        let sub_data = data_seq.get_item(i as usize)?;
        let sub_sizes = sizes_seq.get_item(i as usize)?;
        out.append(unflatten_rec(py, &sub_data, &sub_sizes, dim - 1)?)?;
    }
    Ok(out.into())
}

pub(crate) fn layout_all_valid_no_union(layout: &Layout) -> bool {
    match layout {
        Layout::Leaf(l) => !l.has_nulls,
        Layout::ListOffset(lo) => layout_all_valid_no_union(lo.content.as_ref()),
        Layout::Indexed(ix) => layout_all_valid_no_union(ix.content.as_ref()),
        Layout::OffsetView(v) => layout_all_valid_no_union(v.content.as_ref()),
        Layout::UnionScalarList(_) => false,
    }
}

pub(crate) fn numpy_dtype<'py>(
    np: &Bound<'py, PyModule>,
    dt: DType,
) -> PyResult<Option<(Bound<'py, PyAny>, &'static str)>> {
    let (attr, expected) = match dt {
        DType::Int8 => ("int8", "int8"),
        DType::Int16 => ("int16", "int16"),
        DType::Int32 => ("int32", "int32"),
        DType::Int64 => ("int64", "int64"),
        DType::UInt8 => ("uint8", "uint8"),
        DType::UInt16 => ("uint16", "uint16"),
        DType::UInt32 => ("uint32", "uint32"),
        DType::UInt64 => ("uint64", "uint64"),
        DType::Float16 => ("float16", "float16"),
        DType::Float32 => ("float32", "float32"),
        DType::Float64 => ("float64", "float64"),
        DType::Bool => ("bool_", "bool"),
        DType::Char | DType::String => return Ok(None),
    };
    Ok(Some((np.getattr(attr)?, expected)))
}

pub(crate) fn shape_or_nshape(
    py: Python<'_>,
    arr: &GrumpyArray,
    dim: usize,
    count_non_null_scalars: bool,
) -> PyResult<PyObject> {
    if dim == 0 {
        return Ok((arr.len() as i64).into_py(py));
    }
    let target_axis = dim - 1;
    let out_list = pyo3::types::PyList::empty_bound(py);
    for i in 0..arr.len() {
        let obj = collect_shape_for_element(
            py,
            &arr.layout,
            i,
            0,
            target_axis,
            count_non_null_scalars,
        )?;
        out_list.append(obj)?;
    }
    let out_obj = out_list.into_py(py);
    let built = build_array(py, &out_obj.bind(py), DType::Int64)?;
    Ok(PyGrumpyArray { inner: built }.into_py(py))
}

fn collect_shape_for_element(
    py: Python<'_>,
    layout: &Layout,
    idx: usize,
    axis: usize,
    target_axis: usize,
    count_non_null_scalars: bool,
) -> PyResult<PyObject> {
    if axis == target_axis {
        // At target axis we want the "length at next axis" of list elements only.
        if let Layout::ListOffset(lo) = layout {
            let len = if count_non_null_scalars {
                lo.child_len_non_null_scalars(idx)?
            } else {
                lo.child_len_total(idx)?
            };
            return Ok((len as i64).into_py(py));
        }
        if let Layout::OffsetView(v) = layout {
            let abs = v.start + idx;
            let start = v.offsets[abs] as usize;
            let end = v.offsets[abs + 1] as usize;
            let len = end - start;
            return Ok((len as i64).into_py(py));
        }
        if let Layout::Indexed(ix) = layout {
            // resolve and retry
            let n = ix.content.len() as i64;
            let mut j = ix.index[idx];
            if j < 0 { j += n; }
            if j < 0 || j >= n { return Err(PyValueError::new_err("Index out of bounds.")); }
            return collect_shape_for_element(py, ix.content.as_ref(), j as usize, axis, target_axis, count_non_null_scalars);
        }
        // A scalar (or union element that is scalar) at target axis does not contribute.
        return Ok(py.None());
    }

    match layout {
        Layout::ListOffset(lo) => {
            // We are above the target axis: build a list from children that have a path to the target.
            let start = lo.offsets[idx] as usize;
            let end = lo.offsets[idx + 1] as usize;
            let out = pyo3::types::PyList::empty_bound(py);
            for j in start..end {
                let child = collect_shape_for_any(
                    py,
                    lo.content.as_ref(),
                    j,
                    axis + 1,
                    target_axis,
                    count_non_null_scalars,
                )?;
                if !child.is_none(py) {
                    out.append(child)?;
                }
            }
            Ok(out.into())
        }
        Layout::OffsetView(v) => {
            let abs = v.start + idx;
            let start = v.offsets[abs] as usize;
            let end = v.offsets[abs + 1] as usize;
            let out = pyo3::types::PyList::empty_bound(py);
            for j in start..end {
                let child = collect_shape_for_any(
                    py,
                    v.content.as_ref(),
                    j,
                    axis + 1,
                    target_axis,
                    count_non_null_scalars,
                )?;
                if !child.is_none(py) {
                    out.append(child)?;
                }
            }
            Ok(out.into())
        }
        Layout::Indexed(ix) => {
            // Resolve the indexed element then continue.
            let n = ix.content.len() as i64;
            let mut j = ix.index[idx];
            if j < 0 { j += n; }
            if j < 0 || j >= n {
                return Err(PyValueError::new_err("Index out of bounds."));
            }
            collect_shape_for_element(py, ix.content.as_ref(), j as usize, axis, target_axis, count_non_null_scalars)
        }
        Layout::UnionScalarList(u) => {
            // A union element at this axis: we only descend if it is a list.
            // If scalar, it yields empty list for axes above target (no deeper list nodes).
            let tag = u.tags[idx];
            let ix = u.index[idx] as usize;
            match tag {
                0 => Ok(pyo3::types::PyList::empty_bound(py).into()),
                1 => collect_shape_for_any(
                    py,
                    &Layout::ListOffset(u.lists.clone()),
                    ix,
                    axis,
                    target_axis,
                    count_non_null_scalars,
                ),
                _ => Err(PyValueError::new_err("Invalid union tag.")),
            }
        }
        Layout::Leaf(_) => Ok(pyo3::types::PyList::empty_bound(py).into()),
    }
}

fn collect_shape_for_any(
    py: Python<'_>,
    layout: &Layout,
    idx: usize,
    axis: usize,
    target_axis: usize,
    count_non_null_scalars: bool,
) -> PyResult<PyObject> {
    if axis == target_axis {
        match layout {
            Layout::ListOffset(lo) => {
                let len = if count_non_null_scalars {
                    lo.child_len_non_null_scalars(idx)?
                } else {
                    lo.child_len_total(idx)?
                };
                Ok((len as i64).into_py(py))
            }
            Layout::OffsetView(v) => {
                let abs = v.start + idx;
                let start = v.offsets[abs] as usize;
                let end = v.offsets[abs + 1] as usize;
                Ok(((end - start) as i64).into_py(py))
            }
            Layout::Indexed(ix) => {
                let n = ix.content.len() as i64;
                let mut j = ix.index[idx];
                if j < 0 { j += n; }
                if j < 0 || j >= n {
                    return Err(PyValueError::new_err("Index out of bounds."));
                }
                collect_shape_for_any(py, ix.content.as_ref(), j as usize, axis, target_axis, count_non_null_scalars)
            }
            Layout::UnionScalarList(u) => {
                // At target axis, only list elements contribute.
                let tag = u.tags[idx];
                let ix = u.index[idx] as usize;
                if tag == 1 {
                    let lo = &u.lists;
                    let len = if count_non_null_scalars {
                        lo.child_len_non_null_scalars(ix)?
                    } else {
                        lo.child_len_total(ix)?
                    };
                    Ok((len as i64).into_py(py))
                } else {
                    Ok(py.None())
                }
            }
            Layout::Leaf(_) => Ok(py.None()),
        }
    } else {
        collect_shape_for_element(
            py,
            layout,
            idx,
            axis,
            target_axis,
            count_non_null_scalars,
        )
    }
}

pub(crate) fn wrap_result(py: Python<'_>, out: PyObject, dtype: DType) -> PyResult<PyObject> {
    let bound = out.bind(py);
    if crate::dtype::is_sequence_like(py, &bound)? {
        let arr = build_array(py, &bound, dtype)?;
        Ok(PyGrumpyArray { inner: arr }.into_py(py))
    } else {
        Ok(out)
    }
}

pub(crate) fn wrap_index_layout_result(py: Python<'_>, dtype: DType, layout: Layout) -> PyResult<PyObject> {
    if let Layout::Leaf(l) = &layout {
        if l.len == 1 {
            return l.scalar_to_py(py, 0);
        }
    }
    Ok(PyGrumpyArray {
        inner: GrumpyArray { dtype, layout },
    }
    .into_py(py))
}
