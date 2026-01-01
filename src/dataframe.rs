use crate::dtype::DType;
use crate::layout::{drop_axis0_select_element, take_range, GrumpyArray, Layout};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PySequence, PySlice, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Schema {
    // Each dimension level can have multiple alias names (tuple in Python schema).
    pub levels: Vec<Vec<String>>,
    pub name_to_level: HashMap<String, usize>,
}

impl Schema {
    pub fn parse(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let seq = obj
            .downcast::<PySequence>()
            .map_err(|_| PyValueError::new_err("schema must be a list/tuple of strings or tuples of strings."))?;
        let mut levels: Vec<Vec<String>> = Vec::new();
        for i in 0..seq.len()? {
            let it = seq.get_item(i)?;
            if let Ok(s) = it.extract::<String>() {
                levels.push(vec![s]);
            } else if let Ok(t) = it.downcast::<PyTuple>() {
                let mut al: Vec<String> = Vec::new();
                for j in 0..t.len() {
                    al.push(t.get_item(j)?.extract::<String>().map_err(|_| {
                        PyValueError::new_err("schema tuples must contain only strings.")
                    })?);
                }
                if al.is_empty() {
                    return Err(PyValueError::new_err("schema tuples cannot be empty."));
                }
                levels.push(al);
            } else {
                return Err(PyValueError::new_err(
                    "schema must contain strings or tuples of strings.",
                ));
            }
        }
        if levels.is_empty() {
            return Err(PyValueError::new_err("schema cannot be empty."));
        }
        let mut name_to_level = HashMap::new();
        for (lvl, names) in levels.iter().enumerate() {
            for n in names {
                if name_to_level.insert(n.clone(), lvl).is_some() {
                    return Err(PyValueError::new_err(format!(
                        "schema name '{}' appears more than once.",
                        n
                    )));
                }
            }
        }
        Ok(Self { levels, name_to_level })
    }

    pub fn level_for_column(&self, colname: &str) -> PyResult<usize> {
        // Column must start with "<prefix>_" or equal "<prefix>".
        for (lvl, names) in self.levels.iter().enumerate() {
            for p in names {
                if colname == p || colname.starts_with(&format!("{p}_")) {
                    return Ok(lvl);
                }
            }
        }
        Err(PyValueError::new_err(format!(
            "Column '{}' does not start with any valid schema prefix.",
            colname
        )))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CanonShape {
    pub nrows: Option<usize>,
    // For each level >=1, store canonical offsets of the corresponding listoffset axis.
    pub offsets: Vec<Option<Vec<i64>>>,
}

#[derive(Clone, Debug)]
pub struct GrumpyDataFrame {
    pub names: Vec<String>,
    pub cols: Vec<GrumpyArray>,
    pub schema: Option<Schema>,
    pub canon: CanonShape,
}

impl GrumpyDataFrame {
    pub fn new(names: Vec<String>, cols: Vec<GrumpyArray>, schema: Option<Schema>) -> PyResult<Self> {
        if names.len() != cols.len() {
            return Err(PyValueError::new_err("Internal error: names/cols mismatch."));
        }
        let mut df = Self { names, cols, schema, canon: CanonShape::default() };
        df.recompute_canon()?;
        Ok(df)
    }

    pub fn nrows(&self) -> usize {
        self.canon.nrows.unwrap_or(0)
    }

    pub fn to_pydict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        for (name, col) in self.names.iter().zip(self.cols.iter()) {
            d.set_item(name, col.to_py_list(py)?)?;
        }
        Ok(d.into())
    }

    pub fn max_all(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Column-wise max over all scalar values (flattened), skipping nulls.
        let d = PyDict::new_bound(py);
        for (name, col) in self.names.iter().zip(self.cols.iter()) {
            let v = flat_max_scalar(py, col)?;
            d.set_item(name, v)?;
        }
        Ok(d.into())
    }

    pub fn column_subset(&self, names: &[String]) -> PyResult<Self> {
        let mut out_names = Vec::new();
        let mut out_cols = Vec::new();
        for n in names {
            let mut found = false;
            for (nn, cc) in self.names.iter().zip(self.cols.iter()) {
                if nn == n {
                    out_names.push(nn.clone());
                    out_cols.push(cc.clone());
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(PyValueError::new_err(format!("Unknown column '{}'.", n)));
            }
        }
        GrumpyDataFrame::new(out_names, out_cols, self.schema.clone())
    }

    pub fn row_select_indexed(&self, idx: Arc<Vec<i64>>) -> PyResult<Self> {
        let mut out_cols: Vec<GrumpyArray> = Vec::with_capacity(self.cols.len());
        for c in &self.cols {
            // Clamp indices to this column's length (so dataframes with uneven column lengths can still be sliced).
            let n = c.len() as i64;
            let mut sub: Vec<i64> = Vec::new();
            for &j in idx.iter() {
                let jj = if j < 0 { j + n } else { j };
                if jj >= 0 && jj < n {
                    sub.push(jj);
                }
            }
            // If this is a contiguous increasing range and the column is a top-level ListOffset,
            // use an OffsetView (cheaper and preserves correct offset slicing semantics).
            let layout = if sub.len() > 0
                && sub.windows(2).all(|w| w[1] == w[0] + 1)
                && matches!(&c.layout, Layout::ListOffset(_))
            {
                let start = sub[0] as usize;
                let stop = (sub[sub.len() - 1] + 1) as usize;
                let lo = match &c.layout {
                    Layout::ListOffset(lo) => lo,
                    _ => unreachable!(),
                };
                Layout::OffsetView(crate::layout::OffsetView {
                    offsets: lo.offsets.clone(),
                    start,
                    stop,
                    content: lo.content.clone(),
                })
            } else {
                Layout::Indexed(crate::layout::Indexed { index: Arc::new(sub), content: Box::new(c.layout.clone()) })
            };
            out_cols.push(GrumpyArray { dtype: c.dtype, layout });
        }
        GrumpyDataFrame::new(self.names.clone(), out_cols, self.schema.clone())
    }

    /// Fast axis-0 slice without building an explicit index vector.
    ///
    /// Uses `OffsetView` for top-level `ListOffset` columns to preserve offsets semantics and
    /// avoid copies. For non-ListOffset columns, falls back to `take_range`.
    pub fn row_slice_view(&self, start: usize, stop: usize) -> PyResult<Self> {
        if start > stop {
            return Err(PyValueError::new_err("Invalid slice range."));
        }
        let mut out_cols: Vec<GrumpyArray> = Vec::with_capacity(self.cols.len());
        for c in &self.cols {
            let n = c.len();
            let s = start.min(n);
            let e = stop.min(n);
            let layout = match &c.layout {
                Layout::ListOffset(lo) => Layout::OffsetView(crate::layout::OffsetView {
                    offsets: lo.offsets.clone(),
                    start: s,
                    stop: e,
                    content: lo.content.clone(),
                }),
                Layout::OffsetView(v) => {
                    let ss = (v.start + s).min(v.stop);
                    let ee = (v.start + e).min(v.stop);
                    Layout::OffsetView(crate::layout::OffsetView {
                        offsets: v.offsets.clone(),
                        start: ss,
                        stop: ee,
                        content: v.content.clone(),
                    })
                }
                _ => take_range(&c.layout, s, e)?,
            };
            out_cols.push(GrumpyArray { dtype: c.dtype, layout });
        }
        GrumpyDataFrame::new(self.names.clone(), out_cols, self.schema.clone())
    }

    pub fn set_column(&mut self, py: Python<'_>, name: String, value: &Bound<'_, PyAny>, dtype: Option<DType>) -> PyResult<()> {
        let dt = dtype.unwrap_or(DType::Float64);
        let arr = crate::layout::build_array(py, value, dt)?;
        self.set_column_array(name, arr)
    }

    pub fn set_column_array(&mut self, name: String, arr: GrumpyArray) -> PyResult<()> {
        // Validate against schema + existing canonical shapes.
        self.validate_column(&name, &arr)?;

        // Replace if exists.
        for i in 0..self.names.len() {
            if self.names[i] == name {
                self.cols[i] = arr;
                self.recompute_canon()?;
                return Ok(());
            }
        }
        self.names.push(name);
        self.cols.push(arr);
        self.recompute_canon()?;
        Ok(())
    }

    /// Re-nest a flat-by-level array back into axis-0 nesting using canonical schema offsets.
    ///
    /// This supports dot-notation assignment ergonomics:
    /// - `df.<level>.<col> = rhs` where `rhs` is produced by `df.<level>.<other>.mean(...)`
    ///
    /// Accepted RHS shapes:
    /// - **already nested at axis-0** (outer length == `df.nrows()`): returned unchanged
    /// - **flat-by-level** (outer length == total elements at the schema level): wrapped in nested `ListOffset`s
    ///   using canonical offsets at levels `1..=level`.
    pub fn renest_rhs_for_level(&self, level: usize, level_name: &str, rhs: GrumpyArray) -> PyResult<GrumpyArray> {
        if level == 0 {
            return Ok(rhs);
        }
        if rhs.len() == self.nrows() {
            return Ok(rhs);
        }
        let canon_off_level = self
            .canon
            .offsets
            .get(level)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Cannot re-nest: missing canonical offsets for schema level {level} ('{level_name}')."
                ))
            })?;
        let total = *canon_off_level.last().unwrap() as usize;
        if rhs.len() != total {
            return Err(PyValueError::new_err(format!(
                "Dot-notation assignment at '{level_name}': RHS must have outer length {total} (total elements at that level) or {} (axis-0 length), but has {}.",
                self.nrows(),
                rhs.len()
            )));
        }
        let mut cur = rhs.layout;
        for lev in (1..=level).rev() {
            let canon_off = self
                .canon
                .offsets
                .get(lev)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Cannot re-nest: missing canonical offsets for schema level {lev}."
                    ))
                })?
                .clone();
            cur = Layout::ListOffset(crate::layout::ListOffset {
                offsets: Arc::new(canon_off),
                content: Box::new(cur),
            });
        }
        Ok(GrumpyArray { dtype: rhs.dtype, layout: cur })
    }

    fn validate_column(&self, name: &str, arr: &GrumpyArray) -> PyResult<()> {
        // Determine outer length.
        let n = arr.len();
        if let Some(cur) = self.canon.nrows {
            if self.schema.is_some() && cur != n {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' has length {}, but dataframe has length {} (schema requires equal lengths).",
                    name, n, cur
                )));
            }
        }

        if let Some(schema) = &self.schema {
            let lvl = schema.level_for_column(name)?;
            let ndim = array_ndim(&arr.layout)?;
            if ndim < lvl + 1 {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' must have at least {} dimensions due to schema prefix, but has {}.",
                    name,
                    lvl + 1,
                    ndim
                )));
            }
            // Validate canonical shapes up to the schema level of this column.
            // Axis 0 is length; for each further schema level L>=1, compare the listoffset offsets at axis L-1.
            let max_level = schema.levels.len() - 1;
            let want = std::cmp::min(std::cmp::min(max_level, lvl), ndim - 1);
            // Offsets vector length is (parent_len + 1).
            for lev in 1..=want {
                if let Some(off) = offsets_at_level(&arr.layout, lev)? {
                    if self.canon.offsets.len() <= lev {
                        // no canon recorded here
                    }
                    if let Some(canon_off) = self.canon.offsets.get(lev).and_then(|x| x.as_ref()) {
                        if canon_off.as_slice() != off.as_slice() {
                            return Err(PyValueError::new_err(format!(
                                "Column '{}' does not match schema shape at level {}.",
                                name, lev
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn recompute_canon(&mut self) -> PyResult<()> {
        // Determine canonical nrows and offsets for each schema level (if schema provided).
        if self.cols.is_empty() {
            self.canon = CanonShape::default();
            return Ok(());
        }
        // Without a schema, we allow uneven column lengths (dict-of-arrays style).
        // With a schema, we enforce equal outer length.
        let mut nrows_max = 0usize;
        for c in &self.cols {
            nrows_max = nrows_max.max(c.len());
        }
        if self.schema.is_some() {
            let nrows = self.cols[0].len();
            for (i, c) in self.cols.iter().enumerate() {
                let n = c.len();
                if n != nrows {
                    return Err(PyValueError::new_err(format!(
                        "Dataframe columns must have same length under a schema; '{}' has {} but first column has {}.",
                        self.names[i], n, nrows
                    )));
                }
            }
            self.canon.nrows = Some(nrows);
        } else {
            self.canon.nrows = Some(nrows_max);
        }
        if let Some(schema) = &self.schema {
            let nlev = schema.levels.len();
            let mut offsets: Vec<Option<Vec<i64>>> = vec![None; nlev];
            // Scan columns: for each axis lev>=1 that exists in the column, record canonical offsets (first seen).
            for c in &self.cols {
                let ndim = array_ndim(&c.layout)?;
                for lev in 1..std::cmp::min(nlev, ndim) {
                    if offsets[lev].is_none() {
                        offsets[lev] = offsets_at_level(&c.layout, lev)?;
                    }
                }
            }
            self.canon.offsets = offsets;
        } else {
            self.canon.offsets.clear();
        }
        Ok(())
    }
}

fn array_ndim(layout: &Layout) -> PyResult<usize> {
    // Number of axes = list_chain_depth + 1, for pure list chains; for unions we conservatively error for now.
    let depth = crate::layout::list_chain_depth(layout)
        .ok_or_else(|| PyValueError::new_err("DataFrame currently requires pure list-chain arrays for schema validation."))?;
    Ok(depth + 1)
}

fn offsets_at_level(layout: &Layout, level: usize) -> PyResult<Option<Vec<i64>>> {
    // level=1 => offsets of first ListOffset in chain; level=2 => second ListOffset, etc.
    // For leaf arrays (ndim=1), returns None.
    let mut cur = layout;
    let mut seen = 0usize;
    loop {
        match cur {
            Layout::Leaf(_) => return Ok(None),
            Layout::ListOffset(lo) => {
                seen += 1;
                if seen == level {
                    return Ok(Some(lo.offsets.as_slice().to_vec()));
                }
                cur = lo.content.as_ref();
            }
            Layout::OffsetView(v) => {
                // Materialize the relevant offsets slice as canonical for schema comparisons.
                // NOTE: This is a copy; acceptable for validation.
                seen += 1;
                if seen == level {
                    let start = v.start;
                    let stop = v.stop;
                    let base = v.offsets[start];
                    let mut out = Vec::with_capacity(stop - start + 1);
                    for i in start..=stop {
                        out.push(v.offsets[i] - base);
                    }
                    return Ok(Some(out));
                }
                cur = v.content.as_ref();
            }
            Layout::Indexed(ix) => {
                // If the index wraps a ListOffset at this level, compute canonical offsets *after* applying
                // the selection. This is critical for streamed/sliced dataframes where columns are views.
                if let Layout::ListOffset(lo) = ix.content.as_ref() {
                    // Selecting axis-0 elements of `lo`.
                    seen += 1;
                    if seen == level {
                        let n = lo.len() as i64;
                        let mut out: Vec<i64> = Vec::with_capacity(ix.index.len() + 1);
                        out.push(0);
                        let mut acc: i64 = 0;
                        for &raw in ix.index.iter() {
                            let mut j = raw;
                            if j < 0 {
                                j += n;
                            }
                            if j < 0 || j >= n {
                                continue;
                            }
                            let s = lo.offsets[j as usize];
                            let e = lo.offsets[j as usize + 1];
                            acc += e - s;
                            out.push(acc);
                        }
                        return Ok(Some(out));
                    }
                    // For deeper levels, we currently fall through by ignoring the indexing wrapper.
                    // This may be extended later to propagate selection through deeper offsets.
                    cur = lo.content.as_ref();
                } else {
                    cur = ix.content.as_ref();
                }
            }
            Layout::UnionScalarList(_) => {
                return Err(PyValueError::new_err(
                    "Schema validation across union layouts is not implemented yet.",
                ))
            }
        }
    }
}

fn flat_max_scalar(py: Python<'_>, arr: &GrumpyArray) -> PyResult<PyObject> {
    // Flatten over all scalars in the layout; skip nulls. For floats, propagate NaN like NumPy max (if any NaN present -> NaN).
    match arr.dtype {
        DType::Int32 | DType::Int64 => flat_max_int(py, arr),
        DType::UInt32 | DType::UInt64 => flat_max_uint(py, arr),
        DType::Float32 | DType::Float64 => flat_max_float(py, arr),
        _ => Err(PyValueError::new_err("DataFrame max currently only supports numeric dtypes.")),
    }
}

fn flat_max_int(py: Python<'_>, arr: &GrumpyArray) -> PyResult<PyObject> {
    let mut have = false;
    let mut best: i64 = 0i64;
    walk_numeric(arr, &mut |v_i64, _v_u64, _v_f64, kind| {
        if kind != NumKind::Int {
            return Ok(());
        }
        let x = v_i64;
        if !have {
            best = x;
            have = true;
        } else if x > best {
            best = x;
        }
        Ok(())
    })?;
    if !have {
        Ok(py.None())
    } else {
        Ok(best.into_py(py))
    }
}

fn flat_max_uint(py: Python<'_>, arr: &GrumpyArray) -> PyResult<PyObject> {
    let mut have = false;
    let mut best: u64 = 0;
    walk_numeric(arr, &mut |_v_i64, v_u64, _v_f64, kind| {
        if kind != NumKind::UInt {
            return Ok(());
        }
        let x = v_u64;
        if !have {
            best = x;
            have = true;
        } else if x > best {
            best = x;
        }
        Ok(())
    })?;
    if !have {
        Ok(py.None())
    } else {
        Ok(best.into_py(py))
    }
}

fn flat_max_float(py: Python<'_>, arr: &GrumpyArray) -> PyResult<PyObject> {
    let mut have = false;
    let mut best: f64 = 0.0;
    let mut seen_nan = false;
    walk_numeric(arr, &mut |_v_i64, _v_u64, v_f64, kind| {
        if kind != NumKind::Float {
            return Ok(());
        }
        let x = v_f64;
        if x.is_nan() {
            seen_nan = true;
        }
        if !have {
            best = x;
            have = true;
        } else if x > best {
            best = x;
        }
        Ok(())
    })?;
    if !have {
        Ok(py.None())
    } else if seen_nan {
        Ok(f64::NAN.into_py(py))
    } else {
        Ok(best.into_py(py))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NumKind {
    Int,
    UInt,
    Float,
}

fn walk_numeric<F>(arr: &GrumpyArray, f: &mut F) -> PyResult<()>
where
    F: FnMut(i64, u64, f64, NumKind) -> PyResult<()>,
{
    fn walk<F>(layout: &Layout, dtype: DType, f: &mut F) -> PyResult<()>
    where
        F: FnMut(i64, u64, f64, NumKind) -> PyResult<()>,
    {
        match layout {
            Layout::Leaf(l) => {
                for i in 0..l.len {
                    if !l.validity[i] {
                        continue;
                    }
                    match (&l.buffer, dtype) {
                        (crate::layout::LeafBuffer::I32(v), _) => f(v[i] as i64, 0, 0.0, NumKind::Int)?,
                        (crate::layout::LeafBuffer::I64(v), _) => f(v[i] as i64, 0, 0.0, NumKind::Int)?,
                        (crate::layout::LeafBuffer::U32(v), _) => f(0, v[i] as u64, 0.0, NumKind::UInt)?,
                        (crate::layout::LeafBuffer::U64(v), _) => f(0, v[i] as u64, 0.0, NumKind::UInt)?,
                        (crate::layout::LeafBuffer::F32(v), _) => f(0, 0, v[i] as f64, NumKind::Float)?,
                        (crate::layout::LeafBuffer::F64(v), _) => f(0, 0, v[i], NumKind::Float)?,
                        _ => return Err(PyValueError::new_err("Unsupported dtype for DataFrame reduction.")),
                    }
                }
                Ok(())
            }
            Layout::ListOffset(lo) => {
                for i in 0..lo.len() {
                    let s = lo.offsets[i] as usize;
                    let e = lo.offsets[i + 1] as usize;
                    let sub = take_range(lo.content.as_ref(), s, e)?;
                    walk(&sub, dtype, f)?;
                }
                Ok(())
            }
            Layout::OffsetView(v) => {
                for i in 0..v.len() {
                    let sub = drop_axis0_select_element(layout, i)?;
                    walk(&sub, dtype, f)?;
                }
                Ok(())
            }
            Layout::Indexed(ix) => {
                for i in 0..ix.len() {
                    let sub = drop_axis0_select_element(layout, i)?;
                    walk(&sub, dtype, f)?;
                }
                Ok(())
            }
            Layout::UnionScalarList(u) => {
                // Walk scalar branch and list branch.
                for i in 0..u.len() {
                    let tag = u.tags[i];
                    let ix = u.index[i] as usize;
                    match tag {
                        0 => {
                            // scalar
                            if u.scalars.validity[ix] {
                                // reuse scalar leaf by building a temporary Layout::Leaf view
                                match (&u.scalars.buffer, dtype) {
                                    (crate::layout::LeafBuffer::I32(v), _) => f(v[ix] as i64, 0, 0.0, NumKind::Int)?,
                                    (crate::layout::LeafBuffer::I64(v), _) => f(v[ix] as i64, 0, 0.0, NumKind::Int)?,
                                    (crate::layout::LeafBuffer::U32(v), _) => f(0, v[ix] as u64, 0.0, NumKind::UInt)?,
                                    (crate::layout::LeafBuffer::U64(v), _) => f(0, v[ix] as u64, 0.0, NumKind::UInt)?,
                                    (crate::layout::LeafBuffer::F32(v), _) => f(0, 0, v[ix] as f64, NumKind::Float)?,
                                    (crate::layout::LeafBuffer::F64(v), _) => f(0, 0, v[ix], NumKind::Float)?,
                                    _ => return Err(PyValueError::new_err("Unsupported dtype for DataFrame reduction.")),
                                }
                            }
                        }
                        1 => {
                            // list
                            let start = u.lists.offsets[ix] as usize;
                            let end = u.lists.offsets[ix + 1] as usize;
                            let sub = take_range(u.lists.content.as_ref(), start, end)?;
                            walk(&sub, dtype, f)?;
                        }
                        _ => return Err(PyValueError::new_err("Invalid union tag.")),
                    }
                }
                Ok(())
            }
        }
    }
    walk(&arr.layout, arr.dtype, f)
}

pub fn parse_row_index(py: Python<'_>, idx: &Bound<'_, PyAny>, n: usize) -> PyResult<Arc<Vec<i64>>> {
    // Accept int, slice, or boolean mask sequence.
    if let Ok(slc) = idx.downcast::<PySlice>() {
        let indices = slc.call_method1("indices", (n as i64,))?;
        let t = indices.downcast::<PyTuple>()?;
        let start = t.get_item(0)?.extract::<i64>()?;
        let stop = t.get_item(1)?.extract::<i64>()?;
        let step = t.get_item(2)?.extract::<i64>()?;
        if step == 0 {
            return Err(PyValueError::new_err("slice step cannot be zero."));
        }
        let mut out: Vec<i64> = Vec::new();
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
        return Ok(Arc::new(out));
    }
    if idx.is_instance_of::<pyo3::types::PyInt>() {
        let mut i = idx.extract::<i64>()?;
        if i < 0 {
            i += n as i64;
        }
        if i < 0 || i >= n as i64 {
            return Err(PyValueError::new_err("Index out of bounds."));
        }
        return Ok(Arc::new(vec![i]));
    }
    if crate::dtype::is_sequence_like(py, idx)? {
        let seq = idx.downcast::<PySequence>()?;
        let m = seq.len()? as usize;
        let mut all_bool = true;
        let mut mask: Vec<bool> = Vec::with_capacity(m);
        for i in 0..m {
            let it = seq.get_item(i)?;
            if it.is_instance_of::<pyo3::types::PyBool>() {
                mask.push(it.extract::<bool>()?);
            } else {
                all_bool = false;
                break;
            }
        }
        if all_bool {
            if m != n {
                return Err(PyValueError::new_err(
                    "Boolean indexing requires mask length to match dataframe length.",
                ));
            }
            let mut out: Vec<i64> = Vec::new();
            for (i, b) in mask.iter().enumerate() {
                if *b {
                    out.push(i as i64);
                }
            }
            return Ok(Arc::new(out));
        }
    }
    Err(PyValueError::new_err(
        "Row index must be int, slice, or boolean mask.",
    ))
}


