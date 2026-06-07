//! Lazy on-disk dataframe handle (:class:`OpenDataFrame`) and column proxies.

use crate::dataframe::{entity_count_from_canon, resolve_shape_dim, GrumpyDataFrame};
use crate::dataframe_indexing::{
    accessor_target_level, dataframe_getitem, parse_dataframe_index_key, parse_level_selector,
    selector_indices, LevelSelector,
};
use crate::error::{arg_invalid, io_wrong_type, schema_violation, unknown_column};
use crate::io::{
    self as io_ops, canon_from_handle, column_names_from_handle, load_column_axis0_slice,
    load_dataframe_columns_axis0_slice, schema_from_handle, DatasetHandle, OpenSession,
};
use crate::io_cache::IoCachePolicy;
use crate::layout::{drop_layout_axes, leaf_view, GrumpyArray, Layout};
use crate::py_api::convert::{shape_or_nshape, wrap_result};
use crate::py_api::indexing::{fast_getitem, getitem_array_indexing, getitem_coordinate};
use crate::py_api::types::{PyGrumpyArray, PyGrumpyDataFrame, PyOpenColumn, PyOpenDataFrame};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn is_column_key(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(s) = key.extract::<String>() {
        let _ = s;
        return Ok(true);
    }
    if let Ok(tup) = key.downcast::<PyTuple>() {
        if tup.is_empty() {
            return Ok(false);
        }
        let mut all_str = true;
        for i in 0..tup.len() {
            if tup.get_item(i)?.extract::<String>().is_err() {
                all_str = false;
                break;
            }
        }
        return Ok(all_str);
    }
    let _ = py;
    Ok(false)
}

fn axis0_load_range(indices: &[i64], n: usize) -> PyResult<(usize, usize)> {
    if indices.is_empty() {
        return Err(arg_invalid(
            "index",
            "empty index sequence",
            "pass at least one index at this schema level.",
        ));
    }
    let mut min = n;
    let mut max = 0usize;
    for &raw in indices {
        let mut j = raw;
        if j < 0 {
            j += n as i64;
        }
        if j < 0 || j >= n as i64 {
            return Err(crate::error::index_out_of_bounds(j as usize, n, "on open dataframe index"));
        }
        let i = j as usize;
        min = min.min(i);
        max = max.max(i);
    }
    Ok((min, max + 1))
}

fn adjust_selector(sel: LevelSelector, base: usize) -> PyResult<LevelSelector> {
    if base == 0 {
        return Ok(sel);
    }
    match sel {
        LevelSelector::One(i) => Ok(LevelSelector::One(i - base as i64)),
        LevelSelector::Fancy(idxs) => Ok(LevelSelector::Fancy(
            idxs.into_iter().map(|i| i - base as i64).collect(),
        )),
        LevelSelector::BoolMask(mask) => Ok(LevelSelector::BoolMask(mask)),
        LevelSelector::Slice { start, stop, step } => Ok(LevelSelector::Slice {
            start: start - base as i64,
            stop: stop - base as i64,
            step,
        }),
    }
}

fn materialize_open_index(
    open: &PyOpenDataFrame,
    sel: LevelSelector,
) -> PyResult<GrumpyDataFrame> {
    let n = entity_count_from_canon(&open.canon, open.index_depth);
    let indices = selector_indices(&sel)?;
    let (start, stop) = axis0_load_range(&indices, n)?;
    let colnames = open.column_names.as_deref();
    let handle = open.session.handle()?;
    let mut df = load_dataframe_columns_axis0_slice(handle, colnames, start, stop)?;
    let adj = adjust_selector(sel, start)?;
    dataframe_getitem(&df, adj)
}

fn shell_for_shape(open: &PyOpenDataFrame) -> PyResult<GrumpyDataFrame> {
    let names = if let Some(cols) = &open.column_names {
        cols.clone()
    } else {
        column_names_from_handle(open.session.handle()?)?
    };
    GrumpyDataFrame::from_loaded(
        names,
        Vec::new(),
        open.schema.clone(),
        Some(open.canon.clone()),
    )
}

#[pyfunction]
#[pyo3(signature = (path, cache="chunks", chunk_budget_mb=256))]
pub fn open_dataset(path: String, cache: &str, chunk_budget_mb: usize) -> PyResult<PyOpenDataFrame> {
    let budget_bytes = chunk_budget_mb.saturating_mul(1024 * 1024).max(1);
    let policy = IoCachePolicy::parse(cache, budget_bytes)?;
    let handle = DatasetHandle::open_lazy(&path, policy)?;
    match &handle.meta.root {
        io_ops::RootMeta::DataFrame { .. } => {
            let schema = schema_from_handle(&handle)?;
            let canon = canon_from_handle(&handle)?;
            Ok(PyOpenDataFrame {
                session: OpenSession::new(handle),
                column_names: None,
                schema,
                canon,
                index_depth: 0,
            })
        }
        _ => Err(io_wrong_type("dataframe", &path)),
    }
}

#[pymethods]
impl PyOpenDataFrame {
    fn __repr__(&self) -> String {
        let state = if self.session.is_closed() { ", closed" } else { "" };
        format!("grumpy.OpenDataFrame('{}'{state})", self.session.path())
    }

    fn __len__(&self) -> usize {
        entity_count_from_canon(&self.canon, self.index_depth)
    }

    #[getter]
    fn closed(&self) -> bool {
        self.session.is_closed()
    }

    fn close(&mut self) {
        self.session.close();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.session.close();
        Ok(())
    }

    fn load(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.session.handle()?;
        let df = io_ops::load_dataframe(py, self.session.path())?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn shape(&self, py: Python<'_>, dim: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let shell = shell_for_shape(self)?;
        let d = resolve_shape_dim(shell.schema.as_ref(), shell.index_depth, &dim)?;
        let proxy = shell.shape_proxy_array()?;
        shape_or_nshape(py, &proxy, d, false)
    }

    fn nshape(&self, py: Python<'_>, dim: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let shell = shell_for_shape(self)?;
        let d = resolve_shape_dim(shell.schema.as_ref(), shell.index_depth, &dim)?;
        let proxy = shell.shape_proxy_array()?;
        shape_or_nshape(py, &proxy, d, true)
    }

    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        if is_column_key(py, &key)? {
            if let Ok(s) = key.extract::<String>() {
                return Ok(Py::new(
                    py,
                    PyOpenColumn {
                        session: self.session.clone(),
                        column_name: s,
                        column_names: Some(vec![key.extract()?]),
                        schema: self.schema.clone(),
                        canon: self.canon.clone(),
                        index_depth: self.index_depth,
                        drop_axes: 0,
                        flatten_to_leaf: false,
                    },
                )?
                .into_py(py));
            }
            if let Ok(tup) = key.downcast::<PyTuple>() {
                let mut names: Vec<String> = Vec::new();
                for i in 0..tup.len() {
                    names.push(tup.get_item(i)?.extract::<String>().map_err(|_| {
                        arg_invalid(
                            "key",
                            "column selection must be strings",
                            "pass column names as str or tuple[str, ...].",
                        )
                    })?);
                }
                let first = names[0].clone();
                return Ok(Py::new(
                    py,
                    PyOpenColumn {
                        session: self.session.clone(),
                        column_name: first,
                        column_names: Some(names),
                        schema: self.schema.clone(),
                        canon: self.canon.clone(),
                        index_depth: self.index_depth,
                        drop_axes: 0,
                        flatten_to_leaf: false,
                    },
                )?
                .into_py(py));
            }
        }

        let shell = shell_for_shape(self)?;
        let sel = parse_dataframe_index_key(py, &key, &shell)?;
        let df = materialize_open_index(self, sel)?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn __getattr__(slf: PyRef<'_, Self>, py: Python<'_>, name: String) -> PyResult<PyObject> {
        if let Some(schema) = slf.schema.clone() {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let parent: Py<PyOpenDataFrame> = Py::from(slf);
                    let level = accessor_target_level(&schema, &[name.clone()])?;
                    let acc = PyOpenDataFrameAccessor {
                        parent,
                        path: vec![name],
                        index_level: level,
                    };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        for n in column_names_from_handle(slf.session.handle()?)? {
            if n == name {
                return Ok(Py::new(
                    py,
                    PyOpenColumn {
                        session: slf.session.clone(),
                        column_name: name.clone(),
                        column_names: slf.column_names.clone(),
                        schema: slf.schema.clone(),
                        canon: slf.canon.clone(),
                        index_depth: slf.index_depth,
                        drop_axes: 0,
                        flatten_to_leaf: true,
                    },
                )?
                .into_py(py));
            }
        }
        Err(unknown_column(&name))
    }
}

#[pyclass(name = "OpenDataFrameAccessor")]
pub struct PyOpenDataFrameAccessor {
    pub(crate) parent: Py<PyOpenDataFrame>,
    pub(crate) path: Vec<String>,
    pub(crate) index_level: usize,
}

#[pymethods]
impl PyOpenDataFrameAccessor {
    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let open = self.parent.borrow(py);
        if open.index_depth != self.index_level {
            return Err(schema_violation(
                format!(
                    "accessor level {} does not match open index_depth {}",
                    self.index_level, open.index_depth
                ),
                "narrow outer schema levels before indexing deeper levels.",
                "use open.scene[i].molecule[j] instead of skipping levels.",
            ));
        }
        let shell = shell_for_shape(&open)?;
        let n = entity_count_from_canon(&shell.canon, shell.index_depth);
        let sel = parse_level_selector(py, &key, n)?;
        let df = materialize_open_index(&open, sel)?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn __getattr__(&self, py: Python<'_>, name: String) -> PyResult<PyObject> {
        let open = self.parent.borrow(py);
        if let Some(schema) = &open.schema {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let mut p = self.path.clone();
                    p.push(name.clone());
                    let level = accessor_target_level(schema, &p)?;
                    let acc = PyOpenDataFrameAccessor {
                        parent: self.parent.clone_ref(py),
                        path: p,
                        index_level: level,
                    };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        let schema = open.schema.as_ref().ok_or_else(|| {
            schema_violation(
                "dot-notation requires a schema",
                "the dataset was saved without schema=.",
                "pass schema= when saving to enable open.<level>.<col> access.",
            )
        })?;
        let level0 = *schema.name_to_level.get(&self.path[0]).ok_or_else(|| {
            schema_violation(
                "invalid schema path",
                format!("'{}' is not a declared schema level.", self.path[0]),
                "use a top-level name from schema= when accessing open.<level>.<col>.",
            )
        })?;
        let col_names = if let Some(cols) = &open.column_names {
            cols.clone()
        } else {
            column_names_from_handle(open.session.handle()?)?
        };
        if !col_names.iter().any(|n| n == &name) {
            return Err(unknown_column(&name));
        }
        let out = PyOpenColumn {
            session: open.session.clone(),
            column_name: name,
            column_names: open.column_names.clone(),
            schema: open.schema.clone(),
            canon: open.canon.clone(),
            index_depth: open.index_depth,
            drop_axes: level0,
            flatten_to_leaf: false,
        };
        Ok(Py::new(py, out)?.into_py(py))
    }
}

#[pymethods]
impl PyOpenColumn {
    fn __repr__(&self) -> String {
        format!(
            "grumpy.OpenColumn('{}', column='{}')",
            self.session.path(),
            self.column_name
        )
    }

    fn __len__(&self) -> usize {
        entity_count_from_canon(&self.canon, self.index_depth)
    }

    fn shape(&self, py: Python<'_>, dim: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let shell = GrumpyDataFrame::from_loaded(
            vec![self.column_name.clone()],
            Vec::new(),
            self.schema.clone(),
            Some(self.canon.clone()),
        )?;
        let d = resolve_shape_dim(shell.schema.as_ref(), shell.index_depth, &dim)?;
        let proxy = shell.shape_proxy_array()?;
        shape_or_nshape(py, &proxy, d, false)
    }

    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let handle = self.session.handle()?;
        let n = handle.axis0_len().unwrap_or(0);
        let arr = load_column_axis0_slice(handle, &self.column_name, 0, n)?;
        let layout = if self.drop_axes > 0 {
            drop_layout_axes(&arr.layout, self.drop_axes)?
        } else if self.flatten_to_leaf {
            Layout::Leaf(leaf_view(&arr.layout, arr.dtype)?)
        } else {
            arr.layout.clone()
        };
        let col = GrumpyArray {
            dtype: arr.dtype,
            layout,
        };
        if let Some(out) = fast_getitem(py, &col, &key)? {
            return Ok(out);
        }
        let py_arr = PyGrumpyArray { inner: col };
        let base = py_arr.inner.to_py_list(py)?;
        let out = if key.downcast::<PyTuple>().is_ok() {
            getitem_coordinate(py, &base.bind(py), &key, py_arr.inner.dtype)?
        } else if crate::dtype::is_sequence_like(py, &key)? {
            getitem_array_indexing(py, &base.bind(py), &key, py_arr.inner.dtype)?
        } else {
            getitem_coordinate(py, &base.bind(py), &key, py_arr.inner.dtype)?
        };
        wrap_result(py, out, py_arr.inner.dtype)
    }
}
