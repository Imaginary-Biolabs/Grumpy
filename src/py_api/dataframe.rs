use crate::dataframe::{resolve_shape_dim};
use crate::dataframe_indexing::{
    accessor_getitem, accessor_target_level, dataframe_getitem, parse_dataframe_index_key,
};
use crate::py_api::convert::shape_or_nshape;
use crate::dtype::inferclass_to_dtype;
use crate::error::{arg_invalid, schema_violation, unknown_column};
use crate::layout::{drop_layout_axes, leaf_view, GrumpyArray, Layout};
use std::sync::Arc;
use crate::py_api::types::{PyDataFrameAccessor, PyGrumpyArray, PyGrumpyDataFrame};
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

#[pymethods]
impl PyGrumpyDataFrame {
    fn __repr__(&self) -> String {
        format!("grumpy.dataframe({})", self.inner.names.join(", "))
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner.to_pydict(py)
    }

    fn max(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner.max_all(py)
    }

    fn shape(&self, py: Python<'_>, dim: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let d = resolve_shape_dim(self.inner.schema.as_ref(), self.inner.index_depth, &dim)?;
        let proxy = self.inner.shape_proxy_array()?;
        shape_or_nshape(py, &proxy, d, false)
    }

    fn nshape(&self, py: Python<'_>, dim: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let d = resolve_shape_dim(self.inner.schema.as_ref(), self.inner.index_depth, &dim)?;
        let proxy = self.inner.shape_proxy_array()?;
        shape_or_nshape(py, &proxy, d, true)
    }

    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        if is_column_key(py, &key)? {
            if let Ok(s) = key.extract::<String>() {
                let df = self.inner.column_subset(&[s])?;
                return Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py));
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
                let df = self.inner.column_subset(&names)?;
                return Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py));
            }
        }

        let sel = parse_dataframe_index_key(py, &key, &self.inner)?;
        let df = dataframe_getitem(&self.inner, sel)?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn __setitem__(&mut self, py: Python<'_>, key: String, value: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(arr) = value.extract::<PyRef<'_, PyGrumpyArray>>() {
            self.inner.set_column_array(key, arr.inner.clone())
        } else {
            let inferred = crate::dtype::infer_dtype(py, &value)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = inferclass_to_dtype(inferred);
            self.inner.set_column(py, key, &value, Some(dt))
        }
    }

    fn __getattr__(slf: PyRef<'_, Self>, py: Python<'_>, name: String) -> PyResult<PyObject> {
        if let Some(schema) = slf.inner.schema.clone() {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let parent: Py<PyGrumpyDataFrame> = Py::from(slf);
                    let level = accessor_target_level(&schema, &[name.clone()])?;
                    let acc = PyDataFrameAccessor {
                        parent,
                        path: vec![name],
                        index_level: level,
                    };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        for (n, c) in slf.inner.names.iter().zip(slf.inner.cols.iter()) {
            if n == &name {
                let leaf = leaf_view(&c.layout, c.dtype)?;
                let out = PyGrumpyArray {
                    inner: GrumpyArray {
                        dtype: c.dtype,
                        layout: Layout::Leaf(leaf),
                    },
                };
                return Ok(Py::new(py, out)?.into_py(py));
            }
        }
        Err(unknown_column(&name))
    }
}

#[pymethods]
impl PyDataFrameAccessor {
    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let df_ref = self.parent.borrow(py);
        let df = accessor_getitem(py, &df_ref.inner, self.index_level, &key)?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn __getattr__(&self, py: Python<'_>, name: String) -> PyResult<PyObject> {
        let df_ref = self.parent.borrow(py);

        if let Some(schema) = &df_ref.inner.schema {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let mut p = self.path.clone();
                    p.push(name.clone());
                    let level = accessor_target_level(schema, &p)?;
                    let acc = PyDataFrameAccessor {
                        parent: self.parent.clone_ref(py),
                        path: p,
                        index_level: level,
                    };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }

        let schema = df_ref.inner.schema.as_ref().ok_or_else(|| {
            schema_violation(
                "dot-notation requires a schema",
                "the dataframe was constructed without schema=.",
                "pass schema= when creating the dataframe to enable df.<level>.<col> access.",
            )
        })?;
        let level0 = *schema
            .name_to_level
            .get(&self.path[0])
            .ok_or_else(|| {
                schema_violation(
                    "invalid schema path",
                    format!("'{}' is not a declared schema level.", self.path[0]),
                    "use a top-level name from schema= when accessing df.<level>.<col>.",
                )
            })?;

        let mut col: Option<GrumpyArray> = None;
        for (n, c) in df_ref.inner.names.iter().zip(df_ref.inner.cols.iter()) {
            if n == &name {
                col = Some(c.clone());
                break;
            }
        }
        let col = col.ok_or_else(|| unknown_column(&name))?;

        let layout = drop_layout_axes(&col.layout, level0)?;
        let out = PyGrumpyArray {
            inner: GrumpyArray {
                dtype: col.dtype,
                layout,
            },
        };
        Ok(Py::new(py, out)?.into_py(py))
    }

    fn __setattr__(&mut self, py: Python<'_>, name: String, value: Bound<'_, PyAny>) -> PyResult<()> {
        if name == "parent" || name == "path" {
            return Err(arg_invalid(
                "attribute",
                "cannot set internal accessor fields",
                "assign to dataframe columns via df.<level>.<col> = value.",
            ));
        }
        let mut df_mut = self.parent.borrow_mut(py);
        let schema = df_mut.inner.schema.as_ref().ok_or_else(|| {
            schema_violation(
                "dot-notation requires a schema",
                "the dataframe was constructed without schema=.",
                "pass schema= when creating the dataframe.",
            )
        })?;
        let level0 = *schema
            .name_to_level
            .get(&self.path[0])
            .ok_or_else(|| {
                schema_violation(
                    "invalid schema path",
                    format!("'{}' is not a declared schema level.", self.path[0]),
                    "use a top-level name from schema= when assigning df.<level>.<col>.",
                )
            })?;
        let col_level = schema.level_for_column(&name)?;

        let arr = if let Ok(g) = value.extract::<PyRef<'_, PyGrumpyArray>>() {
            g.inner.clone()
        } else {
            let inferred = crate::dtype::infer_dtype(py, &value)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = inferclass_to_dtype(inferred);
            crate::layout::build_array(py, &value, dt)?
        };

        let arr2 = if level0 >= 1 && level0 == col_level {
            if arr.len() == df_mut.inner.nrows() {
                arr
            } else {
                let canon_off_level = df_mut
                    .inner
                    .canon
                    .offsets
                    .get(level0)
                    .and_then(|x| x.as_ref())
                    .ok_or_else(|| {
                        schema_violation(
                            format!(
                                "cannot re-nest: missing canonical offsets for schema level {level0} ('{}')",
                                self.path[0]
                            ),
                            "canonical offsets for this schema level were not recorded.",
                            "ensure all columns at this level share the same nested shape before assignment.",
                        )
                    })?;
                let total = *canon_off_level.last().unwrap() as usize;
                if arr.len() != total {
                    return Err(schema_violation(
                        format!(
                            "dot-notation assignment at '{}': RHS length mismatch",
                            self.path[0]
                        ),
                        format!(
                            "expected outer length {total} (total elements at that level) or {} (axis-0 length), but got {}.",
                            df_mut.inner.nrows(),
                            arr.len()
                        ),
                        "match the canonical nested shape or pass a column with axis-0 length equal to nrows.",
                    ));
                }
                let mut cur = arr.layout;
                for lev in (1..=level0).rev() {
                    let canon_off = df_mut
                        .inner
                        .canon
                        .offsets
                        .get(lev)
                        .and_then(|x| x.as_ref())
                        .ok_or_else(|| {
                            schema_violation(
                                format!("cannot re-nest: missing canonical offsets for schema level {lev}"),
                                "canonical offsets for this schema level were not recorded.",
                                "ensure all columns at this level share the same nested shape before assignment.",
                            )
                        })?
                        .clone();
                    cur = Layout::ListOffset(crate::layout::ListOffset {
                        offsets: Arc::new(canon_off),
                        content: Box::new(cur),
                    });
                }
                GrumpyArray {
                    dtype: arr.dtype,
                    layout: cur,
                }
            }
        } else {
            arr
        };

        df_mut.inner.set_column_array(name, arr2)
    }
}
