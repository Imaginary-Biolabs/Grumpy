use crate::dataframe as df_ops;
use crate::dtype::{inferclass_to_dtype, DType};
use crate::error::{arg_invalid, schema_violation, unknown_column};
use crate::layout::{drop_layout_axes, leaf_view, GrumpyArray, Layout};
use std::sync::Arc;
use crate::py_api::types::{PyDataFrameAccessor, PyGrumpyArray, PyGrumpyDataFrame};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

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

    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Column selection by string or tuple of strings.
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

        // Row selection: int/slice/bool mask.
        let idx = df_ops::parse_row_index(py, &key, self.inner.nrows())?;
        let df = self.inner.row_select_indexed(idx)?;
        Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
    }

    fn __setitem__(&mut self, py: Python<'_>, key: String, value: Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(arr) = value.extract::<PyRef<'_, PyGrumpyArray>>() {
            self.inner.set_column_array(key, arr.inner.clone())
        } else {
            // Infer dtype from value and build.
            let inferred = crate::dtype::infer_dtype(py, &value)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = crate::dtype::inferclass_to_dtype(inferred);
            self.inner.set_column(py, key, &value, Some(dt))
        }
    }

    fn __getattr__(slf: PyRef<'_, Self>, py: Python<'_>, name: String) -> PyResult<PyObject> {
        // If this is a schema level, return accessor.
        if let Some(schema) = &slf.inner.schema {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let parent: Py<PyGrumpyDataFrame> = slf.into_py(py).extract(py)?;
                    let acc = PyDataFrameAccessor { parent, path: vec![name] };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        // Otherwise, treat as column name and return fully flattened array (default dot-notation behavior).
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
    fn __getattr__(&self, py: Python<'_>, name: String) -> PyResult<PyObject> {
        let df_ref = self.parent.borrow(py);
        // Chain schema levels if applicable.
        if let Some(schema) = &df_ref.inner.schema {
            for names in &schema.levels {
                if names.iter().any(|n| n == &name) {
                    let mut p = self.path.clone();
                    p.push(name);
                    let acc = PyDataFrameAccessor { parent: self.parent.clone_ref(py), path: p };
                    return Ok(Py::new(py, acc)?.into_py(py));
                }
            }
        }
        // Column access under a path: currently require path to start at a valid schema level.
        let schema = df_ref.inner.schema.as_ref().ok_or_else(|| {
            schema_violation(
                "dot-notation requires a schema",
                "the dataframe was constructed without schema=.",
                "pass schema= when creating the dataframe to enable df.level.col access.",
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

        // Find column
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
        // Only handle setting columns; allow normal attribute sets for internal fields.
        if name == "parent" || name == "path" {
            return Err(arg_invalid(
                "name",
                "cannot overwrite accessor internals",
                "assign to column names only (e.g. df.level.col = …).",
            ));
        }
        let mut df_mut = self.parent.borrow_mut(py);
        let schema = df_mut.inner.schema.as_ref().ok_or_else(|| {
            schema_violation(
                "dot-notation assignment requires a schema",
                "the dataframe was constructed without schema=.",
                "pass schema= when creating the dataframe to enable df.level.col = … assignment.",
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

        // Build RHS array from Python value with inferred dtype.
        let arr = if let Ok(g) = value.extract::<PyRef<'_, PyGrumpyArray>>() {
            g.inner.clone()
        } else {
            let inferred = crate::dtype::infer_dtype(py, &value)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = inferclass_to_dtype(inferred);
            crate::layout::build_array(py, &value, dt)?
        };

        // If setting at schema level `level0` for a column belonging to the same schema level,
        // accept a flat-by-level RHS (outer len == total elements at that level) and re-nest it
        // back to axis-0 using canonical offsets at all intermediate levels.
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
                GrumpyArray { dtype: arr.dtype, layout: cur }
            }
        } else {
            arr
        };

        // Delegate to dataframe set column array (schema validation happens there).
        df_mut.inner.set_column_array(name, arr2)
    }
}
