use crate::dataframe as df_ops;
use crate::dtype::{inferclass_to_dtype, InferClass};
use crate::layout::{build_array, GrumpyArray};
use crate::py_api::types::{PyGrumpyArray, PyGrumpyDataFrame};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (mapping, schema=None))]
pub fn dataframe(py: Python<'_>, mapping: Bound<'_, PyAny>, schema: Option<Bound<'_, PyAny>>) -> PyResult<PyGrumpyDataFrame> {
    let d = mapping
        .downcast::<pyo3::types::PyDict>()
        .map_err(|_| PyValueError::new_err("dataframe(mapping, ...) requires a dict."))?;
    let sch = if let Some(s) = schema {
        Some(df_ops::Schema::parse(py, &s)?)
    } else {
        None
    };
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<GrumpyArray> = Vec::new();
    for (k, v) in d.iter() {
        let name = k.extract::<String>().map_err(|_| PyValueError::new_err("dataframe keys must be strings."))?;
        // If already a GrumpyArray, clone it; else build.
        let arr = if let Ok(g) = v.extract::<PyRef<'_, PyGrumpyArray>>() {
            g.inner.clone()
        } else {
            // dtype inference: use infer_dtype then build_array
            let inferred = crate::dtype::infer_dtype(py, &v)?.unwrap_or(crate::dtype::InferClass::Float);
            let dt = crate::dtype::inferclass_to_dtype(inferred);
            crate::layout::build_array(py, &v, dt)?
        };
        names.push(name);
        cols.push(arr);
    }
    let df = df_ops::GrumpyDataFrame::new(names, cols, sch)?;
    Ok(PyGrumpyDataFrame { inner: df })
}
