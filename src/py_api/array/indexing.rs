use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::layout::build_array;
use crate::py_api::convert::wrap_result;
use crate::py_api::indexing::{fast_getitem, fast_setitem, getitem_array_indexing, getitem_coordinate, setitem_array, setitem_coordinate};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pymethods]
impl PyGrumpyArray {
    fn __getitem__(&self, py: Python<'_>, index: Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Some(out) = fast_getitem(py, &self.inner, &index)? {
            return Ok(out);
        }

        // Fallback (correctness): Python-list based indexing.
        let base = self.inner.to_py_list(py)?;
        let out = if index.downcast::<PyTuple>().is_ok() {
            getitem_coordinate(py, &base.bind(py), &index, self.inner.dtype)?
        } else if crate::dtype::is_sequence_like(py, &index)? {
            getitem_array_indexing(py, &base.bind(py), &index, self.inner.dtype)?
        } else {
            // scalar int or slice = coordinate indexing on dim 0
            getitem_coordinate(py, &base.bind(py), &index, self.inner.dtype)?
        };
        wrap_result(py, out, self.inner.dtype)
    }

    fn __setitem__(&mut self, py: Python<'_>, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if fast_setitem(py, &mut self.inner, &index, &value)? {
            return Ok(());
        }

        // Fallback (correctness): mutate Python list and rebuild.
        let base = self.inner.to_py_list(py)?;
        let base_b = base.bind(py);
        if index.downcast::<PyTuple>().is_ok() {
            setitem_coordinate(py, &base_b, &index, &value)?;
        } else if crate::dtype::is_sequence_like(py, &index)? {
            setitem_array(py, &base_b, &index, &value)?;
        } else {
            setitem_coordinate(py, &base_b, &index, &value)?;
        }
        let rebuilt = build_array(py, &base_b, self.inner.dtype)?;
        self.inner = rebuilt;
        Ok(())
    }
}
