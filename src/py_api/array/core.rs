use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::layout::Layout;
use crate::py_api::convert::{layout_all_valid_no_union, leaf_to_numpy_1d_typed, numpy_dtype, shape_or_nshape};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

#[pymethods]
impl PyGrumpyArray {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let lst = self.inner.to_py_list(py)?;
        Ok(format!("GrumpyArray(dtype={}, data={})", self.inner.dtype, lst))
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType { dt: self.inner.dtype }
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.inner.to_py_list(py)
    }

    fn astype(&self, dtype: PyDType, casting: Option<&str>) -> PyResult<Self> {
        let mode = crate::cast::CastMode::parse(casting.unwrap_or("safe"))?;
        let inner = crate::cast::cast_array_with_mode(&self.inner, dtype.dt, mode)?;
        Ok(Self { inner })
    }

    fn shape(&self, py: Python<'_>, dim: usize) -> PyResult<PyObject> {
        shape_or_nshape(py, &self.inner, dim, false)
    }

    fn nshape(&self, py: Python<'_>, dim: usize) -> PyResult<PyObject> {
        shape_or_nshape(py, &self.inner, dim, true)
    }

    fn nanshape(&self, py: Python<'_>, dim: usize) -> PyResult<PyObject> {
        // Alias; for milestone-1 we treat nanshape == nshape (NaN is a value, not null).
        shape_or_nshape(py, &self.inner, dim, true)
    }

    fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Fast leaf (1D) export: create a typed NumPy array and memcpy from the contiguous leaf buffer.
        // This avoids materializing Python lists for common 1D results (e.g. fancy gather).
        if let Layout::Leaf(leaf) = &self.inner.layout {
            if !leaf.has_nulls {
                if let Some(arr) = leaf_to_numpy_1d_typed(py, leaf, self.inner.dtype)? {
                    return Ok(arr);
                }
            }
        }

        let np = PyModule::import_bound(py, "numpy")?;
        let lst = self.inner.to_py_list(py)?;

        // Typed path: if there are no unions and no nulls, ask NumPy to materialize a typed array.
        // (We still need to validate rectangularity; NumPy will error if the nesting is ragged.)
        if layout_all_valid_no_union(&self.inner.layout) {
            if let Some((dtype_obj, expected_name)) = numpy_dtype(&np, self.inner.dtype)? {
                let kwargs = [("dtype", dtype_obj)].into_py_dict_bound(py);
                if let Ok(arr) = np.call_method("array", (lst.clone_ref(py),), Some(&kwargs)) {
                    // Ensure NumPy actually produced the dtype we requested (and not object).
                    let dtype = arr.getattr("dtype")?;
                    let name: String = dtype.getattr("name")?.extract()?;
                    if name == expected_name {
                        return Ok(arr.into());
                    }
                }
            }
        }

        // Fallback: object array via numpy.array(py_list, dtype=object)
        let dtype_obj = np.getattr("object_")?;
        let kwargs = [("dtype", dtype_obj)].into_py_dict_bound(py);
        let arr = np.call_method("array", (lst,), Some(&kwargs))?;
        Ok(arr.into())
    }

    fn copy(&self) -> Self {
        let mut inner = self.inner.clone();
        inner.uniquify_buffers();
        Self { inner }
    }
}
