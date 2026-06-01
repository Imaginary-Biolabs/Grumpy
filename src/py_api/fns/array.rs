use crate::dtype::{infer_dtype, inferclass_to_dtype, DType, PyDType};
use crate::error::{arg_invalid, dtype_mismatch, unsupported};
use crate::layout_ops::concat_arrays;
use crate::layout::{build_array, concat_to_py_list, fill_layout_like, GrumpyArray};
use crate::py_api::types::PyGrumpyArray;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn array(
    py: Python<'_>,
    obj: Bound<'_, pyo3::types::PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let dt = if let Some(d) = dtype {
        d.dt
    } else {
        let inf = infer_dtype(py, &obj)?;
        let cls = inf.ok_or_else(|| {
            arg_invalid(
                "dtype",
                "cannot infer dtype from all-null input",
                "pass dtype= explicitly when constructing from all-null data.",
            )
        })?;
        inferclass_to_dtype(cls)
    };
    let inner = build_array(py, &obj, dt)?;
    Ok(PyGrumpyArray { inner })
}
#[pyfunction]
#[pyo3(signature = (arrays, dim=0))]
pub fn cat(py: Python<'_>, arrays: Vec<PyRef<'_, PyGrumpyArray>>, dim: isize) -> PyResult<PyGrumpyArray> {
    if arrays.is_empty() {
        return Err(arg_invalid(
            "arrays",
            "cat() requires at least one array",
            "pass one or more Grumpy arrays to concatenate.",
        ));
    }
    let dim_u = if dim < 0 {
        return Err(unsupported(
            "cat",
            "negative dim is not supported yet",
            "use a non-negative dim (0 for outermost axis).",
        ));
    } else {
        dim as usize
    };
    let dtype = arrays[0].inner.dtype;
    for a in &arrays[1..] {
        if a.inner.dtype != dtype {
            return Err(dtype_mismatch(dtype, a.inner.dtype, "in cat() inputs"));
        }
    }
    let rust_arrays: Vec<GrumpyArray> = arrays.into_iter().map(|a| a.inner.clone()).collect();
    let merged = if dim_u > 0 && rust_arrays.iter().any(|a| a.layout.has_union()) {
        let merged_list = concat_to_py_list(py, &rust_arrays, dim_u)?;
        build_array(py, &merged_list.bind(py), dtype)?
    } else {
        concat_arrays(&rust_arrays, dim_u)?
    };
    Ok(PyGrumpyArray { inner: merged })
}

#[pyfunction]
#[pyo3(signature = (x, fill_value, dtype=None))]
pub fn full_like(
    py: Python<'_>,
    x: PyRef<'_, PyGrumpyArray>,
    fill_value: Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let dt = dtype.map(|d| d.dt).unwrap_or(x.inner.dtype);
    let layout = fill_layout_like(py, &x.inner.layout, dt, &fill_value)?;
    Ok(PyGrumpyArray {
        inner: GrumpyArray { dtype: dt, layout },
    })
}

#[pyfunction]
#[pyo3(signature = (x, dtype=None))]
pub fn zeros_like(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>, dtype: Option<PyDType>) -> PyResult<PyGrumpyArray> {
    let dt = dtype.map(|d| d.dt).unwrap_or(x.inner.dtype);
    let fill = match dt {
        DType::Bool => false.into_py(py),
        DType::Char => {
            return Err(unsupported(
                "zeros_like",
                "dtype=char is not supported",
                "use a numeric dtype or build a char array with gr.array(...).",
            ))
        }
        _ => 0.into_py(py),
    };
    full_like(py, x, fill.into_bound(py), Some(PyDType { dt }))
}

#[pyfunction]
#[pyo3(signature = (x, dtype=None))]
pub fn ones_like(py: Python<'_>, x: PyRef<'_, PyGrumpyArray>, dtype: Option<PyDType>) -> PyResult<PyGrumpyArray> {
    let dt = dtype.map(|d| d.dt).unwrap_or(x.inner.dtype);
    let fill = match dt {
        DType::Bool => true.into_py(py),
        DType::Char => {
            return Err(unsupported(
                "ones_like",
                "dtype=char is not supported",
                "use a numeric dtype or build a char array with gr.array(...).",
            ))
        }
        _ => 1.into_py(py),
    };
    full_like(py, x, fill.into_bound(py), Some(PyDType { dt }))
}
