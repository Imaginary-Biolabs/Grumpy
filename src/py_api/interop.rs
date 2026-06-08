//! Fast rectangular tensor interop: NumPy, PyTorch, TensorFlow.

use crate::dtype::{DType, PyDType};
use crate::error::{arg_invalid, unsupported};
use crate::layout::{GrumpyArray, Layout};
use crate::py_api::types::PyGrumpyArray;
use crate::rect_array::{
    export_rectangular, layout_from_shape_and_bytes, numpy_dtype_to_grumpy, rectangular_shape,
    RectExport, RectTensor,
};
use numpy::{
    PyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods, IxDyn,
};
use pyo3::exceptions::{PyImportError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAnyMethods};

fn rect_to_numpy<'py>(
    py: Python<'py>,
    arr: &GrumpyArray,
    owner: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match export_rectangular(&arr.layout)? {
        RectExport::View(view) => numpy_from_view(py, owner, view),
        RectExport::Gathered {
            shape,
            dtype,
            bytes,
        } => numpy_from_bytes(py, &shape, dtype, &bytes),
    }
}

fn numpy_from_view<'py>(
    py: Python<'py>,
    owner: Option<Bound<'py, PyAny>>,
    view: RectTensor<'_>,
) -> PyResult<Bound<'py, PyAny>> {
    let elem_size = view.dtype.size_bytes();
    let byte_start = view.start * elem_size;
    let byte_end = byte_start + view.count * elem_size;
    let bytes = &view.leaf.buffer.as_bytes()[byte_start..byte_end];

    if let Some(owner) = owner {
        if let Ok(arr) = try_numpy_borrowed(py, owner, view.dtype, &view.shape, bytes) {
            return Ok(arr);
        }
    }
    numpy_from_bytes(py, &view.shape, view.dtype, bytes)
}

fn try_numpy_borrowed<'py>(
    py: Python<'py>,
    owner: Bound<'py, PyAny>,
    dtype: DType,
    shape: &[usize],
    bytes: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! borrow {
        ($t:ty) => {{
            let elem_size = std::mem::size_of::<$t>();
            let n = bytes.len() / elem_size;
            let slice: &[$t] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const $t, n)
            };
            let view = numpy::ndarray::ArrayView::from_shape(IxDyn(shape), slice).map_err(|e| {
                arg_invalid(
                    "array",
                    format!("failed to build ndarray view: {e}"),
                    "ensure the layout is a dense C-contiguous rectangular tensor.",
                )
            })?;
            let arr = unsafe {
                PyArray::<$t, _>::borrow_from_array_bound(&view, owner)
            };
            Ok(arr.into_any())
        }};
    }
    match dtype {
        DType::Int8 => borrow!(i8),
        DType::Int16 => borrow!(i16),
        DType::Int32 => borrow!(i32),
        DType::Int64 => borrow!(i64),
        DType::UInt8 => borrow!(u8),
        DType::UInt16 => borrow!(u16),
        DType::UInt32 => borrow!(u32),
        DType::UInt64 => borrow!(u64),
        DType::Float32 => borrow!(f32),
        DType::Float64 => borrow!(f64),
        DType::Bool => borrow!(u8),
        DType::Float16 => Err(arg_invalid(
            "dtype",
            "float16 zero-copy export is not supported yet",
            "cast to float32 before export, or use numpy_from_bytes.",
        )),
        DType::Char | DType::String => Err(arg_invalid(
            "dtype",
            "char/string tensors are not supported",
            "use a numeric or bool dtype.",
        )),
    }
}

fn numpy_from_bytes<'py>(
    py: Python<'py>,
    shape: &[usize],
    dtype: DType,
    bytes: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! export {
        ($t:ty) => {{
            let elem_size = std::mem::size_of::<$t>();
            let n = bytes.len() / elem_size;
            let slice: &[$t] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const $t, n)
            };
            let arr = PyArray::<$t, _>::zeros_bound(py, IxDyn(shape), false);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    arr.data(),
                    n,
                );
            }
            Ok(arr.into_any())
        }};
    }
    match dtype {
        DType::Int8 => export!(i8),
        DType::Int16 => export!(i16),
        DType::Int32 => export!(i32),
        DType::Int64 => export!(i64),
        DType::UInt8 => export!(u8),
        DType::UInt16 => export!(u16),
        DType::UInt32 => export!(u32),
        DType::UInt64 => export!(u64),
        DType::Float32 => export!(f32),
        DType::Float64 => export!(f64),
        DType::Bool => export!(u8),
        DType::Float16 => {
            let n = bytes.len() / 2;
            let arr = PyArray::<u16, _>::zeros_bound(py, IxDyn(shape), false);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), arr.data() as *mut u8, n * 2);
            }
            let np = PyModule::import_bound(py, "numpy")?;
            let kwargs = [("dtype", np.getattr("float16")?)].into_py_dict_bound(py);
            Ok(arr
                .into_any()
                .call_method("astype", (), Some(&kwargs))?)
        }
        DType::Char | DType::String => Err(arg_invalid(
            "dtype",
            "char/string tensors are not supported",
            "use a numeric or bool dtype.",
        )),
    }
}

fn import_framework_module<'py>(
    py: Python<'py>,
    name: &str,
    pip_pkg: &str,
) -> PyResult<Bound<'py, PyModule>> {
    match PyModule::import_bound(py, name) {
        Ok(m) => Ok(m),
        Err(e) => {
            if e.is_instance_of::<PyImportError>(py) {
                Err(unsupported(
                    "interop",
                    format!("{name} is not installed"),
                    format!("install with `pip install {pip_pkg}` and retry."),
                ))
            } else {
                Err(e)
            }
        }
    }
}

#[pymethods]
impl PyGrumpyArray {
    /// Export this array as a C-contiguous NumPy ``ndarray``.
    ///
    /// Raises if the layout is ragged, contains nulls/unions, or uses an unsupported dtype.
    pub fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let owner = Py::new(py, PyGrumpyArray {
            inner: self.inner.clone(),
        })?
        .into_bound(py)
        .into_any();
        let arr = rect_to_numpy(py, &self.inner, Some(owner))?;
        Ok(arr.into_py(py))
    }

    /// Export as a PyTorch tensor via ``torch.from_numpy`` (zero-copy when the NumPy view allows).
    pub fn to_torch(&self, py: Python<'_>) -> PyResult<PyObject> {
        let np_arr = self.to_numpy(py)?;
        let torch = import_framework_module(py, "torch", "torch")?;
        let tensor = torch.call_method1("from_numpy", (np_arr,))?;
        Ok(tensor.into_py(py))
    }

    /// Export as a TensorFlow tensor via ``tf.convert_to_tensor`` on the NumPy buffer.
    pub fn to_tensorflow(&self, py: Python<'_>) -> PyResult<PyObject> {
        let np_arr = self.to_numpy(py)?;
        let tf = import_framework_module(py, "tensorflow", "tensorflow")?;
        let tensor = tf.call_method1("convert_to_tensor", (np_arr,))?;
        Ok(tensor.into_py(py))
    }
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn from_numpy(
    py: Python<'_>,
    obj: Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let arr = obj
        .downcast::<PyUntypedArray>()
        .map_err(|_| {
            PyTypeError::new_err("from_numpy() expects a NumPy ndarray (C-contiguous).")
        })?;
    if !arr.is_c_contiguous() {
        return Err(arg_invalid(
            "obj",
            "NumPy array must be C-contiguous",
            "call numpy.ascontiguousarray(obj) before from_numpy().",
        ));
    }
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    if shape.is_empty() {
        return Err(arg_invalid(
            "obj",
            "scalar NumPy arrays are not supported",
            "pass a 1D or higher-dimensional array.",
        ));
    }
    let dt = if let Some(d) = dtype {
        d.dt
    } else {
        numpy_dtype_to_grumpy(py, &arr.dtype())?
    };
    let layout = layout_from_typed_numpy(&obj, &shape, dt)?;
    Ok(PyGrumpyArray {
        inner: GrumpyArray { dtype: dt, layout },
    })
}

fn layout_from_typed_numpy(
    obj: &Bound<'_, PyAny>,
    shape: &[usize],
    dt: DType,
) -> PyResult<Layout> {
    macro_rules! import {
        ($t:ty) => {{
            let ro: PyReadonlyArrayDyn<'_, $t> = obj.extract()?;
            let slice = ro.as_slice()?;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const u8,
                    slice.len() * std::mem::size_of::<$t>(),
                )
            };
            layout_from_shape_and_bytes(shape, dt, bytes)
        }};
    }
    match dt {
        DType::Int8 => import!(i8),
        DType::Int16 => import!(i16),
        DType::Int32 => import!(i32),
        DType::Int64 => import!(i64),
        DType::UInt8 => import!(u8),
        DType::UInt16 => import!(u16),
        DType::UInt32 => import!(u32),
        DType::UInt64 => import!(u64),
        DType::Float16 => import!(u16),
        DType::Float32 => import!(f32),
        DType::Float64 => import!(f64),
        DType::Bool => import!(bool),
        DType::Char | DType::String => Err(arg_invalid(
            "dtype",
            "char/string tensors are not supported",
            "use a numeric or bool dtype.",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn from_torch(
    py: Python<'_>,
    obj: Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let np = tensor_to_numpy(py, &obj, "torch.Tensor")?;
    from_numpy(py, np, dtype)
}

#[pyfunction]
#[pyo3(signature = (obj, dtype=None))]
pub fn from_tensorflow(
    py: Python<'_>,
    obj: Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyGrumpyArray> {
    let np = tensor_to_numpy(py, &obj, "tf.Tensor")?;
    from_numpy(py, np, dtype)
}

fn tensor_to_numpy<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    expected: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(np) = obj.call_method0("numpy") {
        return Ok(np);
    }
    if let Ok(detached) = obj.call_method0("detach") {
        if let Ok(cpu) = detached.call_method0("cpu") {
            if let Ok(np) = cpu.call_method0("numpy") {
                return Ok(np);
            }
        }
    }
    if let Ok(np) = obj.call_method0("cpu") {
        if let Ok(np) = np.call_method0("numpy") {
            return Ok(np);
        }
    }
    Err(PyTypeError::new_err(format!(
        "expected a {expected} with a .numpy() view; got {}",
        obj.get_type().name()?
    )))
}

/// Return ``True`` when ``arr`` can be exported as a dense rectangular tensor.
#[pyfunction]
pub fn is_rectangular(arr: PyRef<'_, PyGrumpyArray>) -> bool {
    rectangular_shape(&arr.inner.layout).is_ok()
}
