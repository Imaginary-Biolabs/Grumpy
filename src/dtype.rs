use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Bool,
    Char,
    String,
}

impl DType {
    pub fn name(&self) -> &'static str {
        match self {
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Bool => "bool",
            DType::Char => "char",
            DType::String => "string",
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Int8 | DType::UInt8 | DType::Bool => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64 | DType::UInt64 | DType::Float64 => 8,
            DType::Char => 4,
            DType::String => 0,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[pyclass(name = "DType", frozen)]
#[derive(Clone)]
pub struct PyDType {
    pub(crate) dt: DType,
}

#[pymethods]
impl PyDType {
    fn __repr__(&self) -> String {
        format!("grumpy.{}", self.dt.name())
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.dt.name()
    }

    #[staticmethod]
    fn int8() -> Self {
        Self { dt: DType::Int8 }
    }
    #[staticmethod]
    fn int16() -> Self {
        Self { dt: DType::Int16 }
    }
    #[staticmethod]
    fn int32() -> Self {
        Self { dt: DType::Int32 }
    }
    #[staticmethod]
    fn int64() -> Self {
        Self { dt: DType::Int64 }
    }
    #[staticmethod]
    fn uint8() -> Self {
        Self { dt: DType::UInt8 }
    }
    #[staticmethod]
    fn uint16() -> Self {
        Self { dt: DType::UInt16 }
    }
    #[staticmethod]
    fn uint32() -> Self {
        Self { dt: DType::UInt32 }
    }
    #[staticmethod]
    fn uint64() -> Self {
        Self { dt: DType::UInt64 }
    }
    #[staticmethod]
    fn float16() -> Self {
        Self { dt: DType::Float16 }
    }
    #[staticmethod]
    fn float32() -> Self {
        Self { dt: DType::Float32 }
    }
    #[staticmethod]
    fn float64() -> Self {
        Self { dt: DType::Float64 }
    }
    #[staticmethod]
    fn bool_() -> Self {
        Self { dt: DType::Bool }
    }
    #[staticmethod]
    fn char() -> Self {
        Self { dt: DType::Char }
    }

    #[staticmethod]
    fn string() -> Self {
        Self { dt: DType::String }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferClass {
    Bool,
    Int,
    Float,
    Char,
    String,
}

pub fn infer_dtype(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<InferClass>> {
    // Returns None if there are no non-null leaves.
    if obj.is_none() {
        return Ok(None);
    }
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Some(InferClass::Bool));
    }
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(Some(InferClass::Int));
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Ok(Some(InferClass::Float));
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        let s = obj.extract::<String>()?;
        if s.chars().count() == 1 {
            return Ok(Some(InferClass::Char));
        }
        return Ok(Some(InferClass::String));
    }

    if is_sequence_like(py, obj)? {
        let seq = obj.downcast::<pyo3::types::PySequence>()?;
        let mut acc: Option<InferClass> = None;
        for i in 0..seq.len()? {
            let item = seq.get_item(i)?;
            if let Some(cls) = infer_dtype(py, &item)? {
                acc = Some(merge_infer(acc, cls)?);
            }
        }
        return Ok(acc);
    }

    Err(PyValueError::new_err(format!(
        "Unsupported value type for grumpy array construction: {}",
        obj.get_type().name()?
    )))
}

fn merge_infer(acc: Option<InferClass>, newv: InferClass) -> PyResult<InferClass> {
    match (acc, newv) {
        (None, x) => Ok(x),
        (Some(InferClass::Bool), InferClass::Bool) => Ok(InferClass::Bool),
        (Some(InferClass::Int), InferClass::Int) => Ok(InferClass::Int),
        (Some(InferClass::Float), InferClass::Float) => Ok(InferClass::Float),
        (Some(InferClass::Char), InferClass::Char) => Ok(InferClass::Char),
        (Some(InferClass::String), InferClass::String) => Ok(InferClass::String),
        (Some(InferClass::Bool), InferClass::Int) => Ok(InferClass::Int),
        (Some(InferClass::Int), InferClass::Bool) => Ok(InferClass::Int),
        (Some(InferClass::Bool), InferClass::Float) => Ok(InferClass::Float),
        (Some(InferClass::Float), InferClass::Bool) => Ok(InferClass::Float),
        (Some(InferClass::Int), InferClass::Float) => Ok(InferClass::Float),
        (Some(InferClass::Float), InferClass::Int) => Ok(InferClass::Float),
        (Some(InferClass::Char), _) | (_, InferClass::Char) => Err(PyValueError::new_err(
            "Cannot infer a single dtype from mixed char and other values. Specify dtype explicitly.",
        )),
        (Some(InferClass::String), _) | (_, InferClass::String) => Err(PyValueError::new_err(
            "Cannot infer a single dtype from mixed string and non-string values. Specify dtype explicitly.",
        )),
    }
}

pub fn inferclass_to_dtype(cls: InferClass) -> DType {
    match cls {
        InferClass::Bool => DType::Bool,
        InferClass::Int => DType::Int64,
        InferClass::Float => DType::Float64,
        InferClass::Char => DType::Char,
        InferClass::String => DType::String,
    }
}

pub fn is_sequence_like(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<pyo3::types::PyString>() || obj.is_instance_of::<pyo3::types::PyBytes>()
    {
        return Ok(false);
    }
    let _ = py; // keep signature stable; no longer needed
    Ok(obj.downcast::<pyo3::types::PySequence>().is_ok())
}


