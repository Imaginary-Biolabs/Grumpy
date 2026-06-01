use crate::dtype::PyDType;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (from_dtype, to_dtype, casting=None))]
pub fn py_can_cast(from_dtype: PyDType, to_dtype: PyDType, casting: Option<&str>) -> PyResult<bool> {
    let mode = crate::cast::CastMode::parse(casting.unwrap_or("safe"))?;
    Ok(crate::cast::can_cast(from_dtype.dt, to_dtype.dt, mode))
}

#[pyfunction]
pub fn py_promote_types(a: PyDType, b: PyDType) -> PyResult<PyDType> {
    Ok(PyDType {
        dt: crate::cast::promote_types(a.dt, b.dt)?,
    })
}
