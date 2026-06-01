use pyo3::prelude::*;

#[pyfunction]
pub fn gpu_available() -> bool {
    crate::gpu::gpu_available_py()
}

#[pyfunction]
pub fn gpu_backend() -> Option<String> {
    crate::gpu::backend_name_py().map(str::to_string)
}
