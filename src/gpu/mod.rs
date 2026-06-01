//! GPU acceleration for select heavy kernels (kNN today).
//!
//! Metal is used on macOS; CUDA when built with `--features cuda`.

pub mod auto;
pub mod knn;

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
mod knn_cuda;
#[cfg(target_os = "macos")]
mod knn_metal;

use pyo3::prelude::*;

/// User-facing GPU preference for neighbors / streaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuPreference {
    Never,
    Auto,
    Force,
}

impl GpuPreference {
    pub fn parse(s: &str) -> PyResult<Self> {
        match s {
            "never" | "false" | "False" | "0" => Ok(Self::Never),
            "auto" => Ok(Self::Auto),
            "true" | "True" | "1" | "force" => Ok(Self::Force),
            other => Err(crate::error::arg_invalid(
                "gpu",
                format!("unknown value '{other}'"),
                "use gpu='auto', True, or False.",
            )),
        }
    }

    pub fn should_try(self) -> bool {
        matches!(self, Self::Auto | Self::Force)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    None,
    Metal,
    Cuda,
}

pub fn active_backend() -> GpuBackend {
    #[cfg(target_os = "macos")]
    {
        if knn_metal::is_available() {
            return GpuBackend::Metal;
        }
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        if knn_cuda::is_available() {
            return GpuBackend::Cuda;
        }
    }
    #[cfg(all(feature = "cuda", target_os = "macos"))]
    {
        let _ = ();
    }
    GpuBackend::None
}

pub fn backend_name() -> Option<&'static str> {
    match active_backend() {
        GpuBackend::None => None,
        GpuBackend::Metal => Some("metal"),
        GpuBackend::Cuda => Some("cuda"),
    }
}

pub fn gpu_available_py() -> bool {
    active_backend() != GpuBackend::None
}

pub fn backend_name_py() -> Option<&'static str> {
    backend_name()
}

pub use auto::{should_use_gpu, GpuOp, GpuWorkEstimate};

pub use knn::{knn_dim0_bruteforce, knn_dim1_bruteforce};
