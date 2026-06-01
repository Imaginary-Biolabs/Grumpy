use crate::layout::Layout;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn rect2d_i32_view<'a>(layout: &'a Layout) -> PyResult<(&'a [i64], &'a [i32])> {
    let lo = match layout {
        Layout::ListOffset(lo) => lo,
        Layout::OffsetView(v) => {
            // treat as a view into an existing offset buffer
            return Ok((
                v.offsets.as_slice(),
                match v.content.as_ref() {
                    Layout::Leaf(leaf) => match &leaf.buffer {
                        crate::layout::LeafBuffer::I32(buf) => buf.as_slice(),
                        _ => return Err(PyValueError::new_err("Expected int32 leaf buffer.")),
                    },
                    _ => return Err(PyValueError::new_err("Expected leaf values.")),
                },
            ));
        }
        _ => return Err(PyValueError::new_err("Expected 2D list layout.")),
    };
    let leaf = match lo.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(PyValueError::new_err("Expected leaf values.")),
    };
    if leaf.has_nulls {
        return Err(PyValueError::new_err("Kernel requires all-valid leaf (no nulls)."));
    }
    let vals = match &leaf.buffer {
        crate::layout::LeafBuffer::I32(v) => v.as_slice(),
        _ => return Err(PyValueError::new_err("Expected int32 leaf buffer.")),
    };
    Ok((lo.offsets.as_slice(), vals))
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn sum_i32_to_i64_neon(a: &[i32]) -> i64 {
    crate::kernels::sum_i32_to_i64(a)
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn sum_i32_to_i64_neon(a: &[i32]) -> i64 {
    crate::kernels::sum_i32_to_i64(a)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn sum_i32_mul_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_mul_to_i64(a, b)
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn sum_i32_mul_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_mul_to_i64(a, b)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn sum_i32_add_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_add_to_i64(a, b)
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn sum_i32_add_neon(a: &[i32], b: &[i32]) -> i64 {
    crate::kernels::sum_i32_add_to_i64(a, b)
}
