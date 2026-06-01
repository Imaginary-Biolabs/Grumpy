use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::layout::{gather_2d_fancy_sum_i64, scatter_2d_fancy_i32, Layout};
use crate::ops::{self, BinOp};
use crate::py_api::bench::{rect2d_i32_view, sum_i32_add_neon, sum_i32_mul_neon, sum_i32_to_i64_neon};
use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    /// Kernel-only: mul for 2D rectangular int32 arrays, returning an i64 checksum.
    /// This avoids allocating an output array so benchmarks can measure compute only.
    fn _mul2d_i32_sum_i64(&self, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 || other.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_mul2d_i32_sum_i64 requires int32 arrays."));
        }
        let (a_off, a_vals) = rect2d_i32_view(&self.inner.layout)?;
        let (b_off, b_vals) = rect2d_i32_view(&other.inner.layout)?;
        if a_off != b_off {
            return Err(PyValueError::new_err("Rectangular shapes differ."));
        }
        Ok(sum_i32_mul_neon(a_vals, b_vals))
    }

    fn _add2d_i32_sum_i64(&self, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 || other.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_add2d_i32_sum_i64 requires int32 arrays."));
        }
        let (a_off, a_vals) = rect2d_i32_view(&self.inner.layout)?;
        let (b_off, b_vals) = rect2d_i32_view(&other.inner.layout)?;
        if a_off != b_off {
            return Err(PyValueError::new_err("Rectangular shapes differ."));
        }
        Ok(sum_i32_add_neon(a_vals, b_vals))
    }

    /// Sum all elements for 2D pure list-chain int32 arrays (leaf sum). Used for benchmarking
    /// full elementwise ops without converting to Python/NumPy.
    fn _sum2d_i32_i64(&self) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_sum2d_i32_i64 requires int32 arrays."));
        }
        let (_off, vals) = rect2d_i32_view(&self.inner.layout)?;
        Ok(sum_i32_to_i64_neon(vals))
    }

    /// Kernel-only: sum over dim=1 for 2D rectangular int32 arrays, returning a checksum (sum of row sums).
    fn _sum2d_dim1_i32_sum_i64(&self) -> PyResult<i64> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_sum2d_dim1_i32_sum_i64 requires int32 array."));
        }
        let (off, vals) = rect2d_i32_view(&self.inner.layout)?;
        if off.len() < 2 {
            return Ok(0);
        }
        let nrows = off.len() - 1;
        let mut acc: i64 = 0;
        for i in 0..nrows {
            let s = off[i] as usize;
            let e = off[i + 1] as usize;
            acc = acc.wrapping_add(sum_i32_to_i64_neon(&vals[s..e]));
        }
        Ok(acc)
    }

    /// Kernel-only: mean over dim=1 for 2D rectangular int32 arrays, returning a checksum (sum of row means).
    fn _mean2d_dim1_i32_sum_f64(&self) -> PyResult<f64> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_mean2d_dim1_i32_sum_f64 requires int32 array."));
        }
        let (off, vals) = rect2d_i32_view(&self.inner.layout)?;
        if off.len() < 2 {
            return Ok(0.0);
        }
        let nrows = off.len() - 1;
        let mut acc: f64 = 0.0;
        for i in 0..nrows {
            let s = off[i] as usize;
            let e = off[i + 1] as usize;
            let len = (e - s) as f64;
            if len == 0.0 {
                continue;
            }
            let sum = sum_i32_to_i64_neon(&vals[s..e]) as f64;
            acc += sum / len;
        }
        Ok(acc)
    }

    /// Full op benchmark helper: compute (self * other).sum() for 2D int32 arrays via Grumpy's
    /// own elementwise implementation (exercises rectangular fast-path vs generic ragged path).
    fn _mul2d_i32_sum_via_op_i64(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        let _ = py;
        let out = ops::elementwise(&self.inner, &other.inner, BinOp::Mul)?;
        if out.dtype != DType::Int32 {
            return Err(PyValueError::new_err("Internal error: expected int32 output."));
        }
        let (_off, vals) = rect2d_i32_view(&out.layout)?;
        Ok(sum_i32_to_i64_neon(vals))
    }

    fn _add2d_i32_sum_via_op_i64(&self, py: Python<'_>, other: PyRef<'_, PyGrumpyArray>) -> PyResult<i64> {
        let _ = py;
        let out = ops::elementwise(&self.inner, &other.inner, BinOp::Add)?;
        if out.dtype != DType::Int32 {
            return Err(PyValueError::new_err("Internal error: expected int32 output."));
        }
        let (_off, vals) = rect2d_i32_view(&out.layout)?;
        Ok(sum_i32_to_i64_neon(vals))
    }

    /// Internal kernel: gather 2D fancy indices and return an i64 checksum (sum) without allocating output.
    /// This exists to benchmark just the gather kernel (no to_list/to_numpy overhead).
    fn _gather2d_sum_i64(&self, _py: Python<'_>, rows: Bound<'_, PyAny>, cols: Bound<'_, PyAny>) -> PyResult<i64> {
        if !self.inner.is_pure_list_chain() {
            return Err(PyValueError::new_err("gather kernel only supports pure list-chain arrays."));
        }
        let r = rows
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("rows must be a 1D NumPy int64 array."))?;
        let c = cols
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("cols must be a 1D NumPy int64 array."))?;
        let rs = r.as_slice()?;
        let cs = c.as_slice()?;
        gather_2d_fancy_sum_i64(&self.inner.layout, rs, cs, self.inner.dtype)
    }

    /// Internal kernel: scatter 2D fancy indices from NumPy arrays without Python conversions (int32 only).
    fn _scatter2d_i32(
        &mut self,
        _py: Python<'_>,
        rows: Bound<'_, PyAny>,
        cols: Bound<'_, PyAny>,
        values: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.inner.dtype != DType::Int32 {
            return Err(PyValueError::new_err("_scatter2d_i32 requires dtype=int32."));
        }
        let r = rows
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("rows must be a 1D NumPy int64 array."))?;
        let c = cols
            .extract::<PyReadonlyArray1<'_, i64>>()
            .map_err(|_| PyValueError::new_err("cols must be a 1D NumPy int64 array."))?;
        let v = values
            .extract::<PyReadonlyArray1<'_, i32>>()
            .map_err(|_| PyValueError::new_err("values must be a 1D NumPy int32 array."))?;
        let rs = r.as_slice()?;
        let cs = c.as_slice()?;
        let vs = v.as_slice()?;
        crate::layout::scatter_2d_fancy_i32(&mut self.inner.layout, rs, cs, vs)?;
        Ok(())
    }
}
