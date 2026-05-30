//! Shared SIMD-friendly leaf kernels used by elementwise ops and reductions.

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::*;

    pub fn mul_i32_slices(a: &[i32], b: &[i32], out: &mut [i32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0usize;
        while i + 4 <= n {
            let va = unsafe { vld1q_s32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_s32(b.as_ptr().add(i)) };
            let vr = unsafe { vmulq_s32(va, vb) };
            unsafe { vst1q_s32(out.as_mut_ptr().add(i), vr) };
            i += 4;
        }
        while i < n {
            out[i] = a[i].wrapping_mul(b[i]);
            i += 1;
        }
    }

    pub fn add_i32_slices(a: &[i32], b: &[i32], out: &mut [i32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0usize;
        while i + 4 <= n {
            let va = unsafe { vld1q_s32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_s32(b.as_ptr().add(i)) };
            let vr = unsafe { vaddq_s32(va, vb) };
            unsafe { vst1q_s32(out.as_mut_ptr().add(i), vr) };
            i += 4;
        }
        while i < n {
            out[i] = a[i].wrapping_add(b[i]);
            i += 1;
        }
    }

    pub fn sub_i32_slices(a: &[i32], b: &[i32], out: &mut [i32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0usize;
        while i + 4 <= n {
            let va = unsafe { vld1q_s32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_s32(b.as_ptr().add(i)) };
            let vr = unsafe { vsubq_s32(va, vb) };
            unsafe { vst1q_s32(out.as_mut_ptr().add(i), vr) };
            i += 4;
        }
        while i < n {
            out[i] = a[i].wrapping_sub(b[i]);
            i += 1;
        }
    }

    pub fn mul_i32_scalar_slice(a: &[i32], scalar: i32, out: &mut [i32]) {
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();
        let mut i = 0usize;
        while i + 4 <= n {
            let va = unsafe { vld1q_s32(a.as_ptr().add(i)) };
            let vs = unsafe { vdupq_n_s32(scalar) };
            let vr = unsafe { vmulq_s32(va, vs) };
            unsafe { vst1q_s32(out.as_mut_ptr().add(i), vr) };
            i += 4;
        }
        while i < n {
            out[i] = a[i].wrapping_mul(scalar);
            i += 1;
        }
    }

    pub fn sum_i32_to_i64(a: &[i32]) -> i64 {
        let n = a.len();
        let mut acc0 = unsafe { vdupq_n_s64(0) };
        let mut acc1 = unsafe { vdupq_n_s64(0) };
        let mut i = 0usize;
        unsafe {
            while i + 4 <= n {
                let va = vld1q_s32(a.as_ptr().add(i));
                let lo = vmovl_s32(vget_low_s32(va));
                let hi = vmovl_s32(vget_high_s32(va));
                acc0 = vaddq_s64(acc0, lo);
                acc1 = vaddq_s64(acc1, hi);
                i += 4;
            }
        }
        let mut sum = unsafe {
            let acc = vaddq_s64(acc0, acc1);
            vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1)
        };
        while i < n {
            sum += a[i] as i64;
            i += 1;
        }
        sum
    }

    pub fn sum_i32_mul_to_i64(a: &[i32], b: &[i32]) -> i64 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut acc0 = unsafe { vdupq_n_s64(0) };
        let mut acc1 = unsafe { vdupq_n_s64(0) };
        let mut i = 0usize;
        unsafe {
            while i + 4 <= n {
                let va = vld1q_s32(a.as_ptr().add(i));
                let vb = vld1q_s32(b.as_ptr().add(i));
                let lo = vmull_s32(vget_low_s32(va), vget_low_s32(vb));
                let hi = vmull_s32(vget_high_s32(va), vget_high_s32(vb));
                acc0 = vaddq_s64(acc0, lo);
                acc1 = vaddq_s64(acc1, hi);
                i += 4;
            }
        }
        let mut sum = unsafe {
            let acc = vaddq_s64(acc0, acc1);
            vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1)
        };
        while i < n {
            sum += (a[i].wrapping_mul(b[i])) as i64;
            i += 1;
        }
        sum
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod imp {
    pub fn mul_i32_slices(a: &[i32], b: &[i32], out: &mut [i32]) {
        for i in 0..a.len() {
            out[i] = a[i].wrapping_mul(b[i]);
        }
    }

    pub fn add_i32_slices(a: &[i32], b: &[i32], out: &mut [i32]) {
        for i in 0..a.len() {
            out[i] = a[i].wrapping_add(b[i]);
        }
    }

    pub fn sub_i32_slices(a: &[i32], b: &[i32], out: &mut [i32]) {
        for i in 0..a.len() {
            out[i] = a[i].wrapping_sub(b[i]);
        }
    }

    pub fn mul_i32_scalar_slice(a: &[i32], scalar: i32, out: &mut [i32]) {
        for i in 0..a.len() {
            out[i] = a[i].wrapping_mul(scalar);
        }
    }

    pub fn sum_i32_to_i64(a: &[i32]) -> i64 {
        a.iter().map(|&x| x as i64).sum()
    }

    pub fn sum_i32_mul_to_i64(a: &[i32], b: &[i32]) -> i64 {
        let mut acc: i64 = 0;
        for i in 0..a.len() {
            acc = acc.wrapping_add((a[i].wrapping_mul(b[i])) as i64);
        }
        acc
    }
}

pub use imp::*;

pub fn sum_i32_rows_to_i64(values: &[i32], offsets: &[i64]) -> i64 {
    let nrows = offsets.len().saturating_sub(1);
    let mut total: i64 = 0;
    for i in 0..nrows {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        total = total.wrapping_add(sum_i32_to_i64(&values[s..e]));
    }
    total
}

pub fn sum_i32_row_sums_to_i64(values: &[i32], offsets: &[i64], out: &mut [i64]) {
    let nrows = offsets.len().saturating_sub(1);
    debug_assert_eq!(out.len(), nrows);
    for i in 0..nrows {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        out[i] = sum_i32_to_i64(&values[s..e]);
    }
}

pub fn sum_i32_mul_scalar_to_i64(a: &[i32], scalar: i32) -> i64 {
    let mut acc: i64 = 0;
    for &x in a {
        acc = acc.wrapping_add(x.wrapping_mul(scalar) as i64);
    }
    acc
}

pub fn sum_i32_add_to_i64(a: &[i32], b: &[i32]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        acc = acc.wrapping_add(a[i].wrapping_add(b[i]) as i64);
    }
    acc
}
