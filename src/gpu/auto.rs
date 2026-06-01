//! Per-kernel GPU auto-selection thresholds.
//!
//! Each accelerated op defines its own minimum work estimate so `gpu="auto"`
//! avoids launch/sync overhead on batches too small to amortize fixed cost.

use super::{active_backend, GpuBackend, GpuPreference};
use pyo3::prelude::*;

/// Kernel tag for auto GPU routing (each op carries its own threshold).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuOp {
    /// Single point cloud kNN (dim=0 brute-force path).
    KnnDim0,
    /// Grouped kNN (dim=1); one launch amortizes over many groups.
    KnnDim1,
}

/// Work estimate for auto routing.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuWorkEstimate {
    /// Query points in this launch (dim=0 `qn`; dim=1 total queries across groups).
    pub n_queries: usize,
    /// Data points for dim=0 (`dn`); unused for dim=1.
    pub n_data: usize,
    /// Sum of per-group qn×dn (dim=1) or qn×dn (dim=0).
    pub total_pairs: usize,
    /// Groups in this launch (1 for dim=0).
    pub n_groups: usize,
    /// Coordinate dimensionality (e.g. 3 for xyz).
    pub coord_dim: usize,
    pub k: usize,
}

impl GpuWorkEstimate {
    pub fn dim0(qn: usize, dn: usize, d: usize, k: usize) -> Self {
        Self {
            n_queries: qn,
            n_data: dn,
            total_pairs: qn.saturating_mul(dn),
            n_groups: 1,
            coord_dim: d,
            k,
        }
    }

    pub fn dim1(total_pairs: usize, n_groups: usize, n_queries: usize, d: usize, k: usize) -> Self {
        Self {
            n_queries,
            n_data: 0,
            total_pairs,
            n_groups,
            coord_dim: d,
            k,
        }
    }

    /// Approximate scalar distance accumulations (pair comparisons × coord dim).
    pub fn distance_evals(&self) -> usize {
        self.total_pairs.saturating_mul(self.coord_dim.max(1))
    }
}

impl GpuOp {
    /// Minimum distance evaluations before auto selects GPU.
    fn auto_min_distance_evals(self) -> usize {
        match self {
            // One launch; ~128×128×3 is enough to hide fixed cost on Metal/CUDA.
            Self::KnnDim0 => 128 * 128 * 3,
            // Ragged per-query kernel: stream batch_size=32 × 256 residues is enough on Metal.
            Self::KnnDim1 => 4_000_000,
        }
    }

    /// Extra gate for ops whose GPU kernel parallelizes over groups.
    fn auto_min_groups(self) -> Option<usize> {
        match self {
            Self::KnnDim0 => None,
            // One thread per query point; batch_size=32 streams are viable.
            Self::KnnDim1 => Some(32),
        }
    }
}

pub fn should_use_gpu(
    pref: GpuPreference,
    op: GpuOp,
    work: &GpuWorkEstimate,
    max_k: usize,
) -> PyResult<bool> {
    if !pref.should_try() {
        return Ok(false);
    }
    let backend = active_backend();
    if backend == GpuBackend::None {
        if pref == GpuPreference::Force {
            return Err(crate::error::unsupported(
                "gpu",
                "no GPU backend available (build with Metal on macOS or --features cuda on Linux)",
                "use gpu=False or install CUDA / use an Apple Silicon Mac.",
            ));
        }
        return Ok(false);
    }
    if work.k == 0 || work.k > max_k {
        return Ok(false);
    }
    if pref == GpuPreference::Force {
        return Ok(true);
    }
    if work.distance_evals() < op.auto_min_distance_evals() {
        return Ok(false);
    }
    if let Some(min_groups) = op.auto_min_groups() {
        if work.n_groups < min_groups {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim1_stream_batch_uses_gpu_auto() {
        // 32 proteins × 256 residues (typical stream batch_size=32).
        let work = GpuWorkEstimate::dim1(32 * 256 * 256, 32, 32 * 256, 3, 16);
        assert!(should_use_gpu(GpuPreference::Auto, GpuOp::KnnDim1, &work, 64).unwrap());
    }

    #[test]
    fn dim1_small_batch_stays_cpu_auto() {
        let work = GpuWorkEstimate::dim1(8 * 64 * 64, 8, 8 * 64, 3, 16);
        assert!(!should_use_gpu(GpuPreference::Auto, GpuOp::KnnDim1, &work, 64).unwrap());
    }

    #[test]
    fn dim1_full_protein_set_uses_gpu_auto() {
        let work = GpuWorkEstimate::dim1(256 * 256 * 256, 256, 256 * 256, 3, 16);
        assert!(should_use_gpu(GpuPreference::Auto, GpuOp::KnnDim1, &work, 64).unwrap());
    }

    #[test]
    fn dim0_small_cloud_stays_cpu_auto() {
        let work = GpuWorkEstimate::dim0(64, 64, 3, 8);
        assert!(!should_use_gpu(GpuPreference::Auto, GpuOp::KnnDim0, &work, 64).unwrap());
    }

    #[test]
    fn dim0_medium_cloud_uses_gpu_auto() {
        let work = GpuWorkEstimate::dim0(256, 256, 3, 16);
        assert!(should_use_gpu(GpuPreference::Auto, GpuOp::KnnDim0, &work, 64).unwrap());
    }
}
