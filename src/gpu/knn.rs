//! Host-side kNN orchestration: dense layout extraction + GPU dispatch + CPU fallback hooks.

use super::{should_use_gpu, GpuOp, GpuPreference, GpuWorkEstimate};
use crate::error::{internal, unsupported};
use pyo3::prelude::*;

pub const KNN_GPU_MAX_K: usize = 64;

pub struct KnnDim0Result {
    pub nn_idx: Vec<usize>,
    pub nn_dist: Option<Vec<f64>>,
}

pub struct KnnDim1Result {
    /// Flat edge pairs (src_local, dst_local) in group iteration order.
    pub edge_vals: Vec<i64>,
    pub point_neighbor_counts: Vec<usize>,
    pub group_point_counts: Vec<usize>,
    pub nn_dist: Option<Vec<f64>>,
}

pub fn knn_dim0_bruteforce(
    q_coords: &[f64],
    d_coords: &[f64],
    qoff: &[i64],
    qbase: usize,
    doff: &[i64],
    dbase: usize,
    qn: usize,
    dn: usize,
    d: usize,
    k: usize,
    loop_: bool,
    same_inputs: bool,
    return_distances: bool,
    gpu: GpuPreference,
) -> PyResult<Option<KnnDim0Result>> {
    if k == 0 {
        return Ok(Some(KnnDim0Result {
            nn_idx: vec![],
            nn_dist: if return_distances { Some(vec![]) } else { None },
        }));
    }
    if !should_use_gpu(
        gpu,
        GpuOp::KnnDim0,
        &GpuWorkEstimate::dim0(qn, dn, d, k),
        KNN_GPU_MAX_K,
    )? {
        return Ok(None);
    }
    let q_dense = dense_coords(q_coords, qoff, qbase, qn, d)?;
    let d_dense = if same_inputs {
        q_dense.clone()
    } else {
        dense_coords(d_coords, doff, dbase, dn, d)?
    };
    let exclude_self = !loop_ && same_inputs;
    let (nn_idx, nn_dist) = dispatch_dim0(
        &q_dense,
        &d_dense,
        qn,
        dn,
        d,
        k,
        exclude_self,
        return_distances,
    )?;
    Ok(Some(KnnDim0Result { nn_idx, nn_dist }))
}

pub fn knn_dim1_bruteforce(
    q_coords: &[f64],
    d_coords: &[f64],
    group_offsets: &[i64],
    group_base: usize,
    point_offsets: &[i64],
    n_groups: usize,
    d: usize,
    k: usize,
    loop_: bool,
    same_inputs: bool,
    return_distances: bool,
    gpu: GpuPreference,
) -> PyResult<Option<KnnDim1Result>> {
    if k == 0 {
        return Ok(Some(KnnDim1Result {
            edge_vals: vec![],
            point_neighbor_counts: vec![],
            group_point_counts: vec![],
            nn_dist: if return_distances { Some(vec![]) } else { None },
        }));
    }
    let mut total_pairs = 0usize;
    let mut n_queries = 0usize;
    for g in 0..n_groups {
        let qn = (group_offsets[group_base + g + 1] - group_offsets[group_base + g]) as usize;
        let dn = qn; // same group layout for query/data
        n_queries = n_queries.saturating_add(qn);
        total_pairs = total_pairs.saturating_add(qn.saturating_mul(dn));
    }
    if !should_use_gpu(
        gpu,
        GpuOp::KnnDim1,
        &GpuWorkEstimate::dim1(total_pairs.max(1), n_groups, n_queries, d, k),
        KNN_GPU_MAX_K,
    )? {
        return Ok(None);
    }
    let exclude_self = !loop_ && same_inputs;
    dispatch_dim1(
        q_coords,
        d_coords,
        group_offsets,
        group_base,
        point_offsets,
        n_groups,
        d,
        k,
        exclude_self,
        return_distances,
    )
    .map(Some)
}

fn dispatch_dim0(
    q_dense: &[f64],
    d_dense: &[f64],
    qn: usize,
    dn: usize,
    d: usize,
    k: usize,
    exclude_self: bool,
    return_distances: bool,
) -> PyResult<(Vec<usize>, Option<Vec<f64>>)> {
    #[cfg(target_os = "macos")]
    {
        if super::knn_metal::is_available() {
            return super::knn_metal::knn_dim0_f64(
                q_dense,
                d_dense,
                qn,
                dn,
                d,
                k,
                exclude_self,
                return_distances,
            );
        }
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        if super::knn_cuda::is_available() {
            return super::knn_cuda::knn_dim0_f64(
                q_dense,
                d_dense,
                qn,
                dn,
                d,
                k,
                exclude_self,
                return_distances,
            );
        }
    }
    Err(unsupported(
        "gpu kNN",
        "no GPU backend available at runtime",
        "use gpu=False or build with Metal/CUDA support.",
    ))
}

fn dispatch_dim1(
    q_coords: &[f64],
    d_coords: &[f64],
    group_offsets: &[i64],
    group_base: usize,
    point_offsets: &[i64],
    n_groups: usize,
    d: usize,
    k: usize,
    exclude_self: bool,
    return_distances: bool,
) -> PyResult<KnnDim1Result> {
    #[cfg(target_os = "macos")]
    {
        if super::knn_metal::is_available() {
            return super::knn_metal::knn_dim1_f64(
                q_coords,
                d_coords,
                group_offsets,
                group_base,
                point_offsets,
                n_groups,
                d,
                k,
                exclude_self,
                return_distances,
            );
        }
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        if super::knn_cuda::is_available() {
            return super::knn_cuda::knn_dim1_f64(
                q_coords,
                d_coords,
                group_offsets,
                group_base,
                point_offsets,
                n_groups,
                d,
                k,
                exclude_self,
                return_distances,
            );
        }
    }
    Err(unsupported(
        "gpu kNN",
        "no GPU backend available at runtime",
        "use gpu=False or build with Metal/CUDA support.",
    ))
}

/// Extract dense row-major coordinates for `n` points from leaf storage.
pub fn dense_coords(
    coords: &[f64],
    offsets: &[i64],
    base: usize,
    n: usize,
    d: usize,
) -> PyResult<Vec<f64>> {
    let mut out = vec![0.0f64; n * d];
    for i in 0..n {
        let start = offsets[base + i] as usize;
        let end = start + d;
        if end > coords.len() {
            return Err(internal(
                "gpu kNN",
                "coordinate slice out of bounds while densifying",
            ));
        }
        out[i * d..(i + 1) * d].copy_from_slice(&coords[start..end]);
    }
    Ok(out)
}
