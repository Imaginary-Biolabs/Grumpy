//! CUDA brute-force kNN via NVRTC (Linux + `--features cuda`).

use super::knn::{KnnDim1Result, KNN_GPU_MAX_K};
use crate::error::{internal, unsupported};
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use pyo3::prelude::*;
use std::sync::{Arc, OnceLock};

const CUDA_DIM0: &str = r#"
extern "C" __global__ void knn_dim0_f64(
    const double *q_coords,
    const double *d_coords,
    int *out_idx,
    double *out_dist,
    int qn, int dn, int d, int k,
    int exclude_self, int write_dist
) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= qn) return;
    double best_d2[64];
    int best_j[64];
    for (int t = 0; t < k; t++) { best_d2[t] = 1.0/0.0; best_j[t] = 0; }
    int qstart = qi * d;
    for (int j = 0; j < dn; j++) {
        if (exclude_self && qi == j) continue;
        int jstart = j * d;
        double acc = 0.0;
        for (int c = 0; c < d; c++) {
            double diff = q_coords[qstart + c] - d_coords[jstart + c];
            acc += diff * diff;
        }
        int worst = 0;
        double worst_val = best_d2[0];
        for (int t = 1; t < k; t++) {
            if (best_d2[t] > worst_val) { worst_val = best_d2[t]; worst = t; }
        }
        if (acc < worst_val || (acc == worst_val && j < best_j[worst])) {
            best_d2[worst] = acc;
            best_j[worst] = j;
        }
    }
    for (int a = 0; a < k - 1; a++) {
        for (int b = a + 1; b < k; b++) {
            int do_swap = (best_d2[b] < best_d2[a]) ||
                (best_d2[b] == best_d2[a] && best_j[b] < best_j[a]);
            if (do_swap) {
                double td = best_d2[a]; best_d2[a] = best_d2[b]; best_d2[b] = td;
                int tj = best_j[a]; best_j[a] = best_j[b]; best_j[b] = tj;
            }
        }
    }
    int base = qi * k;
    for (int t = 0; t < k; t++) {
        out_idx[base + t] = best_j[t];
        if (write_dist) out_dist[base + t] = sqrt(best_d2[t]);
    }
}
"#;

struct CudaKnn {
    dev: Arc<CudaDevice>,
    k_dim0: CudaFunction,
}

fn ctx() -> PyResult<&'static CudaKnn> {
    static INIT: OnceLock<Result<CudaKnn, String>> = OnceLock::new();
    let slot = INIT.get_or_init(|| CudaKnn::new().map_err(|e| e.to_string()));
    slot.as_ref().map_err(|e| internal("cuda", e.clone()))
}

pub fn is_available() -> bool {
    CudaDevice::new(0).is_ok()
}

impl CudaKnn {
    fn new() -> PyResult<Self> {
        let dev = CudaDevice::new(0).map_err(|e| internal("cuda", format!("device init: {e}")))?;
        dev.set_default_stream();
        let ptx = dev
            .compile_ptx(CUDA_DIM0)
            .map_err(|e| internal("cuda", format!("NVRTC failed: {e}")))?;
        dev.load_ptx(ptx, "knn", &["knn_dim0_f64"])
            .map_err(|e| internal("cuda", format!("load ptx: {e}")))?;
        let k_dim0 = dev
            .get_func("knn", "knn_dim0_f64")
            .ok_or_else(|| internal("cuda", "missing knn_dim0_f64"))?;
        Ok(Self { dev, k_dim0 })
    }
}

pub fn knn_dim0_f64(
    q_dense: &[f64],
    d_dense: &[f64],
    qn: usize,
    dn: usize,
    d: usize,
    k: usize,
    exclude_self: bool,
    return_distances: bool,
) -> PyResult<(Vec<usize>, Option<Vec<f64>>)> {
    if k > KNN_GPU_MAX_K {
        return Err(unsupported(
            "gpu kNN",
            format!("k={k} exceeds GPU max k={KNN_GPU_MAX_K}"),
            "reduce k or use cpu.",
        ));
    }
    let c = ctx()?;
    let q_dev = c
        .dev
        .clone_htod(q_dense)
        .map_err(|e| internal("cuda", format!("H2D q: {e}")))?;
    let d_dev = c
        .dev
        .clone_htod(d_dense)
        .map_err(|e| internal("cuda", format!("H2D d: {e}")))?;
    let mut out_idx = vec![0i32; qn * k];
    let out_dev = c
        .dev
        .clone_htod(&out_idx)
        .map_err(|e| internal("cuda", format!("H2D out: {e}")))?;
    let mut out_dist = vec![0.0f64; qn * k];
    let dist_dev = c
        .dev
        .clone_htod(&out_dist)
        .map_err(|e| internal("cuda", format!("H2D dist: {e}")))?;

    let block = 256u32;
    let grid = ((qn as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let ex = i32::from(exclude_self);
    let wd = i32::from(return_distances);
    unsafe {
        c.k_dim0
            .launch(
                cfg,
                (
                    &q_dev,
                    &d_dev,
                    &out_dev,
                    &dist_dev,
                    qn as i32,
                    dn as i32,
                    d as i32,
                    k as i32,
                    ex,
                    wd,
                ),
            )
            .map_err(|e| internal("cuda", format!("launch: {e}")))?;
    }
    c.dev
        .dtoh_sync_copy_into(&out_dev, &mut out_idx)
        .map_err(|e| internal("cuda", format!("D2H idx: {e}")))?;
    let nn_idx: Vec<usize> = out_idx.into_iter().map(|x| x as usize).collect();
    let nn_dist = if return_distances {
        c.dev
            .dtoh_sync_copy_into(&dist_dev, &mut out_dist)
            .map_err(|e| internal("cuda", format!("D2H dist: {e}")))?;
        Some(out_dist)
    } else {
        None
    };
    Ok((nn_idx, nn_dist))
}

pub fn knn_dim1_f64(
    _q_coords: &[f64],
    _d_coords: &[f64],
    _group_offsets: &[i64],
    _group_base: usize,
    _point_offsets: &[i64],
    _n_groups: usize,
    _d: usize,
    _k: usize,
    _exclude_self: bool,
    _return_distances: bool,
) -> PyResult<KnnDim1Result> {
    Err(unsupported(
        "cuda kNN dim1",
        "not implemented yet; use cpu or Metal",
        "call neighbors with dim=1 on CPU for now.",
    ))
}
