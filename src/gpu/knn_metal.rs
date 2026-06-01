//! Metal brute-force kNN (float32 compute; float64 host I/O).
//!
//! dim=1 uses ragged leaf + offset buffers on GPU (no densification).

use super::knn::{KnnDim1Result, KNN_GPU_MAX_K};
use crate::error::{internal, unsupported};
use metal::*;
use pyo3::prelude::*;
use std::mem::size_of;

const MSL_DIM0: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void knn_dim0_f32(
    device const float *q_coords [[buffer(0)]],
    device const float *d_coords [[buffer(1)]],
    device int *out_idx [[buffer(2)]],
    device float *out_dist [[buffer(3)]],
    constant int &qn [[buffer(4)]],
    constant int &dn [[buffer(5)]],
    constant int &d [[buffer(6)]],
    constant int &k [[buffer(7)]],
    constant int &exclude_self [[buffer(8)]],
    constant int &write_dist [[buffer(9)]],
    uint qi [[thread_position_in_grid]]
) {
    if (qi >= uint(qn)) return;
    float best_d2[64];
    int best_j[64];
    for (int t = 0; t < k; t++) { best_d2[t] = INFINITY; best_j[t] = 0; }
    int qstart = int(qi) * d;
    for (int j = 0; j < dn; j++) {
        if (exclude_self != 0 && int(qi) == j) continue;
        int jstart = j * d;
        float acc = 0.0f;
        for (int c = 0; c < d; c++) {
            float diff = q_coords[qstart + c] - d_coords[jstart + c];
            acc += diff * diff;
        }
        int worst = 0;
        float worst_val = best_d2[0];
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
            bool do_swap = (best_d2[b] < best_d2[a]) ||
                (best_d2[b] == best_d2[a] && best_j[b] < best_j[a]);
            if (do_swap) {
                float td = best_d2[a]; best_d2[a] = best_d2[b]; best_d2[b] = td;
                int tj = best_j[a]; best_j[a] = best_j[b]; best_j[b] = tj;
            }
        }
    }
    int base = int(qi) * k;
    for (int t = 0; t < k; t++) {
        out_idx[base + t] = best_j[t];
        if (write_dist != 0) out_dist[base + t] = sqrt(best_d2[t]);
    }
}
"#;

/// dim=1 ragged kNN: one thread per query point; reads coords via point_offsets or fixed stride.
const MSL_DIM1_RAGGED: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline int coord_start(
    int point,
    device const int *point_offsets,
    int point_base,
    int fixed_stride
) {
    if (fixed_stride > 0) return point_base + point * fixed_stride;
    return point_offsets[point];
}

kernel void knn_dim1_ragged_f32(
    device const float *coords [[buffer(0)]],
    device const int *point_offsets [[buffer(1)]],
    device const int *query_info [[buffer(2)]],
    device int *out_idx [[buffer(3)]],
    device float *out_dist [[buffer(4)]],
    constant int &n_queries [[buffer(5)]],
    constant int &d [[buffer(6)]],
    constant int &k [[buffer(7)]],
    constant int &exclude_self [[buffer(8)]],
    constant int &write_dist [[buffer(9)]],
    constant int &fixed_stride [[buffer(10)]],
    constant int &point_base [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(n_queries)) return;
    int qi_point = query_info[int(tid) * 4 + 0];
    int dps = query_info[int(tid) * 4 + 1];
    int ndp = query_info[int(tid) * 4 + 2];
    int qstart = coord_start(qi_point, point_offsets, point_base, fixed_stride);
    float best_d2[64];
    int best_j[64];
    for (int t = 0; t < k; t++) { best_d2[t] = INFINITY; best_j[t] = 0; }
    for (int j_local = 0; j_local < ndp; j_local++) {
        int j_point = dps + j_local;
        if (exclude_self != 0 && qi_point == j_point) continue;
        int jstart = coord_start(j_point, point_offsets, point_base, fixed_stride);
        float acc = 0.0f;
        for (int c = 0; c < d; c++) {
            float diff = coords[qstart + c] - coords[jstart + c];
            acc += diff * diff;
        }
        int worst = 0;
        float worst_val = best_d2[0];
        for (int t = 1; t < k; t++) {
            if (best_d2[t] > worst_val) { worst_val = best_d2[t]; worst = t; }
        }
        if (acc < worst_val || (acc == worst_val && j_local < best_j[worst])) {
            best_d2[worst] = acc;
            best_j[worst] = j_local;
        }
    }
    for (int a = 0; a < k - 1; a++) {
        for (int b = a + 1; b < k; b++) {
            bool do_swap = (best_d2[b] < best_d2[a]) ||
                (best_d2[b] == best_d2[a] && best_j[b] < best_j[a]);
            if (do_swap) {
                float td = best_d2[a]; best_d2[a] = best_d2[b]; best_d2[b] = td;
                int tj = best_j[a]; best_j[a] = best_j[b]; best_j[b] = tj;
            }
        }
    }
    int out_base = int(tid) * k;
    for (int t = 0; t < k; t++) {
        out_idx[out_base + t] = best_j[t];
        if (write_dist != 0) out_dist[out_base + t] = sqrt(best_d2[t]);
    }
}
"#;

struct MetalKnn {
    device: Device,
    queue: CommandQueue,
    pipeline_dim0: ComputePipelineState,
    pipeline_dim1: ComputePipelineState,
}

fn metal_ctx() -> PyResult<&'static MetalKnn> {
    static INIT: std::sync::OnceLock<Result<MetalKnn, String>> = std::sync::OnceLock::new();
    let slot = INIT.get_or_init(|| MetalKnn::new().map_err(|e| e.to_string()));
    slot.as_ref().map_err(|e| internal("metal", e.clone()))
}

pub fn is_available() -> bool {
    Device::system_default().is_some()
}

impl MetalKnn {
    fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            unsupported("metal", "no default Metal device", "run on Apple GPU hardware.")
        })?;
        let queue = device.new_command_queue();
        let pipeline_dim0 = compile_pipeline(&device, MSL_DIM0, "knn_dim0_f32")?;
        let pipeline_dim1 = compile_pipeline(&device, MSL_DIM1_RAGGED, "knn_dim1_ragged_f32")?;
        Ok(Self {
            device,
            queue,
            pipeline_dim0,
            pipeline_dim1,
        })
    }
}

fn compile_pipeline(device: &Device, source: &str, name: &str) -> PyResult<ComputePipelineState> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .map_err(|e| internal("metal", format!("MSL compile failed: {e}")))?;
    let func = library
        .get_function(name, None)
        .map_err(|e| internal("metal", format!("missing kernel '{name}': {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| internal("metal", format!("pipeline build failed: {e}")))
}

fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// True when every point occupies exactly `d` contiguous scalars in the leaf.
fn is_fixed_stride(point_offsets: &[i64], n_points: usize, d: usize) -> bool {
    if n_points == 0 {
        return true;
    }
    let d_i64 = d as i64;
    for i in 0..n_points {
        if point_offsets[i + 1] - point_offsets[i] != d_i64 {
            return false;
        }
    }
    true
}

fn i64_offsets_to_i32(offsets: &[i64]) -> PyResult<Vec<i32>> {
    offsets
        .iter()
        .map(|&o| {
            i32::try_from(o).map_err(|_| {
                internal("gpu kNN", format!("offset {o} exceeds i32 range for GPU upload"))
            })
        })
        .collect()
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
    debug_assert_eq!(q_dense.len(), qn * d);
    debug_assert_eq!(d_dense.len(), dn * d);
    let q_f32 = f64_to_f32(q_dense);
    let d_f32 = f64_to_f32(d_dense);
    let ctx = metal_ctx()?;
    let q_gpu = ctx.device.new_buffer_with_data(
        q_f32.as_ptr() as *const _,
        (q_f32.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_gpu = ctx.device.new_buffer_with_data(
        d_f32.as_ptr() as *const _,
        (d_f32.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut out_idx = vec![0i32; qn * k];
    let out_idx_gpu = ctx.device.new_buffer_with_data(
        out_idx.as_ptr() as *const _,
        (out_idx.len() * size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut out_dist = vec![0.0f32; qn * k];
    let out_dist_gpu = ctx.device.new_buffer_with_data(
        out_dist.as_ptr() as *const _,
        (out_dist.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let qn_i = qn as i32;
    let dn_i = dn as i32;
    let d_i = d as i32;
    let k_i = k as i32;
    let ex_i = i32::from(exclude_self);
    let wd_i = i32::from(return_distances);

    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.pipeline_dim0);
    enc.set_buffer(0, Some(&q_gpu), 0);
    enc.set_buffer(1, Some(&d_gpu), 0);
    enc.set_buffer(2, Some(&out_idx_gpu), 0);
    enc.set_buffer(3, Some(&out_dist_gpu), 0);
    enc.set_bytes(4, size_of::<i32>() as u64, &qn_i as *const _ as _);
    enc.set_bytes(5, size_of::<i32>() as u64, &dn_i as *const _ as _);
    enc.set_bytes(6, size_of::<i32>() as u64, &d_i as *const _ as _);
    enc.set_bytes(7, size_of::<i32>() as u64, &k_i as *const _ as _);
    enc.set_bytes(8, size_of::<i32>() as u64, &ex_i as *const _ as _);
    enc.set_bytes(9, size_of::<i32>() as u64, &wd_i as *const _ as _);
    let tg = pipeline_dim0_threadgroup_size(&ctx.pipeline_dim0);
    let groups = ((qn as u64) + tg - 1) / tg;
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let idx_ptr = out_idx_gpu.contents() as *const i32;
    unsafe {
        std::ptr::copy_nonoverlapping(idx_ptr, out_idx.as_mut_ptr(), out_idx.len());
    }
    let nn_idx: Vec<usize> = out_idx.into_iter().map(|x| x as usize).collect();
    let nn_dist = if return_distances {
        let dist_ptr = out_dist_gpu.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(dist_ptr, out_dist.as_mut_ptr(), out_dist.len());
        }
        Some(out_dist.into_iter().map(f64::from).collect())
    } else {
        None
    };
    Ok((nn_idx, nn_dist))
}

fn pipeline_dim0_threadgroup_size(p: &ComputePipelineState) -> u64 {
    p.max_total_threads_per_threadgroup().min(256)
}

fn pipeline_dim1_threadgroup_size(p: &ComputePipelineState) -> u64 {
    p.max_total_threads_per_threadgroup().min(256)
}

pub fn knn_dim1_f64(
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
    if k > KNN_GPU_MAX_K {
        return Err(unsupported(
            "gpu kNN",
            format!("k={k} exceeds GPU max k={KNN_GPU_MAX_K}"),
            "reduce k or use cpu.",
        ));
    }
    let _ = d_coords; // same leaf when same_inputs
    let n_points = point_offsets.len().saturating_sub(1);

    // Phase 1: upload ragged leaf coords (f32) — no densification.
    let coords_f32 = f64_to_f32(q_coords);

    let fixed_stride = is_fixed_stride(point_offsets, n_points, d);
    let point_base = if fixed_stride {
        i32::try_from(point_offsets[0]).map_err(|_| {
            internal("gpu kNN", "point offset base exceeds i32 range for GPU upload")
        })?
    } else {
        0
    };
    let fixed_stride_i = if fixed_stride { d as i32 } else { 0 };
    let point_off_i32 = if fixed_stride {
        vec![0i32; 1] // unused; Metal requires a bound buffer
    } else {
        i64_offsets_to_i32(point_offsets)?
    };

    // One thread per query point: query_info[tid] = (qi_point, dps, ndp, local_p).
    let mut query_info = Vec::new();
    let mut group_point_counts = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let qps = group_offsets[group_base + g] as i32;
        let qpe = group_offsets[group_base + g + 1] as i32;
        let ndp = qpe - qps;
        group_point_counts.push(ndp as usize);
        for p in 0..ndp {
            query_info.push(qps + p);
            query_info.push(qps);
            query_info.push(ndp);
            query_info.push(p);
        }
    }
    let n_queries = query_info.len() / 4;

    let ctx = metal_ctx()?;
    let coords_gpu = ctx.device.new_buffer_with_data(
        coords_f32.as_ptr() as *const _,
        (coords_f32.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let po_gpu = ctx.device.new_buffer_with_data(
        point_off_i32.as_ptr() as *const _,
        (point_off_i32.len() * size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let qi_gpu = ctx.device.new_buffer_with_data(
        query_info.as_ptr() as *const _,
        (query_info.len() * size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let out_slots = n_queries * k;
    let mut out_idx = vec![0i32; out_slots];
    let out_idx_gpu = ctx.device.new_buffer_with_data(
        out_idx.as_ptr() as *const _,
        (out_idx.len() * size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mut out_dist = vec![0.0f32; out_slots];
    let out_dist_gpu = ctx.device.new_buffer_with_data(
        out_dist.as_ptr() as *const _,
        (out_dist.len() * size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let nq_i = n_queries as i32;
    let d_i = d as i32;
    let k_i = k as i32;
    let ex_i = i32::from(exclude_self);
    let wd_i = i32::from(return_distances);

    let cb = ctx.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.pipeline_dim1);
    enc.set_buffer(0, Some(&coords_gpu), 0);
    enc.set_buffer(1, Some(&po_gpu), 0);
    enc.set_buffer(2, Some(&qi_gpu), 0);
    enc.set_buffer(3, Some(&out_idx_gpu), 0);
    enc.set_buffer(4, Some(&out_dist_gpu), 0);
    enc.set_bytes(5, size_of::<i32>() as u64, &nq_i as *const _ as _);
    enc.set_bytes(6, size_of::<i32>() as u64, &d_i as *const _ as _);
    enc.set_bytes(7, size_of::<i32>() as u64, &k_i as *const _ as _);
    enc.set_bytes(8, size_of::<i32>() as u64, &ex_i as *const _ as _);
    enc.set_bytes(9, size_of::<i32>() as u64, &wd_i as *const _ as _);
    enc.set_bytes(10, size_of::<i32>() as u64, &fixed_stride_i as *const _ as _);
    enc.set_bytes(11, size_of::<i32>() as u64, &point_base as *const _ as _);
    let tg = pipeline_dim1_threadgroup_size(&ctx.pipeline_dim1);
    let groups = ((n_queries as u64) + tg - 1) / tg;
    enc.dispatch_thread_groups(
        metal::MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let idx_ptr = out_idx_gpu.contents() as *const i32;
    unsafe {
        std::ptr::copy_nonoverlapping(idx_ptr, out_idx.as_mut_ptr(), out_idx.len());
    }
    if return_distances {
        let dist_ptr = out_dist_gpu.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(dist_ptr, out_dist.as_mut_ptr(), out_dist.len());
        }
    }

    let mut edge_vals = Vec::new();
    let mut point_neighbor_counts = Vec::new();
    let mut dists_out = if return_distances {
        Some(Vec::new())
    } else {
        None
    };
    for tid in 0..n_queries {
        let local_p = query_info[tid * 4 + 3];
        let out_base = tid * k;
        for t in 0..k {
            edge_vals.push(local_p as i64);
            edge_vals.push(out_idx[out_base + t] as i64);
            if let Some(dv) = dists_out.as_mut() {
                dv.push(f64::from(out_dist[out_base + t]));
            }
        }
        point_neighbor_counts.push(k);
    }

    Ok(KnnDim1Result {
        edge_vals,
        point_neighbor_counts,
        group_point_counts,
        nn_dist: dists_out,
    })
}
