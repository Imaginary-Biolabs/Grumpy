//! kNN and radius neighbor search kernels.
//!
//! Performance model:
//! - dim=0 (single point cloud): adaptive strategy
//!   - brute-force for small `n` (avoids index build overhead)
//!   - kd-tree for larger `n` (improves asymptotics in low dimensions)
//! - dim=1 (grouped point clouds): brute-force within each group (parallelism comes from batch scheduling)
//!
//! Important correctness choices:
//! - Returns **edge indices** (source, target) suitable for graph construction
//! - Optionally also returns distances
//! - Stable ordering: sort by `(distance, index)`
//! - `loop=False` excludes self when query and data share storage
//! - Supports `OffsetView` batches produced by streaming

use crate::dtype::DType;
use crate::error::{
    arg_invalid, broadcast_union_outer_mismatch, dtype_mismatch, dtype_unsupported,
    index_out_of_bounds_simple, internal, layout_unsupported, shape_mismatch, unsupported,
};
use crate::layout::{drop_axis0_select_element, stack_axis0_broadcast, GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/// neighbors(query, data, k=..., radius=..., dim=..., loop=...)
///
/// Current supported layouts:
/// - dim=0: query/data are rectangular 2D point clouds: (n_points, d) represented as ListOffset->Leaf
/// - dim=1: query/data are grouped point clouds: (n_groups, n_points, d) represented as ListOffset->ListOffset->Leaf
///
/// Returns an edge index (source,target) suitable for graph construction.
///
/// - dim=0 kNN: shape (n_points, k, 2)
/// - dim=0 radius: shape (n_points, [*], 2)
/// - dim=1 kNN: shape (n_groups, n_points, k, 2) with **local** point indices within each group
/// - dim=1 radius: shape (n_groups, n_points, [*], 2)
pub fn neighbors(query: &GrumpyArray, data: &GrumpyArray, k: Option<usize>, radius: Option<f64>, dim: isize, loop_: bool) -> PyResult<GrumpyArray> {
    neighbors_with_gpu(query, data, k, radius, dim, loop_, crate::gpu::GpuPreference::Never)
}

pub fn neighbors_with_gpu(
    query: &GrumpyArray,
    data: &GrumpyArray,
    k: Option<usize>,
    radius: Option<f64>,
    dim: isize,
    loop_: bool,
    gpu: crate::gpu::GpuPreference,
) -> PyResult<GrumpyArray> {
    Ok(neighbors_edge_index_and_distances(query, data, k, radius, dim, loop_, false, gpu)?.0)
}

/// Internal: compute edge_index and optionally distances, used by Python bindings.
pub fn neighbors_edge_index_and_distances(
    query: &GrumpyArray,
    data: &GrumpyArray,
    k: Option<usize>,
    radius: Option<f64>,
    dim: isize,
    loop_: bool,
    return_distances: bool,
    gpu: crate::gpu::GpuPreference,
) -> PyResult<(GrumpyArray, Option<GrumpyArray>)> {
    if k.is_some() == radius.is_some() {
        return Err(arg_invalid(
            "k|radius",
            "exactly one of k or radius must be set",
            "pass neighbors(..., k=3) or neighbors(..., radius=1.5), not both or neither.",
        ));
    }
    if query.dtype != data.dtype {
        return Err(dtype_mismatch(data.dtype, query.dtype, "in neighbors(query, data)"));
    }
    if query.layout.has_union() || data.layout.has_union() {
        return neighbors_union(query, data, k, radius, dim, loop_, return_distances, gpu);
    }
    match dim {
        0 | -2 => neighbors_dim0_edges(query, data, k, radius, loop_, return_distances, gpu),
        1 | -3 => neighbors_dim1_edges(query, data, k, radius, loop_, return_distances, gpu),
        _ => Err(unsupported(
            "neighbors",
            "only dim=0 (single point cloud) or dim=1 (grouped point clouds) are supported",
            "pass dim=0 for (n_points, d) arrays or dim=1 for groups->points->coords.",
        )),
    }
}

fn neighbors_union(
    query: &GrumpyArray,
    data: &GrumpyArray,
    k: Option<usize>,
    radius: Option<f64>,
    dim: isize,
    loop_: bool,
    return_distances: bool,
    gpu: crate::gpu::GpuPreference,
) -> PyResult<(GrumpyArray, Option<GrumpyArray>)> {
    match dim {
        0 | -2 => {}
        _ => {
            return Err(unsupported(
                "neighbors",
                "on union layouts currently supports dim=0 only",
                "pass dim=0 for union point clouds.",
            ));
        }
    }
    let q_union = query.layout.has_union();
    let d_union = data.layout.has_union();
    let n = if q_union {
        query.len()
    } else if d_union {
        data.len()
    } else {
        unreachable!()
    };
    if q_union && d_union && query.len() != data.len() {
        return Err(broadcast_union_outer_mismatch(query.len(), data.len()));
    }
    let mut edge_layouts: Vec<Layout> = Vec::with_capacity(n);
    let mut dist_layouts: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let q_arr = if q_union {
            GrumpyArray {
                dtype: query.dtype,
                layout: drop_axis0_select_element(&query.layout, i)?,
            }
        } else {
            query.clone()
        };
        let d_arr = if d_union {
            GrumpyArray {
                dtype: data.dtype,
                layout: drop_axis0_select_element(&data.layout, i)?,
            }
        } else {
            data.clone()
        };
        if rect2d(&q_arr.layout).is_err() || rect2d(&d_arr.layout).is_err() {
            return Err(layout_unsupported(
                "neighbors",
                "each outer union element must be a rectangular 2D point cloud",
            ));
        }
        let (edge, dist) =
            neighbors_dim0_edges(&q_arr, &d_arr, k, radius, loop_, return_distances, gpu)?;
        edge_layouts.push(edge.layout);
        if return_distances {
            dist_layouts.push(dist.ok_or_else(|| {
                internal("neighbors", "missing distance output")
            })?.layout);
        }
    }
    let edge = GrumpyArray {
        dtype: DType::Int64,
        layout: stack_axis0_broadcast(&edge_layouts, DType::Int64)?,
    };
    let dist = if return_distances {
        Some(GrumpyArray {
            dtype: DType::Float64,
            layout: stack_axis0_broadcast(&dist_layouts, DType::Float64)?,
        })
    } else {
        None
    };
    Ok((edge, dist))
}

fn neighbors_dim0_edges(
    query: &GrumpyArray,
    data: &GrumpyArray,
    k: Option<usize>,
    radius: Option<f64>,
    loop_: bool,
    return_distances: bool,
    gpu: crate::gpu::GpuPreference,
) -> PyResult<(GrumpyArray, Option<GrumpyArray>)> {
    let (qoff, qbase, qleaf, qn, d) = rect2d(&query.layout)?;
    let (doff, dbase, dleaf, dn, dd) = rect2d(&data.layout)?;
    if d != dd {
        return Err(shape_mismatch(
            "neighbors",
            format!("point dimension mismatch between query and data ({d} vs {dd})"),
            "ensure query and data share the same coordinate dimension.",
        ));
    }
    if qleaf.has_nulls || dleaf.has_nulls {
        return Err(unsupported(
            "neighbors",
            "does not support null values yet",
            "filter nulls or use all-valid point clouds.",
        ));
    }
    let q = coords_as_f64(query.dtype, qleaf)?;
    let x = coords_as_f64(data.dtype, dleaf)?;
    let same_inputs = Arc::ptr_eq(qoff, doff) && qbase == dbase && leaf_shares_storage(qleaf, dleaf);

    if let Some(k) = k {
        if k == 0 {
            // Return empty neighbors axis: (qn, 0, 2)
            let edge = empty_knn_out(DType::Int64, qn, 2);
            let dist = if return_distances {
                Some(empty_knn_out(DType::Float64, qn, 0))
            } else {
                None
            };
            return Ok((edge, dist));
        }
        let max_k = if !loop_ && same_inputs { dn.saturating_sub(1) } else { dn };
        if k > max_k {
            return Err(arg_invalid(
                "k",
                format!("k={k} is larger than available points ({max_k})"),
                "reduce k or set loop=True when query and data differ.",
            ));
        }

        // Adaptive strategy:
        // - For small batches (common in training), brute-force is faster than building an index.
        // - For larger point clouds, use a kd-tree to avoid O(n^2).
        let use_kdtree = dn >= 2048 && d <= 32;
        if !use_kdtree && query.dtype == DType::Float64 && data.dtype == DType::Float64 {
            if let Some(gpu_res) = crate::gpu::knn_dim0_bruteforce(
                &q,
                &x,
                qoff,
                qbase,
                doff,
                dbase,
                qn,
                dn,
                d,
                k,
                loop_,
                same_inputs,
                return_distances,
                gpu,
            )? {
                let edge = build_knn_edge_index_dim0(qn, k, &gpu_res.nn_idx)?;
                let dist = if return_distances {
                    Some(build_knn_distances_dim0(qn, k, gpu_res.nn_dist.as_ref().ok_or_else(|| {
                        internal("neighbors", "GPU kNN missing distances")
                    })?)?)
                } else {
                    None
                };
                return Ok((edge, dist));
            }
        }
        if use_kdtree {
            let mut tree: KdTree<f64, usize, Vec<f64>> = KdTree::new(d);
            for j in 0..dn {
                let jstart = doff[dbase + j] as usize;
                tree.add(x[jstart..jstart + d].to_vec(), j)
                    .map_err(|e| internal("neighbors", format!("kd-tree add failed ({e})")))?;
            }
            // Note: kd-tree build is done; query in parallel if large enough.
            let extra = if !loop_ && same_inputs { 1 } else { 0 };
            let kk = k + extra;

            let mut nn_idx: Vec<usize> = vec![0; qn * k];
            let mut nn_dist: Vec<f64> = if return_distances { vec![0.0; qn * k] } else { Vec::new() };
            let use_par = qn.saturating_mul(dn) >= 200_000 && rayon::current_num_threads() > 1;
            if use_par {
                if return_distances {
                    nn_idx
                        .par_chunks_mut(k)
                        .zip(nn_dist.par_chunks_mut(k))
                        .enumerate()
                        .try_for_each(|(qi, (out_row, out_d))| -> Result<(), PyErr> {
                            let qstart = qoff[qbase + qi] as usize;
                            let qpt = &q[qstart..qstart + d];
                            let res = tree
                                .nearest(qpt, kk, &squared_euclidean)
                                .map_err(|e| internal("neighbors", format!("kd-tree query failed ({e})")))?;
                            let mut cand: Vec<(f64, usize)> = Vec::with_capacity(res.len());
                            for (dist, &j) in res.iter() {
                                if !loop_ && same_inputs && qi == j {
                                    continue;
                                }
                                cand.push((*dist, j));
                            }
                            cand.sort_by(|(d1, i1), (d2, i2)| {
                                d1.partial_cmp(d2).unwrap().then(i1.cmp(i2))
                            });
                            for t in 0..k {
                                out_row[t] = cand[t].1;
                                out_d[t] = cand[t].0.sqrt();
                            }
                            Ok(())
                        })?;
                } else {
                    nn_idx
                        .par_chunks_mut(k)
                        .enumerate()
                        .try_for_each(|(qi, out_row)| -> Result<(), PyErr> {
                        let qstart = qoff[qbase + qi] as usize;
                        let qpt = &q[qstart..qstart + d];
                        let res = tree
                            .nearest(qpt, kk, &squared_euclidean)
                            .map_err(|e| internal("neighbors", format!("kd-tree query failed ({e})")))?;
                        // Convert to (dist, idx) and enforce stable ordering.
                        let mut cand: Vec<(f64, usize)> = Vec::with_capacity(res.len());
                        for (dist, &j) in res.iter() {
                            if !loop_ && same_inputs && qi == j {
                                continue;
                            }
                            cand.push((*dist, j));
                        }
                        cand.sort_by(|(d1, i1), (d2, i2)| d1.partial_cmp(d2).unwrap().then(i1.cmp(i2)));
                        for t in 0..k {
                            out_row[t] = cand[t].1;
                        }
                        Ok(())
                    })?;
                }
            } else {
                for qi in 0..qn {
                    let qstart = qoff[qbase + qi] as usize;
                    let qpt = &q[qstart..qstart + d];
                    let res = tree
                        .nearest(qpt, kk, &squared_euclidean)
                        .map_err(|e| internal("neighbors", format!("kd-tree query failed ({e})")))?;
                    let mut cand: Vec<(f64, usize)> = Vec::with_capacity(res.len());
                    for (dist, &j) in res.iter() {
                        if !loop_ && same_inputs && qi == j {
                            continue;
                        }
                        cand.push((*dist, j));
                    }
                    cand.sort_by(|(d1, i1), (d2, i2)| d1.partial_cmp(d2).unwrap().then(i1.cmp(i2)));
                    for t in 0..k {
                        nn_idx[qi * k + t] = cand[t].1;
                        if return_distances {
                            nn_dist[qi * k + t] = cand[t].0.sqrt();
                        }
                    }
                }
            }
            let edge = build_knn_edge_index_dim0(qn, k, &nn_idx)?;
            let dist = if return_distances {
                Some(build_knn_distances_dim0(qn, k, &nn_dist)?)
            } else {
                None
            };
            return Ok((edge, dist));
        }

        let mut nn_idx: Vec<usize> = vec![0; qn * k];
        let mut nn_dist: Vec<f64> = if return_distances { vec![0.0; qn * k] } else { Vec::new() };
        let use_par = qn.saturating_mul(dn) >= 200_000 && rayon::current_num_threads() > 1;
        if use_par {
            if return_distances {
                nn_idx
                    .par_chunks_mut(k)
                    .zip(nn_dist.par_chunks_mut(k))
                    .enumerate()
                    .for_each(|(qi, (out_row, out_d))| {
                        let qstart = qoff[qbase + qi] as usize;
                        // max-heap via linear scan of current worst
                        let mut best_d2: Vec<f64> = vec![f64::INFINITY; k];
                        let mut best_j: Vec<usize> = vec![0; k];
                        for j in 0..dn {
                            if !loop_ && same_inputs && qi == j {
                                continue;
                            }
                            let jstart = doff[dbase + j] as usize;
                            let d2 = dist2(&q[qstart..qstart + d], &x[jstart..jstart + d]);
                            // insert if better than worst
                            let mut worst = 0usize;
                            let mut worst_val = best_d2[0];
                            for t in 1..k {
                                if best_d2[t] > worst_val {
                                    worst_val = best_d2[t];
                                    worst = t;
                                }
                            }
                            if d2 < worst_val || (d2 == worst_val && j < best_j[worst]) {
                                best_d2[worst] = d2;
                                best_j[worst] = j;
                            }
                        }
                        // sort by (dist, index) for stable output order
                        let mut ord: Vec<usize> = (0..k).collect();
                        ord.sort_by(|&a, &b| {
                            best_d2[a]
                                .partial_cmp(&best_d2[b])
                                .unwrap()
                                .then(best_j[a].cmp(&best_j[b]))
                        });
                        for t in 0..k {
                            out_row[t] = best_j[ord[t]];
                            out_d[t] = best_d2[ord[t]].sqrt();
                        }
                    });
            } else {
                nn_idx
                    .par_chunks_mut(k)
                    .enumerate()
                    .for_each(|(qi, out_row)| {
                    let qstart = qoff[qbase + qi] as usize;
                    // max-heap via linear scan of current worst
                    let mut best_d2: Vec<f64> = vec![f64::INFINITY; k];
                    let mut best_j: Vec<usize> = vec![0; k];
                    for j in 0..dn {
                        if !loop_ && same_inputs && qi == j {
                            continue;
                        }
                        let jstart = doff[dbase + j] as usize;
                        let d2 = dist2(&q[qstart..qstart + d], &x[jstart..jstart + d]);
                        // insert if better than worst
                        let mut worst = 0usize;
                        let mut worst_val = best_d2[0];
                        for t in 1..k {
                            if best_d2[t] > worst_val {
                                worst_val = best_d2[t];
                                worst = t;
                            }
                        }
                        if d2 < worst_val || (d2 == worst_val && j < best_j[worst]) {
                            best_d2[worst] = d2;
                            best_j[worst] = j;
                        }
                    }
                    // sort by (dist, index) for stable output order
                    let mut ord: Vec<usize> = (0..k).collect();
                    ord.sort_by(|&a, &b| {
                        best_d2[a]
                            .partial_cmp(&best_d2[b])
                            .unwrap()
                            .then(best_j[a].cmp(&best_j[b]))
                    });
                    for t in 0..k {
                        out_row[t] = best_j[ord[t]];
                    }
                });
            }
        } else {
            for qi in 0..qn {
                let qstart = qoff[qbase + qi] as usize;
                // max-heap via linear scan of current worst
                let mut best_d2: Vec<f64> = vec![f64::INFINITY; k];
                let mut best_j: Vec<usize> = vec![0; k];
                for j in 0..dn {
                    if !loop_ && same_inputs && qi == j {
                        continue;
                    }
                    let jstart = doff[dbase + j] as usize;
                    let d2 = dist2(&q[qstart..qstart + d], &x[jstart..jstart + d]);
                    // insert if better than worst
                    let mut worst = 0usize;
                    let mut worst_val = best_d2[0];
                    for t in 1..k {
                        if best_d2[t] > worst_val {
                            worst_val = best_d2[t];
                            worst = t;
                        }
                    }
                    if d2 < worst_val || (d2 == worst_val && j < best_j[worst]) {
                        best_d2[worst] = d2;
                        best_j[worst] = j;
                    }
                }
                // sort by (dist, index) for stable output order
                let mut ord: Vec<usize> = (0..k).collect();
                ord.sort_by(|&a, &b| {
                    best_d2[a]
                        .partial_cmp(&best_d2[b])
                        .unwrap()
                        .then(best_j[a].cmp(&best_j[b]))
                });
                for t in 0..k {
                    nn_idx[qi * k + t] = best_j[ord[t]];
                    if return_distances {
                        nn_dist[qi * k + t] = best_d2[ord[t]].sqrt();
                    }
                }
            }
        }
        let edge = build_knn_edge_index_dim0(qn, k, &nn_idx)?;
        let dist = if return_distances {
            Some(build_knn_distances_dim0(qn, k, &nn_dist)?)
        } else {
            None
        };
        return Ok((edge, dist));
    }

    let r = radius.unwrap();
    if r < 0.0 {
        return Err(arg_invalid("radius", "must be non-negative", "pass radius >= 0."));
    }
    let r2 = r * r;
    let use_kdtree = dn >= 2048 && d <= 32;
    let use_par = qn.saturating_mul(dn) >= 200_000 && rayon::current_num_threads() > 1;
    let mut per_q_counts: Vec<usize> = vec![0; qn];
    let mut all_idx: Vec<usize> = Vec::new();
    let mut all_dist: Vec<f64> = Vec::new();
    if use_kdtree {
        let mut tree: KdTree<f64, usize, Vec<f64>> = KdTree::new(d);
        for j in 0..dn {
            let jstart = doff[dbase + j] as usize;
            tree.add(x[jstart..jstart + d].to_vec(), j)
                .map_err(|e| internal("neighbors", format!("kd-tree add failed ({e})")))?;
        }
        let per_lists: Vec<Vec<(usize, f64)>> = (0..qn)
            .into_par_iter()
            .map(|qi| {
                let qstart = qoff[qbase + qi] as usize;
                let qpt = &q[qstart..qstart + d];
                let res = tree
                    .within(qpt, r2, &squared_euclidean)
                    .map_err(|e| internal("neighbors", format!("kd-tree query failed ({e})")))?;
                let mut cand: Vec<(f64, usize)> = Vec::with_capacity(res.len());
                for (dist, &j) in res.iter() {
                    if !loop_ && same_inputs && qi == j {
                        continue;
                    }
                    cand.push((*dist, j));
                }
                cand.sort_by(|(d1, i1), (d2, i2)| d1.partial_cmp(d2).unwrap().then(i1.cmp(i2)));
                Ok::<_, PyErr>(cand.into_iter().map(|(d2, j)| (j, d2.sqrt())).collect())
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (qi, lst) in per_lists.into_iter().enumerate() {
            per_q_counts[qi] = lst.len();
            all_idx.extend(lst.iter().map(|(j, _)| *j));
            if return_distances {
                all_dist.extend(lst.into_iter().map(|(_, dd)| dd));
            }
        }
    } else if use_par {
        // Compute per-query neighbor lists in parallel, then flatten in order.
        let per_lists: Vec<Vec<(usize, f64)>> = (0..qn)
            .into_par_iter()
            .map(|qi| {
                let qstart = qoff[qbase + qi] as usize;
                let mut idxs: Vec<(usize, f64)> = Vec::new();
                for j in 0..dn {
                    if !loop_ && same_inputs && qi == j {
                        continue;
                    }
                    let jstart = doff[dbase + j] as usize;
                    let d2 = dist2(&q[qstart..qstart + d], &x[jstart..jstart + d]);
                    if d2 <= r2 {
                        idxs.push((j, d2));
                    }
                }
                idxs.sort_by(|(i1, d1), (i2, d2)| d1.partial_cmp(d2).unwrap().then(i1.cmp(i2)));
                idxs.into_iter().map(|(j, d2)| (j, d2.sqrt())).collect()
            })
            .collect();
        for (qi, lst) in per_lists.into_iter().enumerate() {
            per_q_counts[qi] = lst.len();
            all_idx.extend(lst.iter().map(|(j, _)| *j));
            if return_distances {
                all_dist.extend(lst.into_iter().map(|(_, dd)| dd));
            }
        }
    } else {
        for qi in 0..qn {
            let qstart = qoff[qbase + qi] as usize;
            let mut idxs: Vec<(usize, f64)> = Vec::new();
            for j in 0..dn {
                if !loop_ && same_inputs && qi == j {
                    continue;
                }
                let jstart = doff[dbase + j] as usize;
                let d2 = dist2(&q[qstart..qstart + d], &x[jstart..jstart + d]);
                if d2 <= r2 {
                    idxs.push((j, d2));
                }
            }
            idxs.sort_by(|(i1, d1), (i2, d2)| d1.partial_cmp(d2).unwrap().then(i1.cmp(i2)));
            per_q_counts[qi] = idxs.len();
            for (j, d2) in idxs {
                all_idx.push(j);
                if return_distances {
                    all_dist.push(d2.sqrt());
                }
            }
        }
    }
    let edge = build_radius_edge_index_dim0(&per_q_counts, &all_idx)?;
    let dist = if return_distances {
        Some(build_radius_distances_dim0(&per_q_counts, &all_dist)?)
    } else {
        None
    };
    Ok((edge, dist))
}

fn neighbors_dim1_edges(
    query: &GrumpyArray,
    data: &GrumpyArray,
    k: Option<usize>,
    radius: Option<f64>,
    loop_: bool,
    return_distances: bool,
    gpu: crate::gpu::GpuPreference,
) -> PyResult<(GrumpyArray, Option<GrumpyArray>)> {
    // Layout: groups -> points -> coords
    let (qg_off, qg_base, qp, qleaf, qg_n, qd) = grouped_points(&query.layout)?;
    let (dg_off, dg_base, dp, dleaf, dg_n, dd) = grouped_points(&data.layout)?;
    if qd != dd {
        return Err(shape_mismatch(
            "neighbors(dim=1)",
            format!("point dimension mismatch ({qd} vs {dd})"),
            "ensure query and data share the same coordinate dimension.",
        ));
    }
    if qg_n != dg_n {
        return Err(shape_mismatch(
            "neighbors(dim=1)",
            "number of groups must match between query and data",
            "ensure both arrays have the same number of groups.",
        ));
    }

    if qleaf.has_nulls || dleaf.has_nulls {
        return Err(unsupported(
            "neighbors",
            "does not support null values yet",
            "filter nulls or use all-valid point clouds.",
        ));
    }
    let qcoords = coords_as_f64(query.dtype, qleaf)?;
    let dcoords = coords_as_f64(data.dtype, dleaf)?;
    let same_inputs =
        Arc::ptr_eq(qg_off, dg_off) && qg_base == dg_base && Arc::ptr_eq(&qp.offsets, &dp.offsets) && leaf_shares_storage(qleaf, dleaf);

    if let Some(k) = k {
        if k == 0 {
            // groups->points->0->2
        // Reconstruct outer group offsets for empty output (relative, contiguous).
        let mut out_group_offsets: Vec<i64> = Vec::with_capacity(qg_n + 1);
        out_group_offsets.push(0);
        for g in 0..qg_n {
            let qps = qg_off[qg_base + g] as i64;
            let qpe = qg_off[qg_base + g + 1] as i64;
            out_group_offsets.push(*out_group_offsets.last().unwrap() + (qpe - qps));
        }
            let out_point_offsets: Vec<i64> = vec![0; qp.len() + 1];
            let out_neighbor_offsets: Vec<i64> = vec![0];
            let out_vals_i64: Vec<i64> = Vec::new();
            let edge = build_grouped_edge_index(
                qg_n,
                out_group_offsets.clone(),
                out_point_offsets.clone(),
                out_neighbor_offsets,
                out_vals_i64,
            )?;
            let dist = if return_distances {
                Some(build_grouped_distances(qg_n, out_group_offsets, out_point_offsets, Vec::new())?)
            } else {
                None
            };
            return Ok((edge, dist));
        }
        if query.dtype == DType::Float64 && data.dtype == DType::Float64 {
            if let Some(gpu_res) = crate::gpu::knn_dim1_bruteforce(
                &qcoords,
                &dcoords,
                qg_off,
                qg_base,
                &qp.offsets,
                qg_n,
                qd,
                k,
                loop_,
                same_inputs,
                return_distances,
                gpu,
            )? {
                return grouped_knn_from_gpu(gpu_res, qg_n, k, return_distances);
            }
        }
        // We compute flat indices into data points within each group.
        // For group g: points are in [dp.offsets[g]..dp.offsets[g+1]) in the *points* listoffset space.
        // Each point has qd scalars contiguously in leaf.
        let mut out_group_offsets: Vec<i64> = Vec::with_capacity(qg_n + 1);
        out_group_offsets.push(0);
        let mut out_point_offsets: Vec<i64> = Vec::new();
        out_point_offsets.push(0);
        let mut out_neighbor_offsets: Vec<i64> = Vec::new();
        out_neighbor_offsets.push(0);
        let mut out_vals_i64: Vec<i64> = Vec::new();
        let mut out_dists: Vec<f64> = Vec::new();

        for g in 0..qg_n {
            // group boundaries are in the group offsets, indexing points
            let qps = qg_off[qg_base + g] as usize;
            let qpe = qg_off[qg_base + g + 1] as usize;
            let dps = dg_off[dg_base + g] as usize;
            let dpe = dg_off[dg_base + g + 1] as usize;
            let nqp = qpe - qps;
            let ndp = dpe - dps;
            // points offsets for this group (in leaf scalar index space)
            for p in 0..nqp {
                // each query point p in group g
                let qi_point = qps + p;
                let qstart = qg_point_scalar_start(qp, qi_point)?;

                let mut best_d2: Vec<f64> = vec![f64::INFINITY; k];
                let mut best_j: Vec<usize> = vec![0; k];
                for j_local in 0..ndp {
                    let j_point = dps + j_local;
                    if !loop_ && same_inputs && qi_point == j_point {
                        continue;
                    }
                    let jstart = qg_point_scalar_start(dp, j_point)?;
                    let d2 = dist2(&qcoords[qstart..qstart + qd], &dcoords[jstart..jstart + qd]);
                    let mut worst = 0usize;
                    let mut worst_val = best_d2[0];
                    for t in 1..k {
                        if best_d2[t] > worst_val {
                            worst_val = best_d2[t];
                            worst = t;
                        }
                    }
                    if d2 < worst_val || (d2 == worst_val && j_local < best_j[worst]) {
                        best_d2[worst] = d2;
                        best_j[worst] = j_local;
                    }
                }
                let mut ord: Vec<usize> = (0..k).collect();
                ord.sort_by(|&a, &b| best_d2[a].partial_cmp(&best_d2[b]).unwrap().then(best_j[a].cmp(&best_j[b])));
                // append k neighbors as edges (src, dst) with local indices
                for t in 0..k {
                    let src_local = (qi_point - qps) as i64;
                    let dst_local = best_j[ord[t]] as i64;
                    out_vals_i64.push(src_local);
                    out_vals_i64.push(dst_local);
                    out_neighbor_offsets.push(*out_neighbor_offsets.last().unwrap() + 2);
                    if return_distances {
                        out_dists.push(best_d2[ord[t]].sqrt());
                    }
                }
                out_point_offsets.push(*out_point_offsets.last().unwrap() + (k as i64));
            }
            out_group_offsets.push(*out_group_offsets.last().unwrap() + (nqp as i64));
        }
        let edge = build_grouped_edge_index(qg_n, out_group_offsets.clone(), out_point_offsets.clone(), out_neighbor_offsets, out_vals_i64)?;
        let dist = if return_distances {
            Some(build_grouped_distances(qg_n, out_group_offsets, out_point_offsets, out_dists)?)
        } else {
            None
        };
        return Ok((edge, dist));
    }

    let r = radius.unwrap();
    if r < 0.0 {
        return Err(arg_invalid("radius", "must be non-negative", "pass radius >= 0."));
    }
    let r2 = r * r;
    let mut out_group_offsets: Vec<i64> = Vec::with_capacity(qg_n + 1);
    out_group_offsets.push(0);
    let mut out_point_offsets: Vec<i64> = vec![0];
    let mut out_neighbor_offsets: Vec<i64> = vec![0];
    let mut out_vals_i64: Vec<i64> = Vec::new();
    let mut out_dists: Vec<f64> = Vec::new();

    for g in 0..qg_n {
        let qps = qg_off[qg_base + g] as usize;
        let qpe = qg_off[qg_base + g + 1] as usize;
        let dps = dg_off[dg_base + g] as usize;
        let dpe = dg_off[dg_base + g + 1] as usize;
        let nqp = qpe - qps;
        let ndp = dpe - dps;
        for p in 0..nqp {
            let qi_point = qps + p;
            let qstart = qg_point_scalar_start(qp, qi_point)?;
            let mut idxs: Vec<(usize, f64)> = Vec::new();
            for j_local in 0..ndp {
                let j_point = dps + j_local;
                if !loop_ && same_inputs && qi_point == j_point {
                    continue;
                }
                let jstart = qg_point_scalar_start(dp, j_point)?;
                let d2 = dist2(&qcoords[qstart..qstart + qd], &dcoords[jstart..jstart + qd]);
                if d2 <= r2 {
                    idxs.push((j_local, d2));
                }
            }
            idxs.sort_by(|(i1, d1), (i2, d2)| d1.partial_cmp(d2).unwrap().then(i1.cmp(i2)));
            for (j_local, _) in idxs.iter() {
                let j_point = dps + *j_local;
                let src_local = (qi_point - qps) as i64;
                let dst_local = *j_local as i64;
                out_vals_i64.push(src_local);
                out_vals_i64.push(dst_local);
                out_neighbor_offsets.push(*out_neighbor_offsets.last().unwrap() + 2);
                if return_distances {
                    let jstart = qg_point_scalar_start(dp, j_point)?;
                    let d2 = dist2(&qcoords[qstart..qstart + qd], &dcoords[jstart..jstart + qd]);
                    out_dists.push(d2.sqrt());
                }
            }
            out_point_offsets.push(*out_point_offsets.last().unwrap() + (idxs.len() as i64));
        }
        out_group_offsets.push(*out_group_offsets.last().unwrap() + (nqp as i64));
    }
    let edge = build_grouped_edge_index(qg_n, out_group_offsets.clone(), out_point_offsets.clone(), out_neighbor_offsets, out_vals_i64)?;
    let dist = if return_distances {
        Some(build_grouped_distances(qg_n, out_group_offsets, out_point_offsets, out_dists)?)
    } else {
        None
    };
    Ok((edge, dist))
}

// ---------- output builders ----------

fn grouped_knn_from_gpu(
    gpu: crate::gpu::knn::KnnDim1Result,
    n_groups: usize,
    k: usize,
    return_distances: bool,
) -> PyResult<(GrumpyArray, Option<GrumpyArray>)> {
    let mut out_group_offsets: Vec<i64> = Vec::with_capacity(n_groups + 1);
    out_group_offsets.push(0);
    let mut out_point_offsets: Vec<i64> = Vec::new();
    out_point_offsets.push(0);
    let mut out_neighbor_offsets: Vec<i64> = Vec::new();
    out_neighbor_offsets.push(0);
    for &nqp in &gpu.group_point_counts {
        for _ in 0..nqp {
            for _ in 0..k {
                out_neighbor_offsets.push(out_neighbor_offsets.last().unwrap() + 2);
            }
            out_point_offsets.push(out_point_offsets.last().unwrap() + k as i64);
        }
        out_group_offsets.push(out_group_offsets.last().unwrap() + nqp as i64);
    }
    let edge = build_grouped_edge_index(
        n_groups,
        out_group_offsets.clone(),
        out_point_offsets.clone(),
        out_neighbor_offsets,
        gpu.edge_vals,
    )?;
    let dist = if return_distances {
        Some(build_grouped_distances(
            n_groups,
            out_group_offsets,
            out_point_offsets,
            gpu.nn_dist.ok_or_else(|| internal("neighbors", "GPU kNN missing distances"))?,
        )?)
    } else {
        None
    };
    Ok((edge, dist))
}

fn build_knn_edge_index_dim0(qn: usize, k: usize, nn_idx: &[usize]) -> PyResult<GrumpyArray> {
    // Layout: qn -> k -> 2
    let mut off_q: Vec<i64> = Vec::with_capacity(qn + 1);
    off_q.push(0);
    for i in 0..qn {
        off_q.push((i as i64 + 1) * k as i64);
    }
    let n_edges = qn * k;
    let mut off_e: Vec<i64> = Vec::with_capacity(n_edges + 1);
    off_e.push(0);
    for i in 0..n_edges {
        off_e.push((i as i64 + 1) * 2);
    }
    let mut vals: Vec<i64> = Vec::with_capacity(n_edges * 2);
    for src in 0..qn {
        for t in 0..k {
            let dst = nn_idx[src * k + t] as i64;
            vals.push(src as i64);
            vals.push(dst);
        }
    }
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = vals.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len]);
    leaf.buffer = LeafBuffer::I64(Arc::new(vals));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(off_q),
        content: Box::new(Layout::ListOffset(ListOffset {
            offsets: Arc::new(off_e),
            content: Box::new(Layout::Leaf(leaf)),
        })),
    });
    Ok(GrumpyArray { dtype: DType::Int64, layout })
}

fn build_knn_distances_dim0(qn: usize, k: usize, dists: &[f64]) -> PyResult<GrumpyArray> {
    let mut off_q: Vec<i64> = Vec::with_capacity(qn + 1);
    off_q.push(0);
    for i in 0..qn {
        off_q.push((i as i64 + 1) * k as i64);
    }
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = qn * k;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len]);
    leaf.buffer = LeafBuffer::F64(Arc::new(dists.to_vec()));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(off_q),
        content: Box::new(Layout::Leaf(leaf)),
    });
    Ok(GrumpyArray { dtype: DType::Float64, layout })
}

fn build_radius_edge_index_dim0(per_q_counts: &[usize], all_idx: &[usize]) -> PyResult<GrumpyArray> {
    let qn = per_q_counts.len();
    let mut off_q: Vec<i64> = Vec::with_capacity(qn + 1);
    off_q.push(0);
    let mut acc = 0i64;
    for &c in per_q_counts {
        acc += c as i64;
        off_q.push(acc);
    }
    let n_edges = *off_q.last().unwrap() as usize;
    let mut off_e: Vec<i64> = Vec::with_capacity(n_edges + 1);
    off_e.push(0);
    for i in 0..n_edges {
        off_e.push((i as i64 + 1) * 2);
    }
    let mut vals: Vec<i64> = Vec::with_capacity(n_edges * 2);
    let mut pos = 0usize;
    for src in 0..qn {
        for _ in 0..per_q_counts[src] {
            let dst = all_idx[pos] as i64;
            vals.push(src as i64);
            vals.push(dst);
            pos += 1;
        }
    }
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = vals.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len]);
    leaf.buffer = LeafBuffer::I64(Arc::new(vals));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(off_q),
        content: Box::new(Layout::ListOffset(ListOffset {
            offsets: Arc::new(off_e),
            content: Box::new(Layout::Leaf(leaf)),
        })),
    });
    Ok(GrumpyArray { dtype: DType::Int64, layout })
}

fn build_radius_distances_dim0(per_q_counts: &[usize], all_dist: &[f64]) -> PyResult<GrumpyArray> {
    let qn = per_q_counts.len();
    let mut off_q: Vec<i64> = Vec::with_capacity(qn + 1);
    off_q.push(0);
    let mut acc = 0i64;
    for &c in per_q_counts {
        acc += c as i64;
        off_q.push(acc);
    }
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = *off_q.last().unwrap() as usize;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len]);
    leaf.buffer = LeafBuffer::F64(Arc::new(all_dist.to_vec()));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(off_q),
        content: Box::new(Layout::Leaf(leaf)),
    });
    Ok(GrumpyArray { dtype: DType::Float64, layout })
}

fn build_grouped_edge_index(
    n_groups: usize,
    off_g: Vec<i64>,
    off_p: Vec<i64>,
    off_n: Vec<i64>,
    vals: Vec<i64>,
) -> PyResult<GrumpyArray> {
    // groups -> points -> neighbors -> 2
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = vals.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len]);
    leaf.buffer = LeafBuffer::I64(Arc::new(vals));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(off_g),
        content: Box::new(Layout::ListOffset(ListOffset {
            offsets: Arc::new(off_p),
            content: Box::new(Layout::ListOffset(ListOffset {
                offsets: Arc::new(off_n),
                content: Box::new(Layout::Leaf(leaf)),
            })),
        })),
    });
    let _ = n_groups;
    Ok(GrumpyArray { dtype: DType::Int64, layout })
}

fn build_grouped_distances(
    _n_groups: usize,
    off_g: Vec<i64>,
    off_p: Vec<i64>,
    vals: Vec<f64>,
) -> PyResult<GrumpyArray> {
    // groups -> points -> neighbors
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = vals.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len]);
    leaf.buffer = LeafBuffer::F64(Arc::new(vals));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(off_g),
        content: Box::new(Layout::ListOffset(ListOffset {
            offsets: Arc::new(off_p),
            content: Box::new(Layout::Leaf(leaf)),
        })),
    });
    Ok(GrumpyArray { dtype: DType::Float64, layout })
}

fn leaf_shares_storage(a: &Leaf, b: &Leaf) -> bool {
    if !Arc::ptr_eq(&a.validity, &b.validity) {
        return false;
    }
    leafbuffer_shares_storage(&a.buffer, &b.buffer)
}

fn leafbuffer_shares_storage(a: &LeafBuffer, b: &LeafBuffer) -> bool {
    match (a, b) {
        (LeafBuffer::I32(aa), LeafBuffer::I32(bb)) => Arc::ptr_eq(aa, bb),
        (LeafBuffer::I64(aa), LeafBuffer::I64(bb)) => Arc::ptr_eq(aa, bb),
        (LeafBuffer::F32(aa), LeafBuffer::F32(bb)) => Arc::ptr_eq(aa, bb),
        (LeafBuffer::F64(aa), LeafBuffer::F64(bb)) => Arc::ptr_eq(aa, bb),
        (LeafBuffer::U32(aa), LeafBuffer::U32(bb)) => Arc::ptr_eq(aa, bb),
        (LeafBuffer::U64(aa), LeafBuffer::U64(bb)) => Arc::ptr_eq(aa, bb),
        _ => false,
    }
}

fn build_nested_coords(dtype: DType, off0: Vec<i64>, off1: Vec<i64>, vals_f64: Vec<f64>) -> GrumpyArray {
    let leaf = leaf_from_f64(dtype, vals_f64);
    let inner = Layout::ListOffset(ListOffset { offsets: Arc::new(off1), content: Box::new(Layout::Leaf(leaf)) });
    GrumpyArray { dtype, layout: Layout::ListOffset(ListOffset { offsets: Arc::new(off0), content: Box::new(inner) }) }
}

fn empty_knn_out(dtype: DType, qn: usize, _d: usize) -> GrumpyArray {
    let off_q: Vec<i64> = (0..=qn as i64).collect();
    let off_k: Vec<i64> = vec![0];
    build_nested_coords(dtype, off_q, off_k, Vec::new())
}

fn leaf_from_f64(dtype: DType, vals: Vec<f64>) -> Leaf {
    let n = vals.len();
    let mut leaf = Leaf::new(dtype);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = match dtype {
        DType::Float64 => LeafBuffer::F64(Arc::new(vals)),
        DType::Float32 => LeafBuffer::F32(Arc::new(vals.into_iter().map(|x| x as f32).collect())),
        DType::Int32 => LeafBuffer::I32(Arc::new(vals.into_iter().map(|x| x as i32).collect())),
        DType::Int64 => LeafBuffer::I64(Arc::new(vals.into_iter().map(|x| x as i64).collect())),
        _ => LeafBuffer::F64(Arc::new(vals)),
    };
    leaf
}

// ---------- layout parsing ----------

fn rect2d<'a>(layout: &'a Layout) -> PyResult<(&'a Arc<Vec<i64>>, usize, &'a Leaf, usize, usize)> {
    match layout {
        Layout::ListOffset(lo) => {
            let leaf = match lo.content.as_ref() {
                Layout::Leaf(l) => l,
                _ => return Err(layout_unsupported("neighbors", "expected 2D list->leaf array")),
            };
            let n = lo.len();
            if n == 0 {
                return Ok((&lo.offsets, 0, leaf, 0, 0));
            }
            let d = (lo.offsets[1] - lo.offsets[0]) as usize;
            for i in 0..n {
                let len = (lo.offsets[i + 1] - lo.offsets[i]) as usize;
                if len != d {
                    return Err(shape_mismatch(
                        "neighbors",
                        "expected rectangular 2D array (constant row length)",
                        "ensure every row has the same length.",
                    ));
                }
            }
            Ok((&lo.offsets, 0, leaf, n, d))
        }
        Layout::OffsetView(v) => {
            let leaf = match v.content.as_ref() {
                Layout::Leaf(l) => l,
                _ => return Err(layout_unsupported("neighbors", "expected 2D list->leaf array")),
            };
            let n = v.len();
            if n == 0 {
                return Ok((&v.offsets, v.start, leaf, 0, 0));
            }
            if v.start + n >= v.offsets.len() {
                return Err(index_out_of_bounds_simple("on OffsetView in neighbors"));
            }
            let d = (v.offsets[v.start + 1] - v.offsets[v.start]) as usize;
            for i in 0..n {
                let abs = v.start + i;
                let len = (v.offsets[abs + 1] - v.offsets[abs]) as usize;
                if len != d {
                    return Err(shape_mismatch(
                        "neighbors",
                        "expected rectangular 2D array (constant row length)",
                        "ensure every row has the same length.",
                    ));
                }
            }
            Ok((&v.offsets, v.start, leaf, n, d))
        }
        Layout::Indexed(ix) => rect2d(ix.content.as_ref()),
        _ => Err(layout_unsupported("neighbors", "expected 2D list->leaf array")),
    }
}

fn grouped_points<'a>(
    layout: &'a Layout,
) -> PyResult<(&'a Arc<Vec<i64>>, usize, &'a ListOffset, &'a Leaf, usize, usize)> {
    // groups (ListOffset/OffsetView) -> points (ListOffset) -> coords (Leaf with fixed dim per point)
    let (g_off, g_base, g_content, ng) = match layout {
        Layout::ListOffset(g) => (&g.offsets, 0usize, g.content.as_ref(), g.len()),
        Layout::OffsetView(v) => (&v.offsets, v.start, v.content.as_ref(), v.len()),
        Layout::Indexed(ix) => return grouped_points(ix.content.as_ref()),
        _ => return Err(layout_unsupported(
            "neighbors",
            "expected grouped point cloud: groups->points->coords",
        )),
    };
    let p = match g_content {
        Layout::ListOffset(p) => p,
        _ => return Err(layout_unsupported(
            "neighbors",
            "expected grouped point cloud: groups->points->coords",
        )),
    };
    let leaf = match p.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => return Err(layout_unsupported(
            "neighbors",
            "expected grouped point cloud: points->leaf coords",
        )),
    };
    // Determine point dimension from first point (requires at least one point)
    if p.len() == 0 {
        return Ok((g_off, g_base, p, leaf, ng, 0));
    }
    // For points listoffset, offsets give cumulative scalars per point.
    let d = (p.offsets[1] - p.offsets[0]) as usize;
    for i in 0..p.len() {
        let len = (p.offsets[i + 1] - p.offsets[i]) as usize;
        if len != d {
            return Err(shape_mismatch(
                "neighbors",
                "expected fixed coordinate dimension for each point (rectangular points)",
                "ensure every point has the same number of coordinates.",
            ));
        }
    }
    Ok((g_off, g_base, p, leaf, ng, d))
}

fn qg_point_scalar_start(p: &ListOffset, point_index: usize) -> PyResult<usize> {
    Ok(p.offsets[point_index] as usize)
}

fn coords_as_f64(dtype: DType, leaf: &Leaf) -> PyResult<Vec<f64>> {
    Ok(match (&leaf.buffer, dtype) {
        (LeafBuffer::F64(v), _) => v.iter().cloned().collect(),
        (LeafBuffer::F32(v), _) => v.iter().map(|&x| x as f64).collect(),
        (LeafBuffer::I32(v), _) => v.iter().map(|&x| x as f64).collect(),
        (LeafBuffer::I64(v), _) => v.iter().map(|&x| x as f64).collect(),
        _ => return Err(dtype_unsupported("neighbors coords", dtype)),
    })
}

#[inline]
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}


