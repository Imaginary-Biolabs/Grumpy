//! ML dataloader batch planning, shuffle, and DDP sharding for saved datasets.

use crate::error::{arg_invalid, arg_must_be_positive, index_out_of_bounds, schema_violation};
use crate::io::{self, DatasetHandle, RootMeta};
use crate::random::GrumpyRng;
use pyo3::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchRange {
    pub start: usize,
    pub stop: usize,
}

#[derive(Clone, Debug)]
pub struct BatchPlan {
    pub batches: Vec<BatchRange>,
}

pub fn build_batch_plan(
    handle: &DatasetHandle,
    batch_size: usize,
    drop_last: bool,
    batch_on: Option<&str>,
) -> PyResult<BatchPlan> {
    if batch_size == 0 {
        return Err(arg_must_be_positive("batch_size", batch_size));
    }
    let n = handle.axis0_len()?;
    if n == 0 {
        return Ok(BatchPlan { batches: Vec::new() });
    }
    let batches = match batch_on {
        None => build_axis0_batches(n, batch_size, drop_last),
        Some(level) => {
            let depth = resolve_batch_on_depth(handle, level)?;
            let layout = layout_meta_for_batch_on(handle, level)?;
            let counts = io::row_entity_counts_at_depth(&handle.reader(), layout, depth)?;
            build_entity_batches(&counts, batch_size, drop_last)
        }
    };
    Ok(BatchPlan { batches })
}

fn layout_meta_for_batch_on<'a>(
    handle: &'a DatasetHandle,
    level: &str,
) -> PyResult<&'a io::LayoutMeta> {
    match &handle.meta.root {
        io::RootMeta::Array { layout, .. } => Ok(layout),
        io::RootMeta::DataFrame { columns, .. } => {
            if let Some(c) = columns.iter().find(|c| {
                c.name == level || c.name.starts_with(&format!("{level}_"))
            }) {
                return Ok(&c.layout);
            }
            columns
                .first()
                .map(|c| &c.layout)
                .ok_or_else(|| {
                    schema_violation(
                        "empty dataframe has no columns",
                        "batch_on requires at least one column to infer layout.",
                        "save a non-empty dataframe or use batch_on=None for axis-0 batching.",
                    )
                })
        }
    }
}

fn resolve_batch_on_depth(handle: &DatasetHandle, level: &str) -> PyResult<usize> {
    if let Some(schema) = handle.schema() {
        schema.level_index(level)
    } else {
        // Arrays without schema: interpret batch_on as numeric depth string.
        level.parse::<usize>().map_err(|_| {
            arg_invalid(
                "batch_on",
                format!("unknown batch_on level '{level}'"),
                "use a schema level name or a numeric depth for arrays without schema.",
            )
        })
    }
}

fn build_axis0_batches(n: usize, batch_size: usize, drop_last: bool) -> Vec<BatchRange> {
    let end = if drop_last && n % batch_size != 0 {
        n - (n % batch_size)
    } else {
        n
    };
    let mut batches = Vec::new();
    let mut i = 0usize;
    while i < end {
        let j = (i + batch_size).min(end);
        batches.push(BatchRange { start: i, stop: j });
        i = j;
    }
    batches
}

/// Greedy packing of axis-0 rows until entity count at ``batch_on`` depth reaches ``batch_size``.
fn build_entity_batches(counts: &[usize], batch_size: usize, drop_last: bool) -> Vec<BatchRange> {
    let mut batches = Vec::new();
    let mut i = 0usize;
    while i < counts.len() {
        let mut acc = 0usize;
        let start = i;
        while i < counts.len() && (acc == 0 || acc < batch_size) {
            acc += counts[i];
            i += 1;
        }
        if acc == 0 {
            break;
        }
        if drop_last && i == counts.len() && acc < batch_size && !batches.is_empty() {
            // drop trailing partial batch
            break;
        }
        batches.push(BatchRange { start, stop: i });
    }
    batches
}

pub fn shuffle_batch_plan(plan: &mut BatchPlan, seed: u64) {
    if plan.batches.len() <= 1 {
        return;
    }
    let mut rng = GrumpyRng::new(seed);
    let mut idx: Vec<usize> = (0..plan.batches.len()).collect();
    rng.shuffle_usizes(&mut idx);
    let old = std::mem::take(&mut plan.batches);
    plan.batches = idx.into_iter().map(|i| old[i].clone()).collect();
}

pub fn shard_batch_plan(plan: &mut BatchPlan, world_size: usize, rank: usize) -> PyResult<()> {
    if world_size == 0 {
        return Err(arg_must_be_positive("world_size", world_size));
    }
    if rank >= world_size {
        return Err(arg_invalid(
            "rank",
            format!("got {rank}, expected < world_size={world_size}"),
            "pass rank in [0, world_size).",
        ));
    }
    plan.batches = plan
        .batches
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            if i % world_size == rank {
                Some(b.clone())
            } else {
                None
            }
        })
        .collect();
    Ok(())
}

pub fn load_batch(handle: &DatasetHandle, batch: &BatchRange) -> PyResult<BatchPayload> {
    match &handle.meta.root {
        RootMeta::Array { .. } => {
            let arr = io::load_array_axis0_slice(handle, batch.start, batch.stop)?;
            Ok(BatchPayload::Array(arr))
        }
        RootMeta::DataFrame { .. } => {
            let df = io::load_dataframe_axis0_slice(handle, batch.start, batch.stop)?;
            Ok(BatchPayload::DataFrame(df))
        }
    }
}

#[derive(Clone, Debug)]
pub enum BatchPayload {
    Array(crate::layout::GrumpyArray),
    DataFrame(crate::dataframe::GrumpyDataFrame),
}

pub fn shuffle_within_batch(
    payload: &mut BatchPayload,
    shuffle_level: &str,
    handle: &DatasetHandle,
    seed: u64,
) -> PyResult<()> {
    let depth = resolve_batch_on_depth(handle, shuffle_level)?;
    let dim = depth as isize;
    let mut rng = GrumpyRng::new(seed);
    match payload {
        BatchPayload::Array(arr) => {
            crate::random::shuffle(&mut rng, arr, dim)?;
        }
        BatchPayload::DataFrame(df) => {
            for col in df.cols.iter_mut() {
                crate::random::shuffle(&mut rng, col, dim)?;
            }
        }
    }
    Ok(())
}

pub fn filter_batch_plan(plan: &BatchPlan, indices: &[usize]) -> PyResult<BatchPlan> {
    let n = plan.batches.len();
    let mut batches = Vec::with_capacity(indices.len());
    for &i in indices {
        if i >= n {
            return Err(index_out_of_bounds(i, n, "for batch plan indices"));
        }
        batches.push(plan.batches[i].clone());
    }
    Ok(BatchPlan { batches })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis0_batches_drop_last() {
        let b = build_axis0_batches(10, 4, true);
        assert_eq!(b, vec![BatchRange { start: 0, stop: 4 }, BatchRange { start: 4, stop: 8 }]);
    }

    #[test]
    fn entity_batches_greedy() {
        let counts = vec![3, 2, 5, 1];
        let b = build_entity_batches(&counts, 4, false);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], BatchRange { start: 0, stop: 2 }); // 3+2=5 >= 4
        assert_eq!(b[1], BatchRange { start: 2, stop: 4 });
    }
}
