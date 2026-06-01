use crate::io as io_ops;
use crate::stream::{self, BatchPayload, BatchPlan};
use crate::py_api::types::{PyGrumpyArray, PyGrumpyDataFrame, PyStreamBatchesIter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

fn payload_to_py(py: Python<'_>, payload: BatchPayload, is_dataframe: bool) -> PyResult<PyObject> {
    match payload {
        BatchPayload::Array(arr) if !is_dataframe => {
            Ok(Py::new(py, PyGrumpyArray { inner: arr })?.into_py(py))
        }
        BatchPayload::DataFrame(df) => {
            Ok(Py::new(py, PyGrumpyDataFrame { inner: df })?.into_py(py))
        }
        BatchPayload::Array(_) => Err(PyValueError::new_err(
            "Internal error: array payload for dataframe stream.",
        )),
    }
}

pub(crate) fn prepare_stream_plan_with_shuffle_level(
    path: &str,
    batch_size: usize,
    drop_last: bool,
    batch_on: Option<&str>,
    shuffle: Option<&str>,
    seed: Option<u64>,
    world_size: usize,
    rank: usize,
    batch_indices: Option<&[usize]>,
) -> PyResult<(io_ops::DatasetHandle, BatchPlan, bool, Option<String>, Option<u64>)> {
    let handle = io_ops::DatasetHandle::open(path)?;
    let is_df = matches!(handle.meta.root, io_ops::RootMeta::DataFrame { .. });
    let mut plan = stream::build_batch_plan(&handle, batch_size, drop_last, batch_on)?;
    let shuffle_within = shuffle.and_then(|s| {
        if s == "true" || s == "batch" {
            None
        } else {
            handle
                .schema()
                .and_then(|schema| schema.level_index(s).ok().map(|_| s.to_string()))
        }
    });
    if shuffle.is_some() {
        stream::shuffle_batch_plan(&mut plan, seed.unwrap_or(0));
    }
    if world_size > 1 || rank > 0 {
        stream::shard_batch_plan(&mut plan, world_size.max(1), rank)?;
    }
    if let Some(indices) = batch_indices {
        plan = stream::filter_batch_plan(&plan, indices)?;
    }
    Ok((handle, plan, is_df, shuffle_within, seed))
}

pub(crate) fn spawn_prefetch_loader(
    handle: io_ops::DatasetHandle,
    plan: BatchPlan,
    queue_depth: usize,
) -> Receiver<PyResult<BatchPayload>> {
    let (tx, rx): (SyncSender<PyResult<BatchPayload>>, Receiver<PyResult<BatchPayload>>) =
        mpsc::sync_channel(queue_depth.max(1));
    thread::spawn(move || {
        for batch in plan.batches {
            let result = stream::load_batch(&handle, &batch);
            if tx.send(result).is_err() {
                break;
            }
        }
    });
    rx
}

#[pymethods]
impl PyStreamBatchesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __len__(&self) -> usize {
        self.plan.batches.len()
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if slf.pos >= slf.plan.batches.len() {
            return Ok(None);
        }
        let batch_idx = slf.pos;
        let mut payload = if let Some(rx) = slf.prefetch_rx.as_ref() {
            rx.recv()
                .map_err(|e| PyValueError::new_err(format!("Prefetch loader failed: {e}")))??
        } else {
            let batch = &slf.plan.batches[batch_idx];
            stream::load_batch(&slf.handle, batch)?
        };
        if let (Some(level), Some(seed)) = (&slf.shuffle_within, slf.seed) {
            stream::shuffle_within_batch(
                &mut payload,
                level,
                &slf.handle,
                seed.wrapping_add(batch_idx as u64),
            )?;
        }
        slf.pos += 1;
        Ok(Some(payload_to_py(py, payload, slf.is_dataframe)?))
    }
}

#[pyfunction]
#[pyo3(signature = (path, batch_size, drop_last=false, batch_on=None, shuffle=None, seed=None, workers=0, world_size=1, rank=0, batch_indices=None))]
pub fn stream_batches(
    path: String,
    batch_size: usize,
    drop_last: bool,
    batch_on: Option<String>,
    shuffle: Option<String>,
    seed: Option<u64>,
    workers: usize,
    world_size: usize,
    rank: usize,
    batch_indices: Option<Vec<usize>>,
) -> PyResult<PyStreamBatchesIter> {
    let (handle, plan, is_dataframe, shuffle_within, seed) = prepare_stream_plan_with_shuffle_level(
        &path,
        batch_size,
        drop_last,
        batch_on.as_deref(),
        shuffle.as_deref(),
        seed,
        world_size,
        rank,
        batch_indices.as_deref(),
    )?;
    let prefetch_rx = if workers > 0 {
        Some(spawn_prefetch_loader(handle.clone(), plan.clone(), workers))
    } else {
        None
    };
    Ok(PyStreamBatchesIter {
        handle,
        is_dataframe,
        plan,
        pos: 0,
        shuffle_within,
        seed,
        prefetch_rx,
    })
}

#[pyfunction]
#[pyo3(signature = (path, batch_size, drop_last=false, batch_on=None, world_size=1, rank=0, batch_indices=None))]
pub fn stream_len(
    path: String,
    batch_size: usize,
    drop_last: bool,
    batch_on: Option<String>,
    world_size: usize,
    rank: usize,
    batch_indices: Option<Vec<usize>>,
) -> PyResult<usize> {
    let handle = io_ops::DatasetHandle::open(&path)?;
    let mut plan = stream::build_batch_plan(&handle, batch_size, drop_last, batch_on.as_deref())?;
    if world_size > 1 || rank > 0 {
        stream::shard_batch_plan(&mut plan, world_size.max(1), rank)?;
    }
    if let Some(indices) = batch_indices {
        plan = stream::filter_batch_plan(&plan, &indices)?;
    }
    Ok(plan.batches.len())
}
