//! Per-dataset I/O caches: path-persistent leaf buffers and shared offset arrays.

use crate::error::io_failed;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use zarrs::array::{Array, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableWritableListableStorage;

/// After this many range reads touch the same path, promote to a full in-memory buffer.
const PROMOTE_TOUCHES: u32 = 2;

#[derive(Default)]
struct LeafSlot<T> {
    full: Option<Arc<Vec<T>>>,
    touches: u32,
}

macro_rules! leaf_cache_field {
    ($ty:ty, $field:ident, $read:ident, $read_range:ident, $arc:ident) => {
        pub fn $arc(
            &self,
            store: &ReadableWritableListableStorage,
            path: &str,
        ) -> PyResult<Arc<Vec<$ty>>> {
            let mut map = self.$field.lock().unwrap();
            if let Some(slot) = map.get(path) {
                if let Some(full) = &slot.full {
                    return Ok(Arc::clone(full));
                }
            }
            let v = read_vec_uncached::<$ty>(store, path)?;
            crate::io::record_io_bytes(v.len() * std::mem::size_of::<$ty>());
            let arc = Arc::new(v);
            map.insert(
                path.to_string(),
                LeafSlot {
                    full: Some(Arc::clone(&arc)),
                    touches: PROMOTE_TOUCHES,
                },
            );
            Ok(arc)
        }

        pub fn $read(
            &self,
            store: &ReadableWritableListableStorage,
            path: &str,
        ) -> PyResult<Vec<$ty>> {
            Ok(self.$arc(store, path)?.as_ref().clone())
        }

        pub fn $read_range(
            &self,
            store: &ReadableWritableListableStorage,
            path: &str,
            start: usize,
            stop: usize,
        ) -> PyResult<Vec<$ty>> {
            if start >= stop {
                return Ok(Vec::new());
            }
            let mut map = self.$field.lock().unwrap();
            if let Some(slot) = map.get(path) {
                if let Some(full) = &slot.full {
                    return Ok(full[start..stop].to_vec());
                }
            }
            let slice = read_vec_range_uncached::<$ty>(store, path, start, stop)?;
            let slot = map
                .entry(path.to_string())
                .or_insert_with(LeafSlot::default);
            slot.touches += 1;
            if slot.touches >= PROMOTE_TOUCHES {
                let v = read_vec_uncached::<$ty>(store, path)?;
                crate::io::record_io_bytes(v.len() * std::mem::size_of::<$ty>());
                slot.full = Some(Arc::new(v));
            }
            Ok(slice)
        }
    };
}

#[derive(Default)]
pub struct IoCache {
    bool_leaf: Mutex<HashMap<String, LeafSlot<bool>>>,
    i8_leaf: Mutex<HashMap<String, LeafSlot<i8>>>,
    i16_leaf: Mutex<HashMap<String, LeafSlot<i16>>>,
    i32_leaf: Mutex<HashMap<String, LeafSlot<i32>>>,
    i64_leaf: Mutex<HashMap<String, LeafSlot<i64>>>,
    u8_leaf: Mutex<HashMap<String, LeafSlot<u8>>>,
    u16_leaf: Mutex<HashMap<String, LeafSlot<u16>>>,
    u32_leaf: Mutex<HashMap<String, LeafSlot<u32>>>,
    u64_leaf: Mutex<HashMap<String, LeafSlot<u64>>>,
    f32_leaf: Mutex<HashMap<String, LeafSlot<f32>>>,
    f64_leaf: Mutex<HashMap<String, LeafSlot<f64>>>,
    string_leaf: Mutex<HashMap<String, LeafSlot<String>>>,
}

impl IoCache {
    leaf_cache_field!(bool, bool_leaf, read_bool, read_bool_range, read_bool_arc);
    leaf_cache_field!(i8, i8_leaf, read_i8, read_i8_range, read_i8_arc);
    leaf_cache_field!(i16, i16_leaf, read_i16, read_i16_range, read_i16_arc);
    leaf_cache_field!(i32, i32_leaf, read_i32, read_i32_range, read_i32_arc);
    leaf_cache_field!(i64, i64_leaf, read_i64, read_i64_range, read_i64_arc);
    leaf_cache_field!(u8, u8_leaf, read_u8, read_u8_range, read_u8_arc);
    leaf_cache_field!(u16, u16_leaf, read_u16, read_u16_range, read_u16_arc);
    leaf_cache_field!(u32, u32_leaf, read_u32, read_u32_range, read_u32_arc);
    leaf_cache_field!(u64, u64_leaf, read_u64, read_u64_range, read_u64_arc);
    leaf_cache_field!(f32, f32_leaf, read_f32, read_f32_range, read_f32_arc);
    leaf_cache_field!(f64, f64_leaf, read_f64, read_f64_range, read_f64_arc);
    leaf_cache_field!(String, string_leaf, read_string, read_string_range, read_string_arc);
}

pub struct IoReader<'a> {
    pub store: &'a ReadableWritableListableStorage,
    pub cache: &'a IoCache,
}

fn read_vec_uncached<T: ElementOwned>(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> PyResult<Vec<T>> {
    let arr = Array::open(store.clone(), path).map_err(|e| {
        io_failed(
            "I/O operation failed",
            format!("{e}"),
            "verify the saved dataset path and file permissions.",
        )
    })?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset_elements::<T>(&subset).map_err(|e| {
        io_failed(
            "I/O operation failed",
            format!("{e}"),
            "verify the saved dataset path and file permissions.",
        )
    })
}

fn read_vec_range_uncached<T: ElementOwned>(
    store: &ReadableWritableListableStorage,
    path: &str,
    start: usize,
    stop: usize,
) -> PyResult<Vec<T>> {
    if start >= stop {
        return Ok(Vec::new());
    }
    let arr = Array::open(store.clone(), path).map_err(|e| {
        io_failed(
            "I/O operation failed",
            format!("{e}"),
            "verify the saved dataset path and file permissions.",
        )
    })?;
    let subset = ArraySubset::new_with_ranges(&[start as u64..stop as u64]);
    crate::io::record_io_bytes((stop - start) * std::mem::size_of::<T>());
    arr.retrieve_array_subset_elements::<T>(&subset).map_err(|e| {
        io_failed(
            "I/O operation failed",
            format!("{e}"),
            "verify the saved dataset path and file permissions.",
        )
    })
}
