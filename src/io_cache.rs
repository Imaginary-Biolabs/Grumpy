//! Session-scoped I/O caches: pinned metadata buffers and optional chunk LRU.

use crate::error::io_failed;
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use zarrs::array::{Array, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableWritableListableStorage;

pub const DEFAULT_CHUNK_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Session I/O cache policy for lazy reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoCachePolicy {
    /// No caching; every read hits the store.
    None,
    /// Pin small metadata buffers (offsets, validity) for the session; leaf ranges are uncached.
    Metadata,
    /// Pin metadata and LRU-cache decoded Zarr chunks up to a byte budget.
    Chunks { budget_bytes: usize },
}

impl IoCachePolicy {
    pub fn parse(mode: &str, budget_bytes: usize) -> PyResult<Self> {
        match mode {
            "none" => Ok(Self::None),
            "metadata" => Ok(Self::Metadata),
            "chunks" => Ok(Self::Chunks {
                budget_bytes: budget_bytes.max(1),
            }),
            other => Err(crate::error::arg_invalid(
                "cache",
                format!("unknown cache mode '{other}'"),
                "use 'none', 'metadata', or 'chunks'.",
            )),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct ChunkKey {
    path: String,
    chunk_index: u64,
}

macro_rules! chunk_data_enum {
    ($($variant:ident($ty:ty)),* $(,)?) => {
        enum ChunkData {
            $($variant(Arc<Vec<$ty>>),)*
        }

        impl ChunkData {
            fn byte_len(&self) -> usize {
                match self {
                    $(ChunkData::$variant(v) => v.len() * std::mem::size_of::<$ty>(),)*
                }
            }
        }
    };
}

chunk_data_enum!(
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
);

struct ChunkLru {
    budget_bytes: usize,
    used_bytes: usize,
    order: VecDeque<ChunkKey>,
    entries: HashMap<ChunkKey, ChunkData>,
}

impl ChunkLru {
    fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes: budget_bytes.max(1),
            used_bytes: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.used_bytes = 0;
        self.order.clear();
        self.entries.clear();
    }

    fn stats(&self) -> (usize, usize) {
        (self.used_bytes, self.entries.len())
    }

    fn touch(&mut self, key: &ChunkKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }

    fn evict_one(&mut self) {
        let Some(key) = self.order.pop_front() else {
            return;
        };
        if let Some(entry) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.byte_len());
        }
    }

    fn insert(&mut self, key: ChunkKey, data: ChunkData) {
        let bytes = data.byte_len();
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.byte_len());
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
        }
        self.entries.insert(key.clone(), data);
        self.used_bytes += bytes;
        self.order.push_back(key);
        while self.used_bytes > self.budget_bytes {
            self.evict_one();
            if self.entries.is_empty() {
                break;
            }
        }
    }
}

macro_rules! chunk_lru_get {
    ($slf:expr, $key:expr, $variant:ident) => {{
        let hit = match $slf.entries.get($key) {
            Some(ChunkData::$variant(v)) => Some(Arc::clone(v)),
            _ => None,
        };
        if hit.is_some() {
            $slf.touch($key);
        }
        hit
    }};
}

impl ChunkLru {
    fn get_bool(&mut self, key: &ChunkKey) -> Option<Arc<Vec<bool>>> {
        chunk_lru_get!(self, key, Bool)
    }
    fn get_i8(&mut self, key: &ChunkKey) -> Option<Arc<Vec<i8>>> {
        chunk_lru_get!(self, key, I8)
    }
    fn get_i16(&mut self, key: &ChunkKey) -> Option<Arc<Vec<i16>>> {
        chunk_lru_get!(self, key, I16)
    }
    fn get_i32(&mut self, key: &ChunkKey) -> Option<Arc<Vec<i32>>> {
        chunk_lru_get!(self, key, I32)
    }
    fn get_i64(&mut self, key: &ChunkKey) -> Option<Arc<Vec<i64>>> {
        chunk_lru_get!(self, key, I64)
    }
    fn get_u8(&mut self, key: &ChunkKey) -> Option<Arc<Vec<u8>>> {
        chunk_lru_get!(self, key, U8)
    }
    fn get_u16(&mut self, key: &ChunkKey) -> Option<Arc<Vec<u16>>> {
        chunk_lru_get!(self, key, U16)
    }
    fn get_u32(&mut self, key: &ChunkKey) -> Option<Arc<Vec<u32>>> {
        chunk_lru_get!(self, key, U32)
    }
    fn get_u64(&mut self, key: &ChunkKey) -> Option<Arc<Vec<u64>>> {
        chunk_lru_get!(self, key, U64)
    }
    fn get_f32(&mut self, key: &ChunkKey) -> Option<Arc<Vec<f32>>> {
        chunk_lru_get!(self, key, F32)
    }
    fn get_f64(&mut self, key: &ChunkKey) -> Option<Arc<Vec<f64>>> {
        chunk_lru_get!(self, key, F64)
    }
    fn get_string(&mut self, key: &ChunkKey) -> Option<Arc<Vec<String>>> {
        chunk_lru_get!(self, key, String)
    }
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
    let v = arr.retrieve_array_subset_elements::<T>(&subset).map_err(|e| {
        io_failed(
            "I/O operation failed",
            format!("{e}"),
            "verify the saved dataset path and file permissions.",
        )
    })?;
    crate::io::record_io_bytes(v.len() * std::mem::size_of::<T>());
    Ok(v)
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

macro_rules! read_range_via_chunk_lru {
    ($ty:ty, $get:ident, $variant:ident, $fn_name:ident) => {
        fn $fn_name(
            store: &ReadableWritableListableStorage,
            path: &str,
            start: usize,
            stop: usize,
            lru: &Mutex<ChunkLru>,
        ) -> PyResult<Vec<$ty>> {
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
            let array_subset = ArraySubset::new_with_ranges(&[start as u64..stop as u64]);
            let chunks = arr
                .chunks_in_array_subset(&array_subset)
                .map_err(|e| {
                    io_failed(
                        "I/O operation failed",
                        format!("{e}"),
                        "verify the saved dataset path and file permissions.",
                    )
                })?
                .ok_or_else(|| {
                    io_failed(
                        "I/O operation failed",
                        "invalid array subset for chunk lookup",
                        "verify the saved dataset indices.",
                    )
                })?;

            let mut out: Vec<$ty> = Vec::with_capacity(stop - start);
            for chunk_indices in chunks.indices().into_iter() {
                let chunk_index = chunk_indices[0];
                let key = ChunkKey {
                    path: path.to_string(),
                    chunk_index,
                };
                let chunk_elems = {
                    let mut guard = lru.lock().unwrap();
                    if let Some(hit) = guard.$get(&key) {
                        hit
                    } else {
                        let elems = arr.retrieve_chunk_elements::<$ty>(&chunk_indices).map_err(|e| {
                            io_failed(
                                "I/O operation failed",
                                format!("{e}"),
                                "verify the saved dataset path and file permissions.",
                            )
                        })?;
                        crate::io::record_io_bytes(elems.len() * std::mem::size_of::<$ty>());
                        let arc = Arc::new(elems);
                        guard.insert(key.clone(), ChunkData::$variant(Arc::clone(&arc)));
                        arc
                    }
                };

                let chunk_subset = arr.chunk_subset(&chunk_indices).map_err(|e| {
                    io_failed(
                        "I/O operation failed",
                        format!("{e}"),
                        "verify the saved dataset path and file permissions.",
                    )
                })?;
                let chunk_origin = chunk_subset.start()[0] as usize;
                let copy_start = start.max(chunk_origin);
                let copy_end = stop.min(chunk_origin + chunk_elems.len());
                if copy_start < copy_end {
                    let src_off = copy_start - chunk_origin;
                    let len = copy_end - copy_start;
                    out.extend_from_slice(&chunk_elems[src_off..src_off + len]);
                }
            }
            Ok(out)
        }
    };
}

read_range_via_chunk_lru!(bool, get_bool, Bool, read_bool_range_via_chunk_lru);
read_range_via_chunk_lru!(i8, get_i8, I8, read_i8_range_via_chunk_lru);
read_range_via_chunk_lru!(i16, get_i16, I16, read_i16_range_via_chunk_lru);
read_range_via_chunk_lru!(i32, get_i32, I32, read_i32_range_via_chunk_lru);
read_range_via_chunk_lru!(i64, get_i64, I64, read_i64_range_via_chunk_lru);
read_range_via_chunk_lru!(u8, get_u8, U8, read_u8_range_via_chunk_lru);
read_range_via_chunk_lru!(u16, get_u16, U16, read_u16_range_via_chunk_lru);
read_range_via_chunk_lru!(u32, get_u32, U32, read_u32_range_via_chunk_lru);
read_range_via_chunk_lru!(u64, get_u64, U64, read_u64_range_via_chunk_lru);
read_range_via_chunk_lru!(f32, get_f32, F32, read_f32_range_via_chunk_lru);
read_range_via_chunk_lru!(f64, get_f64, F64, read_f64_range_via_chunk_lru);
read_range_via_chunk_lru!(String, get_string, String, read_string_range_via_chunk_lru);

macro_rules! meta_cache_field {
    ($ty:ty, $field:ident, $read:ident, $read_range:ident, $arc:ident, $chunk_read:ident) => {
        pub fn $arc(
            &self,
            store: &ReadableWritableListableStorage,
            path: &str,
        ) -> PyResult<Arc<Vec<$ty>>> {
            if matches!(self.policy, IoCachePolicy::None) {
                return Ok(Arc::new(read_vec_uncached::<$ty>(store, path)?));
            }
            let mut map = self.$field.lock().unwrap();
            if let Some(v) = map.get(path) {
                return Ok(Arc::clone(v));
            }
            let v = read_vec_uncached::<$ty>(store, path)?;
            let arc = Arc::new(v);
            map.insert(path.to_string(), Arc::clone(&arc));
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
            match self.policy {
                IoCachePolicy::Chunks { .. } => $chunk_read(store, path, start, stop, &self.chunk_lru),
                _ => read_vec_range_uncached::<$ty>(store, path, start, stop),
            }
        }
    };
}

pub struct IoCache {
    policy: IoCachePolicy,
    chunk_lru: Mutex<ChunkLru>,
    bool_meta: Mutex<HashMap<String, Arc<Vec<bool>>>>,
    i8_meta: Mutex<HashMap<String, Arc<Vec<i8>>>>,
    i16_meta: Mutex<HashMap<String, Arc<Vec<i16>>>>,
    i32_meta: Mutex<HashMap<String, Arc<Vec<i32>>>>,
    i64_meta: Mutex<HashMap<String, Arc<Vec<i64>>>>,
    u8_meta: Mutex<HashMap<String, Arc<Vec<u8>>>>,
    u16_meta: Mutex<HashMap<String, Arc<Vec<u16>>>>,
    u32_meta: Mutex<HashMap<String, Arc<Vec<u32>>>>,
    u64_meta: Mutex<HashMap<String, Arc<Vec<u64>>>>,
    f32_meta: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    f64_meta: Mutex<HashMap<String, Arc<Vec<f64>>>>,
    string_meta: Mutex<HashMap<String, Arc<Vec<String>>>>,
}

impl IoCache {
    pub fn new(policy: IoCachePolicy) -> Self {
        let budget = match policy {
            IoCachePolicy::Chunks { budget_bytes } => budget_bytes,
            _ => DEFAULT_CHUNK_BUDGET_BYTES,
        };
        Self {
            policy,
            chunk_lru: Mutex::new(ChunkLru::new(budget)),
            bool_meta: Mutex::new(HashMap::new()),
            i8_meta: Mutex::new(HashMap::new()),
            i16_meta: Mutex::new(HashMap::new()),
            i32_meta: Mutex::new(HashMap::new()),
            i64_meta: Mutex::new(HashMap::new()),
            u8_meta: Mutex::new(HashMap::new()),
            u16_meta: Mutex::new(HashMap::new()),
            u32_meta: Mutex::new(HashMap::new()),
            u64_meta: Mutex::new(HashMap::new()),
            f32_meta: Mutex::new(HashMap::new()),
            f64_meta: Mutex::new(HashMap::new()),
            string_meta: Mutex::new(HashMap::new()),
        }
    }

    pub fn policy(&self) -> IoCachePolicy {
        self.policy
    }

    pub fn clear(&self) {
        self.chunk_lru.lock().unwrap().clear();
        self.bool_meta.lock().unwrap().clear();
        self.i8_meta.lock().unwrap().clear();
        self.i16_meta.lock().unwrap().clear();
        self.i32_meta.lock().unwrap().clear();
        self.i64_meta.lock().unwrap().clear();
        self.u8_meta.lock().unwrap().clear();
        self.u16_meta.lock().unwrap().clear();
        self.u32_meta.lock().unwrap().clear();
        self.u64_meta.lock().unwrap().clear();
        self.f32_meta.lock().unwrap().clear();
        self.f64_meta.lock().unwrap().clear();
        self.string_meta.lock().unwrap().clear();
    }

    pub fn stats(&self) -> (usize, usize) {
        self.chunk_lru.lock().unwrap().stats()
    }

    meta_cache_field!(bool, bool_meta, read_bool, read_bool_range, read_bool_arc, read_bool_range_via_chunk_lru);
    meta_cache_field!(i8, i8_meta, read_i8, read_i8_range, read_i8_arc, read_i8_range_via_chunk_lru);
    meta_cache_field!(i16, i16_meta, read_i16, read_i16_range, read_i16_arc, read_i16_range_via_chunk_lru);
    meta_cache_field!(i32, i32_meta, read_i32, read_i32_range, read_i32_arc, read_i32_range_via_chunk_lru);
    meta_cache_field!(i64, i64_meta, read_i64, read_i64_range, read_i64_arc, read_i64_range_via_chunk_lru);
    meta_cache_field!(u8, u8_meta, read_u8, read_u8_range, read_u8_arc, read_u8_range_via_chunk_lru);
    meta_cache_field!(u16, u16_meta, read_u16, read_u16_range, read_u16_arc, read_u16_range_via_chunk_lru);
    meta_cache_field!(u32, u32_meta, read_u32, read_u32_range, read_u32_arc, read_u32_range_via_chunk_lru);
    meta_cache_field!(u64, u64_meta, read_u64, read_u64_range, read_u64_arc, read_u64_range_via_chunk_lru);
    meta_cache_field!(f32, f32_meta, read_f32, read_f32_range, read_f32_arc, read_f32_range_via_chunk_lru);
    meta_cache_field!(f64, f64_meta, read_f64, read_f64_range, read_f64_arc, read_f64_range_via_chunk_lru);
    meta_cache_field!(String, string_meta, read_string, read_string_range, read_string_arc, read_string_range_via_chunk_lru);
}

pub struct IoReader<'a> {
    pub store: &'a ReadableWritableListableStorage,
    pub cache: &'a IoCache,
}
