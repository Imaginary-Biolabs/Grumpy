//! Structure geometry kernels (pairwise distances, grid pooling / voxelization).

pub mod grid_pool;
pub mod pairwise;
pub(crate) mod points;

pub use grid_pool::grid_pool;
pub use pairwise::pairwise_distances;
