use crate::{ffi, DecodePosition, VertexDataAdapter};
use std::mem;

/// Reorders indices to reduce the number of GPU vertex shader invocations.
///
/// If index buffer contains multiple ranges for multiple draw calls,
/// this function needs to be called on each range individually.
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut optimized: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            optimized.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
        );
    }
    optimized
}

/// Reorders indices to reduce the number of GPU vertex shader invocations.
///
/// If index buffer contains multiple ranges for multiple draw calls,
/// this function needs to be called on each range individually.
pub fn optimize_vertex_cache_in_place(indices: &mut [u32], vertex_count: usize) {
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
        );
    }
}

/// Vertex transform cache optimizer for FIFO caches.
///
/// Reorders indices to reduce the number of GPU vertex shader invocations.
///
/// Generally takes ~3x less time to optimize meshes but produces inferior
/// results compared to `optimize_vertex_cache`.
///
/// If index buffer contains multiple ranges for multiple draw calls,
/// this function needs to be called on each range individually.
pub fn optimize_vertex_cache_fifo(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
) -> Vec<u32> {
    let mut optimized: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            optimized.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
    optimized
}

/// Vertex transform cache optimizer for FIFO caches (in place).
///
/// Reorders indices to reduce the number of GPU vertex shader invocations.
///
/// Generally takes ~3x less time to optimize meshes but produces inferior
/// results compared to `optimize_vertex_cache_fifo_in_place`.
///
/// If index buffer contains multiple ranges for multiple draw calls,
/// this function needs to be called on each range individually.
pub fn optimize_vertex_cache_fifo_in_place(
    indices: &mut [u32],
    vertex_count: usize,
    cache_size: u32,
) {
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
}

/// Reorders vertices and changes indices to reduce the amount of GPU
/// memory fetches during vertex processing.
///
/// This functions works for a single vertex stream; for multiple vertex streams,
/// use `optimize_vertex_fetch_remap` + `remap_vertex_buffer` for each stream.
///
/// `indices` is used both as an input and as an output index buffer.
pub fn optimize_vertex_fetch<T: Clone + Default>(indices: &mut [u32], vertices: &[T]) -> Vec<T> {
    let mut result: Vec<T> = vec![T::default(); vertices.len()];
    let next_vertex = unsafe {
        ffi::meshopt_optimizeVertexFetch(
            result.as_mut_ptr().cast(),
            indices.as_mut_ptr(),
            indices.len(),
            vertices.as_ptr().cast(),
            vertices.len(),
            mem::size_of::<T>(),
        )
    };
    result.resize(next_vertex, T::default());
    result
}

/// Vertex fetch cache optimizer (modifies in place)
/// Reorders vertices and changes indices to reduce the amount of GPU
/// memory fetches during vertex processing.
///
/// This functions works for a single vertex stream; for multiple vertex streams,
/// use `optimize_vertex_fetch_remap` + `remap_vertex_buffer` for each stream.
///
/// `indices` and `vertices` are used both as an input and as an output buffer.
pub fn optimize_vertex_fetch_in_place<T>(indices: &mut [u32], vertices: &mut [T]) -> usize {
    unsafe {
        ffi::meshopt_optimizeVertexFetch(
            vertices.as_mut_ptr().cast(),
            indices.as_mut_ptr(),
            indices.len(),
            vertices.as_ptr().cast(),
            vertices.len(),
            mem::size_of::<T>(),
        )
    }
}

/// Generates vertex remap to reduce the amount of GPU memory fetches during
/// vertex processing.
///
/// The resulting remap table should be used to reorder vertex/index buffers
/// using `optimize_remap_vertex_buffer`/`optimize_remap_index_buffer`.
pub fn optimize_vertex_fetch_remap(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; vertex_count];
    let next_vertex = unsafe {
        ffi::meshopt_optimizeVertexFetchRemap(
            result.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
        )
    };
    result.resize(next_vertex, 0u32);
    result
}

/// Reorders indices to reduce the number of GPU vertex shader invocations
/// and the pixel overdraw.
///
/// `indices` must contain index data that is the result of `optimize_vertex_cache`
/// (*not* the original mesh indices!)
///
/// `threshold` indicates how much the overdraw optimizer can degrade vertex cache
/// efficiency (1.05 = up to 5%) to reduce overdraw more efficiently.
pub fn optimize_overdraw_in_place(
    indices: &mut [u32],
    vertices: &VertexDataAdapter<'_>,
    threshold: f32,
) {
    unsafe {
        ffi::meshopt_optimizeOverdraw(
            indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            threshold,
        );
    }
}

/// Reorders indices to reduce the number of GPU vertex shader invocations
/// and the pixel overdraw.
///
/// `indices` must contain index data that is the result of `optimize_vertex_cache`
/// (*not* the original mesh indices!)
///
/// `threshold` indicates how much the overdraw optimizer can degrade vertex cache
/// efficiency (1.05 = up to 5%) to reduce overdraw more efficiently.
pub fn optimize_overdraw_in_place_decoder<T: DecodePosition>(
    indices: &mut [u32],
    vertices: &[T],
    threshold: f32,
) {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_optimizeOverdraw(
            indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            threshold,
        );
    }
}
