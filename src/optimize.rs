use crate::utilities::{check_indices, check_triangles, check_vertex_size};
use crate::{DecodePosition, VertexDataAdapter, ffi};
use bytemuck::Pod;
use std::mem;

/// Panics unless `indices` is a triangle list referring to `vertex_count` vertices that the
/// index reordering functions can process: they count indices and triangles using 32 bits.
#[track_caller]
fn check_reorder_input(indices: &[u32], vertex_count: usize) {
    assert!(
        u32::try_from(indices.len()).is_ok(),
        "index count ({}) must fit in 32 bits",
        indices.len()
    );
    check_triangles(indices, vertex_count);
}

/// Reorders indices to reduce the number of GPU vertex shader invocations.
///
/// If index buffer contains multiple ranges for multiple draw calls,
/// this function needs to be called on each range individually.
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    check_reorder_input(indices, vertex_count);
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
    check_reorder_input(indices, vertex_count);
    // the library copies the indices when the destination is the source
    let indices_ptr = indices.as_mut_ptr();
    unsafe {
        ffi::meshopt_optimizeVertexCache(indices_ptr, indices_ptr, indices.len(), vertex_count);
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
///
/// `cache_size` is raised to 3 if it is smaller.
pub fn optimize_vertex_cache_fifo(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
) -> Vec<u32> {
    check_reorder_input(indices, vertex_count);
    let cache_size = cache_size.max(3);
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
///
/// `cache_size` is raised to 3 if it is smaller.
pub fn optimize_vertex_cache_fifo_in_place(
    indices: &mut [u32],
    vertex_count: usize,
    cache_size: u32,
) {
    check_reorder_input(indices, vertex_count);
    let cache_size = cache_size.max(3);
    // the library copies the indices when the destination is the source
    let indices_ptr = indices.as_mut_ptr();
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            indices_ptr,
            indices_ptr,
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
}

/// Vertex transform cache optimizer for strip-like caches.
///
/// Produces inferior results to `optimize_vertex_cache` from the GPU vertex cache perspective.
/// However, the resulting index order is more optimal if the goal is to reduce the triangle strip
/// length or improve compression efficiency.
pub fn optimize_vertex_cache_strip(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    check_reorder_input(indices, vertex_count);
    let mut optimized: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_optimizeVertexCacheStrip(
            optimized.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
        );
    }
    optimized
}

/// Vertex transform cache optimizer for strip-like caches (in place).
///
/// Produces inferior results to `optimize_vertex_cache_in_place` from the GPU vertex cache perspective.
/// However, the resulting index order is more optimal if the goal is to reduce the triangle strip
/// length or improve compression efficiency.
pub fn optimize_vertex_cache_strip_in_place(indices: &mut [u32], vertex_count: usize) {
    check_reorder_input(indices, vertex_count);
    // the library copies the indices when the destination is the source
    let indices_ptr = indices.as_mut_ptr();
    unsafe {
        ffi::meshopt_optimizeVertexCacheStrip(
            indices_ptr,
            indices_ptr,
            indices.len(),
            vertex_count,
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
///
/// Vertices are copied bytewise, so `T` must be plain data, see [`bytemuck::Pod`].
pub fn optimize_vertex_fetch<T: Pod>(indices: &mut [u32], vertices: &[T]) -> Vec<T> {
    check_vertex_size::<T>();
    check_indices(indices, vertices.len());
    let mut result: Vec<T> = vec![T::zeroed(); vertices.len()];
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
    // each referenced vertex is written once, so this never grows the result
    result.truncate(next_vertex);
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
/// Returns the number of vertices that `indices` refers to; these are moved to the front
/// of `vertices`, and the remaining vertices are left with unspecified contents.
///
/// Vertices are copied bytewise, so `T` must be plain data, see [`bytemuck::Pod`].
pub fn optimize_vertex_fetch_in_place<T: Pod>(indices: &mut [u32], vertices: &mut [T]) -> usize {
    check_vertex_size::<T>();
    check_indices(indices, vertices.len());
    // the library copies the vertices when the destination is the source
    let vertices_ptr = vertices.as_mut_ptr();
    unsafe {
        ffi::meshopt_optimizeVertexFetch(
            vertices_ptr.cast(),
            indices.as_mut_ptr(),
            indices.len(),
            vertices_ptr.cast_const().cast(),
            vertices.len(),
            mem::size_of::<T>(),
        )
    }
}

/// Generates vertex remap to reduce the amount of GPU memory fetches during
/// vertex processing.
///
/// The resulting remap table should be used to reorder vertex/index buffers
/// using `remap_vertex_buffer`/`remap_index_buffer`.
///
/// The table has an entry for each of the `vertex_count` vertices; vertices that `indices`
/// doesn't refer to are marked with `u32::MAX`.
pub fn optimize_vertex_fetch_remap(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    check_indices(indices, vertex_count);
    let mut result: Vec<u32> = vec![0; vertex_count];
    unsafe {
        ffi::meshopt_optimizeVertexFetchRemap(
            result.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
        );
    }
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
    check_reorder_input(indices, vertices.vertex_count);
    // the library copies the indices when the destination is the source
    let indices_ptr = indices.as_mut_ptr();
    unsafe {
        ffi::meshopt_optimizeOverdraw(
            indices_ptr,
            indices_ptr,
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
    check_reorder_input(indices, vertices.len());
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    // the library copies the indices when the destination is the source
    let indices_ptr = indices.as_mut_ptr();
    unsafe {
        ffi::meshopt_optimizeOverdraw(
            indices_ptr,
            indices_ptr,
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            threshold,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    // 3x3 vertex grid split into 8 triangles, in scrambled order
    #[rustfmt::skip]
    const GRID: [u32; 24] = [
        4, 8, 5, 0, 4, 1, 3, 7, 4, 1, 5, 2,
        4, 7, 8, 1, 4, 5, 0, 3, 4, 3, 6, 7,
    ];

    fn grid_positions() -> Vec<[f32; 3]> {
        (0..9)
            .map(|i| [(i % 3) as f32, (i / 3) as f32, 0.0])
            .collect()
    }

    /// Triangles in a canonical order, to compare index buffers that only differ by reordering.
    fn sorted(indices: &[u32]) -> Vec<u32> {
        let mut triangles: Vec<&[u32]> = indices.chunks(3).collect();
        triangles.sort_unstable();
        triangles.concat()
    }

    #[test]
    fn test_optimize_vertex_cache_strip() {
        let optimized = optimize_vertex_cache_strip(&GRID, 9);

        let mut in_place = GRID;
        optimize_vertex_cache_strip_in_place(&mut in_place, 9);
        assert_eq!(optimized, in_place);

        // same triangles, possibly reordered
        assert_eq!(sorted(&optimized), sorted(&GRID));
    }

    #[test]
    fn test_optimize_vertex_cache_in_place() {
        let mut in_place = GRID;
        optimize_vertex_cache_in_place(&mut in_place, 9);
        assert_eq!(optimize_vertex_cache(&GRID, 9), in_place);
        assert_eq!(sorted(&in_place), sorted(&GRID));

        let mut in_place = GRID;
        optimize_vertex_cache_fifo_in_place(&mut in_place, 9, 16);
        assert_eq!(optimize_vertex_cache_fifo(&GRID, 9, 16), in_place);
        assert_eq!(sorted(&in_place), sorted(&GRID));
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_optimize_vertex_cache_index_out_of_range() {
        optimize_vertex_cache(&GRID, 8);
    }

    #[test]
    fn test_optimize_vertex_fetch() {
        // vertex 1 is not referenced
        let vertices = [[0u32; 2], [1; 2], [2; 2], [3; 2]];
        let indices = [3, 0, 2, 2, 0, 3];

        let mut out_indices = indices;
        let result = optimize_vertex_fetch(&mut out_indices, &vertices);
        assert_eq!(result, [[3; 2], [0; 2], [2; 2]]);
        assert_eq!(out_indices, [0, 1, 2, 2, 1, 0]);

        let mut in_place_indices = indices;
        let mut in_place = vertices;
        let count = optimize_vertex_fetch_in_place(&mut in_place_indices, &mut in_place);
        assert_eq!(in_place[..count], result[..]);
        assert_eq!(in_place_indices, out_indices);
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_optimize_vertex_fetch_index_out_of_range() {
        optimize_vertex_fetch(&mut [0, 1, 2], &[0u32; 2]);
    }

    #[test]
    #[should_panic(expected = "vertex size")]
    fn test_optimize_vertex_fetch_large_vertex() {
        optimize_vertex_fetch(&mut [0, 0, 0], &[[0u8; 512]]);
    }

    #[test]
    fn test_optimize_vertex_fetch_remap() {
        // vertex 1 is not referenced, and still has an entry in the table
        let remap = optimize_vertex_fetch_remap(&[3, 0, 2, 2, 0, 3], 4);
        assert_eq!(remap, [1, u32::MAX, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_optimize_vertex_fetch_remap_index_out_of_range() {
        optimize_vertex_fetch_remap(&[0, 1, 4], 4);
    }

    #[test]
    fn test_optimize_overdraw() {
        let positions = grid_positions();
        let adapter = VertexDataAdapter::new(typed_to_bytes(&positions), 12, 0).unwrap();

        let mut indices = optimize_vertex_cache(&GRID, 9);
        let mut decoded = indices.clone();
        optimize_overdraw_in_place(&mut indices, &adapter, 1.05);
        optimize_overdraw_in_place_decoder(&mut decoded, &positions, 1.05);

        assert_eq!(indices, decoded);
        assert_eq!(sorted(&indices), sorted(&GRID));

        // the threshold only steers clustering, so any value is accepted
        optimize_overdraw_in_place(&mut indices, &adapter, f32::NAN);
        assert_eq!(sorted(&indices), sorted(&GRID));
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_optimize_overdraw_index_out_of_range() {
        let positions = grid_positions();
        let adapter = VertexDataAdapter::new(typed_to_bytes(&positions[..8]), 12, 0).unwrap();
        optimize_overdraw_in_place(&mut GRID.clone(), &adapter, 1.05);
    }
}
