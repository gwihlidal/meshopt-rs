use ffi;
use std::mem;
use DecodePosition;

/// Vertex transform cache optimizer
/// Reorders indices to reduce the number of GPU vertex shader invocations
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut optimized: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            optimized.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        );
    }
    optimized
}

/// Vertex transform cache optimizer
/// Reorders indices to reduce the number of GPU vertex shader invocations
pub fn optimize_vertex_cache_in_place(indices: &mut [u32], vertex_count: usize) {
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        );
    }
}

pub fn optimize_vertex_cache_fifo(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
) -> Vec<u32> {
    let mut optimized: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            optimized.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
    optimized
}

pub fn optimize_vertex_cache_fifo_in_place(
    indices: &mut [u32],
    vertex_count: usize,
    cache_size: u32,
) {
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
}

/// Vertex fetch cache optimizer
/// Reorders vertices and changes indices to reduce the amount of GPU memory fetches during vertex processing
///
/// destination must contain enough space for the resulting vertex buffer (vertex_count elements)
/// indices is used both as an input and as an output index buffer
pub fn optimize_vertex_fetch<T: Clone + Default>(indices: &mut [u32], vertices: &[T]) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    result.resize(vertices.len(), T::default());
    let next_vertex = unsafe {
        ffi::meshopt_optimizeVertexFetch(
            result.as_mut_ptr() as *mut ::std::os::raw::c_void,
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.len(),
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            vertices.len(),
            mem::size_of::<T>(),
        )
    };
    result.resize(next_vertex, T::default());
    result
}

pub fn optimize_vertex_fetch_in_place<T>(indices: &mut [u32], vertices: &mut [T]) -> usize {
    unsafe {
        ffi::meshopt_optimizeVertexFetch(
            vertices.as_mut_ptr() as *mut ::std::os::raw::c_void,
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.len(),
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            vertices.len(),
            mem::size_of::<T>(),
        )
    }
}

/// Vertex fetch cache optimizer
/// Generates vertex remap to reduce the amount of GPU memory fetches during vertex processing
/// The resulting remap table should be used to reorder vertex/index buffers using `optimize_remap_vertex_buffer`/`optimize_remap_index_buffer`
///
/// destination must contain enough space for the resulting remap table (`vertex_count` elements)
pub fn optimize_vertex_fetch_remap(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut result: Vec<u32> = Vec::new();
    result.resize(vertex_count, 0u32);
    let next_vertex = unsafe {
        ffi::meshopt_optimizeVertexFetchRemap(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        )
    };
    result.resize(next_vertex, 0u32);
    result
}

/// Overdraw optimizer
/// Reorders indices to reduce the number of GPU vertex shader invocations and the pixel overdraw
///
/// destination must contain enough space for the resulting index buffer (index_count elements)
/// indices must contain index data that is the result of optimizeVertexCache (*not* the original mesh indices!)
/// vertex_positions should have float3 position in the first 12 bytes of each vertex - similar to glVertexPointer
/// threshold indicates how much the overdraw optimizer can degrade vertex cache efficiency (1.05 = up to 5%) to reduce overdraw more efficiently
pub fn optimize_overdraw_in_place<T: DecodePosition>(
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
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            positions.as_ptr() as *const f32,
            positions.len(),
            mem::size_of::<f32>() * 3,
            threshold,
        );
    }
}
