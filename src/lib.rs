pub mod ffi;

pub type VertexCacheStatistics = ffi::meshopt_VertexCacheStatistics;

pub fn analyze_vertex_cache(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
    warp_size: u32,
    prim_group_size: u32,
) -> VertexCacheStatistics {
    unsafe {
        ffi::meshopt_analyzeVertexCache(
            indices.as_ptr() as *mut ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            cache_size,
            warp_size,
            prim_group_size,
        )
    }
}
