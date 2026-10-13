use crate::utilities::{MAX_VERTEX_SIZE, check_indices, check_triangles};
use crate::{DecodePosition, VertexDataAdapter, ffi};
use std::mem;

pub type VertexCacheStatistics = ffi::meshopt_VertexCacheStatistics;
pub type VertexFetchStatistics = ffi::meshopt_VertexFetchStatistics;
pub type OverdrawStatistics = ffi::meshopt_OverdrawStatistics;
pub type CoverageStatistics = ffi::meshopt_CoverageStatistics;

/// Returns cache hit statistics using a simplified FIFO model.
/// Results may not match actual GPU performance.
pub fn analyze_vertex_cache(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
    warp_size: u32,
    prim_group_size: u32,
) -> VertexCacheStatistics {
    check_triangles(indices, vertex_count);
    unsafe {
        ffi::meshopt_analyzeVertexCache(
            indices.as_ptr(),
            indices.len(),
            vertex_count,
            cache_size,
            warp_size,
            prim_group_size,
        )
    }
}

/// Returns cache hit statistics using a simplified direct mapped model.
/// Results may not match actual GPU performance.
///
/// `vertex_size` must be between 1 and 256 bytes.
pub fn analyze_vertex_fetch(
    indices: &[u32],
    vertex_count: usize,
    vertex_size: usize,
) -> VertexFetchStatistics {
    check_indices(indices, vertex_count);
    assert!(
        vertex_size > 0 && vertex_size <= MAX_VERTEX_SIZE,
        "vertex size ({vertex_size}) must be between 1 and {MAX_VERTEX_SIZE} bytes"
    );
    unsafe {
        ffi::meshopt_analyzeVertexFetch(indices.as_ptr(), indices.len(), vertex_count, vertex_size)
    }
}

/// Returns overdraw statistics using a software rasterizer.
/// Results may not match actual GPU performance.
pub fn analyze_overdraw_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
) -> OverdrawStatistics {
    check_triangles(indices, vertices.len());
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_analyzeOverdraw(
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
        )
    }
}

/// Returns overdraw statistics using a software rasterizer.
/// Results may not match actual GPU performance.
pub fn analyze_overdraw(indices: &[u32], vertices: &VertexDataAdapter<'_>) -> OverdrawStatistics {
    check_triangles(indices, vertices.vertex_count);
    unsafe {
        ffi::meshopt_analyzeOverdraw(
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        )
    }
}

/// Returns coverage statistics (ratio of viewport pixels covered from each axis) using a software rasterizer.
///
/// See [`analyze_coverage`] for details.
pub fn analyze_coverage_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
) -> CoverageStatistics {
    check_triangles(indices, vertices.len());
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_analyzeCoverage(
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
        )
    }
}

/// Returns coverage statistics (ratio of viewport pixels covered from each axis) using a software rasterizer.
pub fn analyze_coverage(indices: &[u32], vertices: &VertexDataAdapter<'_>) -> CoverageStatistics {
    check_triangles(indices, vertices.vertex_count);
    unsafe {
        ffi::meshopt_analyzeCoverage(
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    // unit cube
    const CUBE: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    #[rustfmt::skip]
    const CUBE_INDICES: [u32; 36] = [
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        3, 6, 2, 3, 7, 6,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];

    fn adapter(vertices: &[[f32; 3]]) -> VertexDataAdapter<'_> {
        VertexDataAdapter::new(typed_to_bytes(vertices), mem::size_of::<[f32; 3]>(), 0).unwrap()
    }

    #[test]
    fn test_analyze_coverage() {
        let stats = analyze_coverage_decoder(&CUBE_INDICES, &CUBE);
        assert_eq!(stats.coverage, [1.0; 3]);
        assert_eq!(stats.extent, 1.0);

        let stats = analyze_coverage(&CUBE_INDICES[..6], &adapter(&CUBE));
        assert_eq!(stats.coverage[2], 1.0);
        assert_eq!(stats.coverage[0], 0.0);
        assert_eq!(stats.coverage[1], 0.0);
    }

    #[test]
    fn test_analyze_overdraw() {
        let stats = analyze_overdraw(&CUBE_INDICES, &adapter(&CUBE));
        let decoded = analyze_overdraw_decoder(&CUBE_INDICES, &CUBE);
        assert_eq!(stats.pixels_covered, decoded.pixels_covered);
        assert_eq!(stats.pixels_shaded, decoded.pixels_shaded);
        assert!(stats.pixels_covered > 0);
        assert_eq!(stats.overdraw, 1.0);

        // nothing to rasterize
        assert_eq!(analyze_overdraw_decoder::<[f32; 3]>(&[], &[]).overdraw, 0.0);
        assert_eq!(analyze_coverage(&[], &adapter(&[])).coverage, [0.0; 3]);
    }

    #[test]
    fn test_analyze_vertex_cache_and_fetch() {
        let stats = analyze_vertex_cache(&CUBE_INDICES, 8, 16, 0, 0);
        assert_eq!(stats.vertices_transformed, 8);

        let stats = analyze_vertex_fetch(&CUBE_INDICES, 8, 12);
        assert_eq!(stats.bytes_fetched, 128);
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_analyze_vertex_cache_index_out_of_range() {
        analyze_vertex_cache(&CUBE_INDICES, 7, 16, 0, 0);
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_analyze_vertex_fetch_index_out_of_range() {
        analyze_vertex_fetch(&[0, 1, 8], 8, 12);
    }

    #[test]
    #[should_panic(expected = "vertex size")]
    fn test_analyze_vertex_fetch_large_vertex() {
        analyze_vertex_fetch(&CUBE_INDICES, 8, 257);
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_analyze_overdraw_index_out_of_range() {
        analyze_overdraw(&CUBE_INDICES, &adapter(&CUBE[..7]));
    }
}
