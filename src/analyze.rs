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
pub fn analyze_vertex_fetch(
    indices: &[u32],
    vertex_count: usize,
    vertex_size: usize,
) -> VertexFetchStatistics {
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

    #[test]
    fn test_analyze_coverage() {
        // unit cube
        let vertices: &[[f32; 3]] = &[
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
        let indices = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            3, 6, 2, 3, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ];

        let stats = analyze_coverage_decoder(&indices, vertices);
        assert_eq!(stats.coverage, [1.0; 3]);
        assert_eq!(stats.extent, 1.0);

        let adapter =
            VertexDataAdapter::new(typed_to_bytes(vertices), mem::size_of::<[f32; 3]>(), 0)
                .unwrap();
        let stats = analyze_coverage(&indices[..6], &adapter);
        assert_eq!(stats.coverage[2], 1.0);
        assert_eq!(stats.coverage[0], 0.0);
        assert_eq!(stats.coverage[1], 0.0);
    }
}
