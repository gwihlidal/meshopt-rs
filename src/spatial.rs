use crate::utilities::check_triangles;
use crate::{DecodePosition, VertexDataAdapter, ffi};
use std::mem;

/// Generates a remap table that can be used to reorder points for spatial locality.
///
/// Resulting remap table maps old vertices to new vertices and can be used in `remap_vertex_buffer`.
pub fn spatial_sort_remap(vertices: &VertexDataAdapter<'_>) -> Vec<u32> {
    let mut remap: Vec<u32> = vec![0; vertices.vertex_count];
    unsafe {
        ffi::meshopt_spatialSortRemap(
            remap.as_mut_ptr(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        );
    }
    remap
}

/// Generates a remap table that can be used to reorder points for spatial locality.
///
/// See [`spatial_sort_remap`] for details.
pub fn spatial_sort_remap_decoder<T: DecodePosition>(vertices: &[T]) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut remap: Vec<u32> = vec![0; positions.len()];
    unsafe {
        ffi::meshopt_spatialSortRemap(
            remap.as_mut_ptr(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
        );
    }
    remap
}

/// Reorders triangles for spatial locality, and generates a new index buffer.
///
/// The resulting index buffer can be used with other functions like `optimize_vertex_cache`.
pub fn spatial_sort_triangles(indices: &[u32], vertices: &VertexDataAdapter<'_>) -> Vec<u32> {
    check_triangles(indices, vertices.vertex_count);
    let mut result: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_spatialSortTriangles(
            result.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        );
    }
    result
}

/// Reorders triangles for spatial locality, and generates a new index buffer.
///
/// See [`spatial_sort_triangles`] for details.
pub fn spatial_sort_triangles_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    check_triangles(indices, positions.len());
    let mut result: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_spatialSortTriangles(
            result.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
        );
    }
    result
}

/// Reorders points into clusters optimized for spatial locality, and generates a new index buffer.
///
/// Ensures the output can be split into `cluster_size` chunks where each chunk has good positional locality.
/// Only the last chunk will be smaller than `cluster_size`.
pub fn spatial_cluster_points(vertices: &VertexDataAdapter<'_>, cluster_size: usize) -> Vec<u32> {
    let cluster_size = cluster_size.max(1);
    let mut result: Vec<u32> = vec![0; vertices.vertex_count];
    unsafe {
        ffi::meshopt_spatialClusterPoints(
            result.as_mut_ptr(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            cluster_size,
        );
    }
    result
}

/// Reorders points into clusters optimized for spatial locality, and generates a new index buffer.
///
/// See [`spatial_cluster_points`] for details.
pub fn spatial_cluster_points_decoder<T: DecodePosition>(
    vertices: &[T],
    cluster_size: usize,
) -> Vec<u32> {
    let cluster_size = cluster_size.max(1);
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; positions.len()];
    unsafe {
        ffi::meshopt_spatialClusterPoints(
            result.as_mut_ptr(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            cluster_size,
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    // 4x4 grid of points in shuffled order
    fn grid() -> Vec<[f32; 3]> {
        [5, 12, 0, 9, 3, 14, 7, 1, 10, 15, 2, 8, 13, 4, 11, 6]
            .iter()
            .map(|&i| [(i % 4) as f32, (i / 4) as f32, 0.0])
            .collect()
    }

    fn is_permutation(values: &[u32], count: usize) -> bool {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted == (0..count as u32).collect::<Vec<_>>()
    }

    #[test]
    fn test_spatial_sort_remap() {
        let points = grid();
        let adapter =
            VertexDataAdapter::new(typed_to_bytes(&points), mem::size_of::<[f32; 3]>(), 0).unwrap();

        let remap = spatial_sort_remap(&adapter);
        assert!(is_permutation(&remap, points.len()));
        assert_eq!(spatial_sort_remap_decoder(&points), remap);
    }

    #[test]
    fn test_spatial_sort_triangles() {
        let points = grid();
        let adapter =
            VertexDataAdapter::new(typed_to_bytes(&points), mem::size_of::<[f32; 3]>(), 0).unwrap();
        let indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

        let result = spatial_sort_triangles(&indices, &adapter);
        assert_eq!(spatial_sort_triangles_decoder(&indices, &points), result);

        // the output has the same triangles, possibly reordered
        let mut input_triangles: Vec<&[u32]> = indices.chunks(3).collect();
        let mut output_triangles: Vec<&[u32]> = result.chunks(3).collect();
        input_triangles.sort_unstable();
        output_triangles.sort_unstable();
        assert_eq!(input_triangles, output_triangles);
    }

    #[test]
    fn test_spatial_cluster_points() {
        let points = grid();
        let adapter =
            VertexDataAdapter::new(typed_to_bytes(&points), mem::size_of::<[f32; 3]>(), 0).unwrap();

        let result = spatial_cluster_points(&adapter, 4);
        assert!(is_permutation(&result, points.len()));
        assert_eq!(spatial_cluster_points_decoder(&points, 4), result);
    }

    #[test]
    #[should_panic(expected = "index (16) must be less than vertex count (16)")]
    fn test_spatial_sort_triangles_index_out_of_range() {
        let points = grid();
        let adapter =
            VertexDataAdapter::new(typed_to_bytes(&points), mem::size_of::<[f32; 3]>(), 0).unwrap();
        spatial_sort_triangles(&[0, 1, 16], &adapter);
    }

    #[test]
    fn test_spatial_cluster_points_zero_cluster_size() {
        let points = grid();
        assert_eq!(
            spatial_cluster_points_decoder(&points, 0),
            spatial_cluster_points_decoder(&points, 1)
        );
    }

    #[test]
    fn test_spatial_empty() {
        let points: Vec<[f32; 3]> = vec![];
        assert!(spatial_sort_remap_decoder(&points).is_empty());
        assert!(spatial_sort_triangles_decoder(&[], &points).is_empty());
        assert!(spatial_cluster_points_decoder(&points, 4).is_empty());
    }

    #[test]
    fn test_spatial_non_finite() {
        // positions the library can't quantize must still produce a valid ordering
        let mut points = grid();
        points[3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        points[7] = [f32::MAX, f32::MIN, f32::NAN];

        assert!(is_permutation(
            &spatial_sort_remap_decoder(&points),
            points.len()
        ));
        assert!(is_permutation(
            &spatial_cluster_points_decoder(&points, 4),
            points.len()
        ));
    }
}
