use crate::{DecodePosition, VertexDataAdapter, ffi};
use bitflags::bitflags;
use std::mem;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SimplifyOptions : u32 {
        const None = 0;
        /// Locks the vertices that lie on the topological border of the mesh in place such that
        /// they don't move during simplification.
        /// This can be valuable to simplify independent chunks of a mesh, for example terrain,
        /// to ensure that individual levels of detail can be stitched together later without gaps.
        const LockBorder = 1;
        /// Improve simplification performance assuming input indices are a sparse subset of the mesh.
        /// Note that error becomes relative to subset extents.
        const Sparse = 2;
        /// Treat error limit and resulting error as absolute instead of relative to mesh extents.
        const ErrorAbsolute = 4;
        /// Remove disconnected parts of the mesh during simplification incrementally, regardless of
        /// the topological restrictions inside components.
        const Prune = 8;
        /// Produce more regular triangle sizes and shapes during simplification, at some cost to
        /// geometric and attribute quality.
        const Regularize = 16;
        /// Allow collapses across attribute discontinuities, except for vertices that are tagged with
        /// `SimplifyVertex::Protect` in `vertex_lock`.
        const Permissive = 32;
        /// Produce more regular triangle sizes and shapes during simplification, at a small cost to
        /// geometric and attribute quality.
        const RegularizeLight = 64;
        /// Experimental: Try to preserve fold lines between opposite-facing triangles, at a small
        /// performance cost.
        const PreserveFolds = 128;
        /// Experimental: Clamp attribute error to match position error scale and avoid extreme
        /// error values in areas with high attribute variance.
        const ErrorClamped = 256;
    }
}

bitflags! {
    /// Simplification vertex flags/locks, for use in `vertex_lock` arrays in simplification APIs.
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SimplifyVertex : u8 {
        const None = 0;
        /// Do not move this vertex.
        const Lock = 1;
        /// Protect attribute discontinuity at this vertex; must be used together with
        /// `SimplifyOptions::Permissive` option.
        const Protect = 2;
        /// Increase priority for this vertex, making it more likely that it's preserved during
        /// simplification.
        const Priority = 4;
    }
}

/// Element types for `vertex_lock` arrays: `bool` (`true` locks the vertex) or `SimplifyVertex` flags.
pub trait VertexLock: private::Sealed {}

impl VertexLock for bool {}
impl VertexLock for SimplifyVertex {}

mod private {
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for super::SimplifyVertex {}
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh
/// appearance as much as possible.
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer
/// using `optimize_vertex_fetch` is recommended.
pub fn simplify(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplify(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh
/// appearance as much as possible.
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer
/// using `optimize_vertex_fetch` is recommended.
pub fn simplify_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplify(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh
/// appearance as much as possible, while respecting the given vertex locks
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer
/// using `optimize_vertex_fetch` is recommended.
pub fn simplify_with_locks(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    vertex_lock: &[impl VertexLock],
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyWithAttributes(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            vertex_lock.as_ptr().cast(),
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh
/// appearance as much as possible, while respecting the given vertex locks
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer
/// using `optimize_vertex_fetch` is recommended.
pub fn simplify_with_locks_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    vertex_lock: &[impl VertexLock],
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyWithAttributes(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            vertex_lock.as_ptr().cast(),
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh
/// appearance as much as possible, weighing vertex attributes by the supplied weights,
/// while respecting the given vertex locks
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer
/// using `optimize_vertex_fetch` is recommended.
#[allow(clippy::too_many_arguments)]
pub fn simplify_with_attributes_and_locks(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    vertex_attributes: &[f32],
    vertex_attribute_weights: &[f32],
    vertex_attributes_stride: usize,
    vertex_lock: &[impl VertexLock],
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyWithAttributes(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            vertex_attributes.as_ptr(),
            vertex_attributes_stride,
            vertex_attribute_weights.as_ptr(),
            vertex_attribute_weights.len(),
            vertex_lock.as_ptr().cast(),
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh
/// appearance as much as possible, weighing vertex attributes by the supplied weights,
/// while respecting the given vertex locks
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer
/// using `optimize_vertex_fetch` is recommended.
#[allow(clippy::too_many_arguments)]
pub fn simplify_with_attributes_and_locks_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    vertex_attributes: &[f32],
    vertex_attribute_weights: &[f32],
    vertex_attributes_stride: usize,
    vertex_lock: &[impl VertexLock],
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyWithAttributes(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            vertex_attributes.as_ptr(),
            vertex_attributes_stride,
            vertex_attribute_weights.as_ptr(),
            vertex_attribute_weights.len(),
            vertex_lock.as_ptr().cast(),
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, attempting to preserve mesh appearance as much as possible.
///
/// Similar to `simplify_with_attributes_and_locks`, but destructively updates positions and attribute values for
/// optimal appearance. The algorithm tries to preserve mesh topology and can stop short of the target goal based on
/// topology constraints or target error. If not all attributes from the input mesh are needed, it's recommended to
/// reindex the mesh without them prior to simplification.
/// Returns the number of indices after simplification, `indices` are destructively updated with new index data.
///
/// The updated index buffer references vertices from the original vertex buffer, however the vertex positions and
/// attributes are updated in-place. Creating a compact vertex buffer using `optimize_vertex_fetch` is recommended;
/// if the original vertex data is needed, it should be copied before simplification.
/// Note that the number of attributes with non-zero weights affects memory requirements and running time.
/// Attributes with zero weights are not updated.
///
/// * `vertex_attributes` should have `vertex_attribute_weights.len()` floats for each vertex
/// * `vertex_attribute_weights` determine relative priority of attributes between each other and wrt position;
///   `vertex_attribute_weights.len()` must be <= 32
/// * `vertex_lock` can be `None`; when it's not `None`, it should have a value for each vertex
/// * `target_error` represents the error relative to mesh extents that can be tolerated, e.g. 0.01 = 1% deformation
///   (unless absolute error option is used)
/// * `result_error` can be `None`; when it's not `None`, it will contain the resulting (relative/absolute) error
///   after simplification
///
/// Panics if `vertex_attributes` or `vertex_lock` is too small for the number of vertices, if
/// `vertex_attributes_stride` is too small for `vertex_attribute_weights`, or if there are more than 32
/// attribute weights.
#[allow(clippy::too_many_arguments)]
pub fn simplify_with_update(
    indices: &mut [u32],
    vertex_positions: &mut [[f32; 3]],
    vertex_attributes: &mut [f32],
    vertex_attribute_weights: &[f32],
    vertex_attributes_stride: usize,
    vertex_lock: Option<&[SimplifyVertex]>,
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> usize {
    let vertex_count = vertex_positions.len();
    assert!(
        vertex_count
            .checked_mul(vertex_attributes_stride)
            .is_some_and(|size| mem::size_of_val(vertex_attributes) >= size),
        "vertex_attributes ({}) must have room for {vertex_count} vertices at stride {vertex_attributes_stride}",
        vertex_attributes.len()
    );
    assert!(
        mem::size_of_val(vertex_attribute_weights) <= vertex_attributes_stride,
        "vertex_attributes_stride ({vertex_attributes_stride}) must have room for {} attributes",
        vertex_attribute_weights.len()
    );
    assert!(
        vertex_attribute_weights.len() <= 32,
        "vertex_attribute_weights ({}) must have at most 32 attributes",
        vertex_attribute_weights.len()
    );
    assert!(
        vertex_lock.is_none_or(|lock| lock.len() >= vertex_count),
        "vertex_lock must have a value for each of the {vertex_count} vertices"
    );

    unsafe {
        ffi::meshopt_simplifyWithUpdate(
            indices.as_mut_ptr(),
            indices.len(),
            vertex_positions.as_mut_ptr().cast(),
            vertex_count,
            mem::size_of::<f32>() * 3,
            vertex_attributes.as_mut_ptr(),
            vertex_attributes_stride,
            vertex_attribute_weights.as_ptr(),
            vertex_attribute_weights.len(),
            vertex_lock.map_or(std::ptr::null(), |lock| lock.as_ptr().cast()),
            target_count,
            target_error,
            options.bits(),
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    }
}

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance.
///
/// The algorithm doesn't preserve mesh topology but is always able to reach target triangle count.
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer using `optimize_vertex_fetch`
/// is recommended.
pub fn simplify_sloppy(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    target_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifySloppy(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            std::ptr::null(),
            target_count,
            target_error,
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance.
///
/// The algorithm doesn't preserve mesh topology but is always able to reach target triangle count.
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer using `optimize_vertex_fetch`
/// is recommended.
pub fn simplify_sloppy_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    target_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifySloppy(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            std::ptr::null(),
            target_count,
            target_error,
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance,
/// while respecting the given vertex locks.
///
/// The algorithm doesn't preserve mesh topology but is always able to reach target triangle count.
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer using `optimize_vertex_fetch`
/// is recommended.
pub fn simplify_sloppy_with_locks(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    vertex_lock: &[impl VertexLock],
    target_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let positions = vertices.pos_ptr();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifySloppy(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.cast(),
            vertices.vertex_count,
            vertices.vertex_stride,
            vertex_lock.as_ptr().cast(),
            target_count,
            target_error,
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance,
/// while respecting the given vertex locks.
///
/// The algorithm doesn't preserve mesh topology but is always able to reach target triangle count.
///
/// The resulting index buffer references vertices from the original vertex buffer.
///
/// If the original vertex data isn't required, creating a compact vertex buffer using `optimize_vertex_fetch`
/// is recommended.
pub fn simplify_sloppy_with_locks_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    vertex_lock: &[impl VertexLock],
    target_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifySloppy(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            vertex_lock.as_ptr().cast(),
            target_count,
            target_error,
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh by removing small isolated parts of the mesh.
///
/// The resulting index buffer references vertices from the original vertex buffer.
/// If the original vertex data isn't needed, creating a compact vertex buffer using `optimize_vertex_fetch`
/// is recommended.
///
/// `target_error` represents the error relative to mesh extents that can be tolerated, e.g. 0.01 = 1% deformation;
/// value range [0..1]
pub fn simplify_prune(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    target_error: f32,
) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyPrune(
            result.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            target_error,
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh by removing small isolated parts of the mesh.
///
/// See [`simplify_prune`] for details.
pub fn simplify_prune_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    target_error: f32,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyPrune(
            result.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            target_error,
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of points in the cloud to reach the given target.
///
/// The resulting index buffer references vertices from the original vertex buffer.
/// If the original vertex data isn't needed, creating a compact vertex buffer using `optimize_vertex_fetch`
/// is recommended.
///
/// * `vertex_colors` can be `None`; when it's not `None`, it should have a color for each vertex
/// * `color_weight` determines relative priority of color wrt position; 1.0 is a safe default
///
/// Panics if `vertex_colors` is shorter than the number of vertices.
pub fn simplify_points(
    vertices: &VertexDataAdapter<'_>,
    vertex_colors: Option<&[[f32; 3]]>,
    color_weight: f32,
    target_count: usize,
) -> Vec<u32> {
    assert!(
        vertex_colors.is_none_or(|colors| colors.len() >= vertices.vertex_count),
        "vertex_colors must have a color for each of the {} vertices",
        vertices.vertex_count
    );
    let mut result: Vec<u32> = vec![0; target_count];
    let point_count = unsafe {
        ffi::meshopt_simplifyPoints(
            result.as_mut_ptr(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            vertex_colors.map_or(std::ptr::null(), |colors| colors.as_ptr().cast()),
            mem::size_of::<f32>() * 3,
            color_weight,
            target_count,
        )
    };
    result.resize(point_count, 0u32);
    result
}

/// Reduces the number of points in the cloud to reach the given target.
///
/// See [`simplify_points`] for details.
pub fn simplify_points_decoder<T: DecodePosition>(
    vertices: &[T],
    vertex_colors: Option<&[[f32; 3]]>,
    color_weight: f32,
    target_count: usize,
) -> Vec<u32> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    assert!(
        vertex_colors.is_none_or(|colors| colors.len() >= positions.len()),
        "vertex_colors must have a color for each of the {} vertices",
        positions.len()
    );
    let mut result: Vec<u32> = vec![0; target_count];
    let point_count = unsafe {
        ffi::meshopt_simplifyPoints(
            result.as_mut_ptr(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            vertex_colors.map_or(std::ptr::null(), |colors| colors.as_ptr().cast()),
            mem::size_of::<f32>() * 3,
            color_weight,
            target_count,
        )
    };
    result.resize(point_count, 0u32);
    result
}

/// Returns the error scaling factor used by the simplifier to convert between absolute and relative extents
///
/// Absolute error must be *divided* by the scaling factor before passing it to `simplify` as `target_error`
/// Relative error returned by `simplify` via `result_error` must be *multiplied* by the scaling factor to get absolute error.
pub fn simplify_scale(vertices: &VertexDataAdapter<'_>) -> f32 {
    unsafe {
        ffi::meshopt_simplifyScale(
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        )
    }
}

/// Returns the error scaling factor used by the simplifier to convert between absolute and relative extents
///
/// Absolute error must be *divided* by the scaling factor before passing it to `simplify` as `target_error`
/// Relative error returned by `simplify` via `result_error` must be *multiplied* by the scaling factor to get absolute error.
pub fn simplify_scale_decoder<T: DecodePosition>(vertices: &[T]) -> f32 {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();

    unsafe {
        ffi::meshopt_simplifyScale(
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    #[test]
    fn test_simplify_sloppy_with_locks() {
        // Test mesh from vendor tests - triangle fan with spine
        // 0
        // 1 2
        // 3 4 5
        let indices = &[
            0, 2, 1, // triangle 0
            1, 2, 3, // triangle 1
            3, 2, 4, // triangle 2
            2, 5, 4, // triangle 3
        ];

        let vertices: &[f32] = &[
            0.0, 4.0, 0.0, // vertex 0
            0.0, 1.0, 0.0, // vertex 1
            2.0, 2.0, 0.0, // vertex 2
            0.0, 0.0, 0.0, // vertex 3
            1.0, 0.0, 0.0, // vertex 4
            4.0, 0.0, 0.0, // vertex 5
        ];

        // Lock the spine vertices (0, 2, 5)
        let vertex_locks = &[true, false, true, false, false, true];

        let vertices_adapter =
            VertexDataAdapter::new(typed_to_bytes(vertices), 3 * mem::size_of::<f32>(), 0).unwrap();

        let mut result_error = 0.0f32;
        let result = simplify_sloppy_with_locks(
            indices,
            &vertices_adapter,
            vertex_locks,
            3, // target 3 indices (1 triangle)
            1.0,
            Some(&mut result_error),
        );

        // Should preserve 6 indices (2 triangles) because of locks
        assert_eq!(result.len(), 6);
        assert_eq!(result_error, 0.0);

        // Expected result based on vendor test
        let expected = &[0, 2, 1, 1, 2, 5];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simplify_with_lock_flags() {
        // 0 1 2
        // 3 4 5
        // 6 7 8
        #[rustfmt::skip]
        let vertices: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 0.0],
            [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0],
        ];
        #[rustfmt::skip]
        let indices = &[
            0, 1, 3, 3, 1, 4, 1, 2, 4, 4, 2, 5,
            3, 4, 6, 6, 4, 7, 4, 5, 7, 7, 5, 8,
        ];

        let mut vertex_lock = [SimplifyVertex::Lock; 9];
        vertex_lock[4] = SimplifyVertex::None;

        let result = simplify_with_locks_decoder(
            indices,
            vertices,
            &vertex_lock,
            3,
            1e-3,
            SimplifyOptions::None,
            None,
        );

        // from meshoptimizer's simplifyLockFlags test
        let expected = &[0, 1, 3, 1, 2, 3, 3, 2, 5, 6, 3, 7, 3, 5, 7, 7, 5, 8];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simplify_with_update() {
        //     1
        //    3 4
        // 0       2
        let mut positions = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.9, 0.2, 0.1],
            [1.1, 0.2, 0.1],
        ];
        let mut attributes = [0.0, 0.0, 0.0, 0.2, 0.1];
        let mut indices = [0, 1, 3, 3, 1, 4, 4, 1, 2, 0, 3, 2, 3, 4, 2];

        let index_count = simplify_with_update(
            &mut indices,
            &mut positions,
            &mut attributes,
            &[1.0],
            mem::size_of::<f32>(),
            None,
            9,
            1.0,
            SimplifyOptions::None,
            None,
        );

        // from meshoptimizer's simplifyUpdate test
        assert_eq!(index_count, 9);
        assert_eq!(&indices[..index_count], &[0, 1, 3, 3, 1, 2, 0, 3, 2]);

        // border vertices haven't moved but may have small floating point drift
        for attribute in &attributes[0..3] {
            assert!(attribute.abs() < 1e-6);
        }

        // center vertex got updated
        assert!((positions[3][0] - 0.88).abs() < 1e-2);
        assert!((positions[3][1] - 0.19).abs() < 1e-2);
        assert!((positions[3][2] - 0.11).abs() < 1e-2);
        assert!((attributes[3] - 0.18).abs() < 1e-2);
    }

    #[test]
    fn test_simplify_with_update_locked() {
        let mut positions = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.9, 0.2, 0.1],
            [1.1, 0.2, 0.1],
        ];
        let mut attributes = [0.0, 0.0, 0.0, 0.2, 0.1];
        let mut indices = [0, 1, 3, 3, 1, 4, 4, 1, 2, 0, 3, 2, 3, 4, 2];

        let mut vertex_lock = [SimplifyVertex::None; 5];
        vertex_lock[3] = SimplifyVertex::Lock;

        let index_count = simplify_with_update(
            &mut indices,
            &mut positions,
            &mut attributes,
            &[1.0],
            mem::size_of::<f32>(),
            Some(&vertex_lock),
            9,
            1.0,
            SimplifyOptions::None,
            None,
        );

        assert_eq!(index_count, 9);
        assert_eq!(&indices[..index_count], &[0, 1, 3, 3, 1, 2, 0, 3, 2]);

        // locking guarantees exact result
        assert_eq!(positions[3], [0.9, 0.2, 0.1]);
        assert_eq!(attributes[3], 0.2);
    }

    #[test]
    fn test_simplify_prune() {
        let vertices: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 2.0],
        ];
        let indices = &[0, 1, 2, 3, 4, 5, 6, 7, 8];

        let vertices_adapter =
            VertexDataAdapter::new(typed_to_bytes(vertices), mem::size_of::<[f32; 3]>(), 0)
                .unwrap();

        // from meshoptimizer's simplifyPrune test: only the largest triangle survives
        assert_eq!(simplify_prune(indices, &vertices_adapter, 0.5), &[6, 7, 8]);
        assert_eq!(simplify_prune_decoder(indices, vertices, 0.5), &[6, 7, 8]);
    }

    #[test]
    fn test_simplify_points() {
        // simplifying down to 0 points results in 0 immediately
        let vertices: &[[f32; 3]] = &[[0.0; 3]; 3];
        assert!(simplify_points_decoder(vertices, None, 0.0, 0).is_empty());

        let vertices: Vec<[f32; 3]> = (0..64)
            .map(|i| [(i % 8) as f32, (i / 8) as f32, 0.0])
            .collect();
        let colors: Vec<[f32; 3]> = (0..64).map(|i| [i as f32 / 64.0, 0.0, 0.0]).collect();

        let vertices_adapter =
            VertexDataAdapter::new(typed_to_bytes(&vertices), mem::size_of::<[f32; 3]>(), 0)
                .unwrap();

        let result = simplify_points(&vertices_adapter, Some(&colors), 1.0, 16);
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&i| i < 64));

        let mut unique = result.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 16);

        assert_eq!(
            simplify_points_decoder(&vertices, Some(&colors), 1.0, 16),
            result
        );
    }
}
