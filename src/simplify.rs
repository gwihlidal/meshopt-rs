use crate::{ffi, DecodePosition, VertexDataAdapter};
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
    }
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
    let vertex_data = vertices.reader.get_ref();
    let vertex_data = vertex_data.as_ptr().cast::<u8>();
    let positions = unsafe { vertex_data.add(vertices.position_offset) };
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplify(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.cast::<f32>(),
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
    vertex_lock: &[bool],
    target_count: usize,
    target_error: f32,
    options: SimplifyOptions,
    result_error: Option<&mut f32>,
) -> Vec<u32> {
    let vertex_data = vertices.reader.get_ref();
    let vertex_data = vertex_data.as_ptr().cast::<u8>();
    let positions = unsafe { vertex_data.add(vertices.position_offset) };
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifyWithAttributes(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.cast::<f32>(),
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
    vertex_lock: &[bool],
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

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance.
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
    let vertex_data = vertices.reader.get_ref();
    let vertex_data = vertex_data.as_ptr().cast::<u8>();
    let positions = unsafe { vertex_data.add(vertices.position_offset) };
    let mut result: Vec<u32> = vec![0; indices.len()];
    let index_count = unsafe {
        ffi::meshopt_simplifySloppy(
            result.as_mut_ptr().cast(),
            indices.as_ptr().cast(),
            indices.len(),
            positions.cast(),
            vertices.vertex_count,
            vertices.vertex_stride,
            target_count,
            target_error,
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
    result
}

/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance.
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
            target_count,
            target_error,
            result_error.map_or_else(std::ptr::null_mut, |v| v as *mut _),
        )
    };
    result.resize(index_count, 0u32);
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
