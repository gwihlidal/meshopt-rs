use crate::{ffi, DecodePosition, VertexDataAdapter};
use std::mem;

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
            std::ptr::null_mut(),
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
            std::ptr::null_mut(),
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
            std::ptr::null_mut(),
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
            std::ptr::null_mut(),
        )
    };
    result.resize(index_count, 0u32);
    result
}
