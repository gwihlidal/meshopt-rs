use crate::{ffi, DecodePosition, VertexDataAdapter, VertexStream};

/// Generate index buffer that can be used for more efficient rendering when only a subset of the vertex
/// attributes is necessary. All vertices that are binary equivalent (wrt first `vertex_size` bytes) map to
/// the first vertex in the original vertex buffer.
///
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using
/// the original index buffer for regular rendering.
pub fn generate_shadow_indices(indices: &[u32], vertices: &VertexDataAdapter<'_>) -> Vec<u32> {
    let vertex_data = vertices.reader.get_ref();
    let vertex_data = vertex_data.as_ptr().cast::<u8>();
    let positions = unsafe { vertex_data.add(vertices.position_offset) };
    let mut shadow_indices: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_generateShadowIndexBuffer(
            shadow_indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions.cast(),
            vertices.vertex_count,
            std::mem::size_of::<f32>() * 3,
            vertices.vertex_stride,
        );
    }
    shadow_indices
}

/// Generate index buffer that can be used for more efficient rendering when only a subset of the vertex
/// attributes is necessary. All vertices that are binary equivalent (wrt first `vertex_size` bytes) map to
/// the first vertex in the original vertex buffer.
///
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using
/// the original index buffer for regular rendering.
pub fn generate_shadow_indices_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
) -> Vec<u32> {
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let positions = vertices.as_ptr().cast();
    let mut shadow_indices: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_generateShadowIndexBuffer(
            shadow_indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions,
            vertices.len() * 3,
            std::mem::size_of::<f32>() * 3,
            std::mem::size_of::<f32>() * 3,
        );
    }
    shadow_indices
}

/// Generate index buffer that can be used for more efficient rendering when only a subset of the vertex
/// attributes is necessary. All vertices that are binary equivalent (wrt specified streams) map to the
/// first vertex in the original vertex buffer.
///
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using
/// the original index buffer for regular rendering.
pub fn generate_shadow_indices_multi(
    indices: &[u32],
    vertex_count: usize,
    streams: &[VertexStream<'_>],
) -> Vec<u32> {
    let streams: Vec<ffi::meshopt_Stream> = streams
        .iter()
        .map(|stream| ffi::meshopt_Stream {
            data: stream.data.cast(),
            size: stream.size,
            stride: stream.stride,
        })
        .collect();
    let mut shadow_indices: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_generateShadowIndexBufferMulti(
            shadow_indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
            streams.as_ptr(),
            streams.len(),
        );
    }
    shadow_indices
}
