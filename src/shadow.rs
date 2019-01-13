use crate::ffi;
use crate::DecodePosition;
use crate::VertexStream;

/// Generate index buffer that can be used for more efficient rendering when only a subset of the vertex attributes is necessary.
/// All vertices that are binary equivalent (wrt first vertex_size bytes) map to the first vertex in the original vertex buffer.
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using the original vertex/index
/// buffers elsewhere.
pub fn generate_shadow_indices<T: DecodePosition>(indices: &[u32], vertices: &[T]) -> Vec<u32> {
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let positions = vertices.as_ptr() as *const ::std::ffi::c_void;
    let mut shadow_indices: Vec<u32> = vec![0; indices.len()];
    unsafe {
        ffi::meshopt_generateShadowIndexBuffer(
            shadow_indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions,
            vertices.len() * 3,
            ::std::mem::size_of::<f32>() * 3,
            ::std::mem::size_of::<f32>() * 3,
        );
    }
    shadow_indices
}

/// Generate index buffer that can be used for more efficient rendering when only a subset of the vertex attributes is necessary.
/// All vertices that are binary equivalent (wrt specified streams) map to the first vertex in the original vertex buffer.
/// This makes it possible to use the index buffer for Z pre-pass or shadowmap rendering, while using the original vertex/index
/// buffers elsewhere.
pub fn generate_shadow_indices_multi(indices: &[u32], vertex_count: usize, streams: &[VertexStream]) -> Vec<u32> {
    let streams: Vec<ffi::meshopt_Stream> = streams.iter().map(|stream| {
        ffi::meshopt_Stream {
            data: stream.data.as_ptr() as *const ::std::ffi::c_void,
            size: stream.data.len(),
            stride: stream.stride,
        }
    }).collect();
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
