use ffi;
use DecodePosition;

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
