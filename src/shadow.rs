use ffi;
use DecodePosition;

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
