use ffi;
use std::mem;
use DecodePosition;

pub fn simplify<T: DecodePosition>(
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
            result.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            positions.as_ptr() as *const f32,
            positions.len(),
            mem::size_of::<f32>() * 3,
            target_count,
            target_error,
        )
    };
    result.resize(index_count, 0u32);
    result
}
