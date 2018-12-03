use ffi;

pub fn stripify(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; indices.len() / 3 * 4];
    let index_count = unsafe {
        ffi::meshopt_stripify(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        )
    };
    assert!(index_count <= result.len());
    result.resize(index_count, 0u32);
    result
}

/// Mesh unstripifier
/// Converts a triangle strip to a triangle list
pub fn unstripify(indices: &[u32]) -> Vec<u32> {
    let mut result: Vec<u32> = vec![0; (indices.len() - 2) * 3];
    let index_count = unsafe {
        ffi::meshopt_unstripify(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
        )
    };
    assert!(index_count <= result.len());
    result.resize(index_count, 0u32);
    result
}
