use crate::{ffi, Error, Result};

/// Converts a previously vertex cache optimized triangle list to triangle
/// strip, stitching strips using restart index.
///
/// For maximum efficiency the index buffer being converted has to be
/// optimized for vertex cache first.
pub fn stripify(indices: &[u32], vertex_count: usize) -> Result<Vec<u32>> {
    let mut result: Vec<u32> = vec![0; indices.len() / 3 * 4];
    let index_count = unsafe {
        ffi::meshopt_stripify(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        )
    };
    if index_count <= result.len() {
        result.resize(index_count, 0u32);
        Ok(result)
    } else {
        Err(Error::memory("index count is larger than result"))
    }
}

/// Converts a triangle strip to a triangle list
pub fn unstripify(indices: &[u32]) -> Result<Vec<u32>> {
    let mut result: Vec<u32> = vec![0; (indices.len() - 2) * 3];
    let index_count = unsafe {
        ffi::meshopt_unstripify(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
        )
    };
    if index_count <= result.len() {
        result.resize(index_count, 0u32);
        Ok(result)
    } else {
        Err(Error::memory("index count is larger than result"))
    }
}
