use ffi;
use std::mem;

pub fn encode_index_buffer(indices: &[u32], vertex_count: usize) -> Vec<u8> {
    let bounds = unsafe { ffi::meshopt_encodeIndexBufferBound(indices.len(), vertex_count) };
    let mut result: Vec<u8> = Vec::new();
    result.resize(bounds, 0u8);
    let size = unsafe {
        ffi::meshopt_encodeIndexBuffer(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uchar,
            result.len(),
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
        )
    };
    result.resize(size, 0u8);
    result
}

pub fn decode_index_buffer<T: Clone + Default>(encoded: &[u8], index_count: usize) -> Vec<T> {
    assert!(mem::size_of::<T>() == 2 || mem::size_of::<T>() == 4);
    let mut result: Vec<T> = Vec::new();
    result.resize(index_count, Default::default());
    let success = unsafe {
        ffi::meshopt_decodeIndexBuffer(
            result.as_mut_ptr() as *mut ::std::os::raw::c_void,
            index_count,
            mem::size_of::<T>(),
            encoded.as_ptr() as *const ::std::os::raw::c_uchar,
            encoded.len(),
        )
    };
    assert_eq!(success, 0); // GW-TODO: Respect error code and throw a Result object
    result
}

pub fn encode_vertex_buffer<T>(vertices: &[T]) -> Vec<u8> {
    let bounds =
        unsafe { ffi::meshopt_encodeVertexBufferBound(vertices.len(), mem::size_of::<T>()) };
    let mut result: Vec<u8> = Vec::new();
    result.resize(bounds, 0u8);
    let size = unsafe {
        ffi::meshopt_encodeVertexBuffer(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uchar,
            result.len(),
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            vertices.len(),
            mem::size_of::<T>(),
        )
    };
    result.resize(size, 0u8);
    result
}

pub fn decode_vertex_buffer<T: Clone + Default>(encoded: &[u8], vertex_count: usize) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    result.resize(vertex_count, Default::default());
    let success = unsafe {
        ffi::meshopt_decodeVertexBuffer(
            result.as_mut_ptr() as *mut ::std::os::raw::c_void,
            vertex_count,
            mem::size_of::<T>(),
            encoded.as_ptr() as *const ::std::os::raw::c_uchar,
            encoded.len(),
        )
    };
    assert_eq!(success, 0); // GW-TODO: Respect error code and throw a Result object
    result
}
