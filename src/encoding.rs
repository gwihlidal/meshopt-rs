use crate::ffi;
use crate::{Error, Result};
use std::mem;

pub fn encode_index_buffer(indices: &[u32], vertex_count: usize) -> Result<Vec<u8>> {
    let bounds = unsafe { ffi::meshopt_encodeIndexBufferBound(indices.len(), vertex_count) };
    let mut result: Vec<u8> = vec![0; bounds];
    let size = unsafe {
        ffi::meshopt_encodeIndexBuffer(
            result.as_mut_ptr() as *mut ::std::os::raw::c_uchar,
            result.len(),
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
        )
    };
    result.resize(size, 0u8);
    Ok(result)
}

pub fn decode_index_buffer<T: Clone + Default>(
    encoded: &[u8],
    index_count: usize,
) -> Result<Vec<T>> {
    if mem::size_of::<T>() == 2 || mem::size_of::<T>() == 4 {
        let mut result: Vec<T> = vec![Default::default(); index_count];
        let result_code = unsafe {
            ffi::meshopt_decodeIndexBuffer(
                result.as_mut_ptr() as *mut ::std::os::raw::c_void,
                index_count,
                mem::size_of::<T>(),
                encoded.as_ptr() as *const ::std::os::raw::c_uchar,
                encoded.len(),
            )
        };
        match result_code {
            0 => Ok(result),
            _ => Err(Error::native(result_code)),
        }
    } else {
        Err(Error::memory(
            "size of result type must be 2 or 4 bytes wide",
        ))
    }
}

pub fn encode_vertex_buffer<T>(vertices: &[T]) -> Result<Vec<u8>> {
    let bounds =
        unsafe { ffi::meshopt_encodeVertexBufferBound(vertices.len(), mem::size_of::<T>()) };
    let mut result: Vec<u8> = vec![0; bounds];
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
    Ok(result)
}

pub fn decode_vertex_buffer<T: Clone + Default>(
    encoded: &[u8],
    vertex_count: usize,
) -> Result<Vec<T>> {
    let mut result: Vec<T> = vec![Default::default(); vertex_count];
    let result_code = unsafe {
        ffi::meshopt_decodeVertexBuffer(
            result.as_mut_ptr() as *mut ::std::os::raw::c_void,
            vertex_count,
            mem::size_of::<T>(),
            encoded.as_ptr() as *const ::std::os::raw::c_uchar,
            encoded.len(),
        )
    };
    match result_code {
        0 => Ok(result),
        _ => Err(Error::native(result_code)),
    }
}

pub struct Header {
    pub magic: [i8; 4], // OPTM

    pub group_count: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_data_size: u32,
    pub index_data_size: u32,

    pub pos_offset: [f32; 3],
    pub pos_scale: [f32; 3],
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],

    pub(crate) reserved: [u32; 2],
}

pub struct Object {
    pub index_offset: u32,
    pub index_count: u32,
    pub material_length: u32,
    pub(crate) reserved: u32,
}

pub fn calc_pos_offset_and_scale(positions: &[f32]) -> ([f32; 3], f32) {
    let mut pos_offset: [f32; 3] = [std::f32::MAX, std::f32::MAX, std::f32::MAX];
    let mut pos_scale = 0f32;

    for i in 0..(positions.len() / 3) {
        pos_offset = [
            pos_offset[0].min(positions[(i * 3) + 0]),
            pos_offset[1].min(positions[(i * 3) + 1]),
            pos_offset[2].min(positions[(i * 3) + 2]),
        ];
    }

    for i in 0..(positions.len() / 3) {
        pos_scale = pos_scale.max(positions[(i * 3) + 0] - pos_offset[0]);
        pos_scale = pos_scale.max(positions[(i * 3) + 1] - pos_offset[1]);
        pos_scale = pos_scale.max(positions[(i * 3) + 2] - pos_offset[2]);
    }

    (pos_offset, pos_scale)
}
