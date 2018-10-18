use std::mem;

pub mod ffi;

pub type VertexCacheStatistics = ffi::meshopt_VertexCacheStatistics;
pub type VertexFetchStatistics = ffi::meshopt_VertexFetchStatistics;
pub type OverdrawStatistics = ffi::meshopt_OverdrawStatistics;

pub trait DecodePosition {
    fn decode_position(&self) -> [f32; 3];
}

pub fn analyze_vertex_cache(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
    warp_size: u32,
    prim_group_size: u32,
) -> VertexCacheStatistics {
    unsafe {
        ffi::meshopt_analyzeVertexCache(
            indices.as_ptr() as *mut ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            cache_size,
            warp_size,
            prim_group_size,
        )
    }
}

pub fn analyze_vertex_fetch(
    indices: &[u32],
    vertex_count: usize,
    vertex_size: usize,
) -> VertexFetchStatistics {
    unsafe {
        ffi::meshopt_analyzeVertexFetch(
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            vertex_size,
        )
    }
}

pub fn analyze_overdraw<T: DecodePosition>(indices: &[u32], vertices: &[T]) -> OverdrawStatistics {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_analyzeOverdraw(
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            positions.as_ptr() as *const f32,
            positions.len(),
            mem::size_of::<f32>() * 3,
        )
    }
}

/// Vertex transform cache optimizer
/// Reorders indices to reduce the number of GPU vertex shader invocations
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    let mut optimized: Vec<u32> = Vec::with_capacity(indices.len());
    optimized.resize(indices.len(), 0u32);
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            optimized.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        );
    }
    optimized
}

pub fn optimize_vertex_cache_in_place(indices: &mut [u32], vertex_count: usize) {
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
        );
    }
}

pub fn optimize_vertex_cache_fifo(
    indices: &[u32],
    vertex_count: usize,
    cache_size: u32,
) -> Vec<u32> {
    let mut optimized: Vec<u32> = Vec::with_capacity(indices.len());
    optimized.resize(indices.len(), 0u32);
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            optimized.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
    optimized
}

pub fn optimize_vertex_cache_fifo_in_place(
    indices: &mut [u32],
    vertex_count: usize,
    cache_size: u32,
) {
    unsafe {
        ffi::meshopt_optimizeVertexCacheFifo(
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count,
            cache_size,
        );
    }
}

/// Vertex fetch cache optimizer
/// Reorders vertices and changes indices to reduce the amount of GPU memory fetches during vertex processing
///
/// destination must contain enough space for the resulting vertex buffer (vertex_count elements)
/// indices is used both as an input and as an output index buffer
pub fn optimize_vertex_fetch<T: Clone + Default>(indices: &mut [u32], vertices: &[T]) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    result.resize(vertices.len(), T::default());
    let next_vertex = unsafe {
        ffi::meshopt_optimizeVertexFetch(
            result.as_mut_ptr() as *mut ::std::os::raw::c_void,
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.len(),
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            vertices.len(),
            mem::size_of::<T>(),
        )
    };
    result.resize(next_vertex, T::default());
    result
}

pub fn optimize_vertex_fetch_in_place<T>(indices: &mut [u32], vertices: &mut [T]) -> usize {
    let next_vertex = unsafe {
        ffi::meshopt_optimizeVertexFetch(
            vertices.as_mut_ptr() as *mut ::std::os::raw::c_void,
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.len(),
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            vertices.len(),
            mem::size_of::<T>(),
        )
    };
    next_vertex
}

/// Overdraw optimizer
/// Reorders indices to reduce the number of GPU vertex shader invocations and the pixel overdraw
///
/// destination must contain enough space for the resulting index buffer (index_count elements)
/// indices must contain index data that is the result of optimizeVertexCache (*not* the original mesh indices!)
/// vertex_positions should have float3 position in the first 12 bytes of each vertex - similar to glVertexPointer
/// threshold indicates how much the overdraw optimizer can degrade vertex cache efficiency (1.05 = up to 5%) to reduce overdraw more efficiently
pub fn optimize_overdraw_in_place<T: DecodePosition>(indices: &mut [u32], vertices: &[T], threshold: f32) {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_optimizeOverdraw(
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            positions.as_ptr() as *const f32,
            positions.len(),
            mem::size_of::<f32>() * 3,
            threshold,
        );
    }
}

pub fn encode_index_buffer(indices: &[u32], vertex_count: usize) -> Vec<u8> {
    // TODO: Support using either 32 or 16 bit indices
    //assert!(mem::size_of::<T>() == 2 || mem::size_of::<T>() == 4);
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

/// Index buffer decoder
/// Decodes index data from an array of bytes generated by meshopt_encodeIndexBuffer
/// Returns 0 if decoding was successful, and an error code otherwise
///
/// destination must contain enough space for the resulting index buffer (index_count elements)
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
    assert_eq!(success, 0); // TODO: Respect error code and throw a Result object
    result
}

pub fn encode_vertex_buffer<T>(vertices: &[T]) -> Vec<u8> {
    let bounds = unsafe { ffi::meshopt_encodeVertexBufferBound(vertices.len(), mem::size_of::<T>()) };
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

/// Vertex buffer decoder
/// Decodes vertex data from an array of bytes generated by meshopt_encodeVertexBuffer
/// Returns 0 if decoding was successful, and an error code otherwise
///
/// destination must contain enough space for the resulting vertex buffer (vertex_count * vertex_size bytes)
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
    assert_eq!(success, 0); // TODO: Respect error code and throw a Result object
    result
}

pub fn convert_indices_32_to_16(indices: &[u32]) -> Vec<u16> {
    let mut result: Vec<u16> = Vec::with_capacity(indices.len());
    for index in indices {
        assert!(index <= &65536);
        result.push(*index as u16);
    }
    result
}

pub fn convert_indices_16_to_32(indices: &[u16]) -> Vec<u32> {
    let mut result: Vec<u32> = Vec::with_capacity(indices.len());
    for index in indices {
        result.push(*index as u32);
    }
    result
}

// Quantize a float in [0..1] range into an N-bit fixed point unorm value
// Assumes reconstruction function (q / (2^N-1)), which is the case for fixed-function normalized fixed point conversion
// Maximum reconstruction error: 1/2^(N+1)
#[inline]
pub fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << n) - 1i32) as f32;
    let v = if v >= 0f32 { v } else { 0f32 };
    let v = if v <= 1f32 { v } else { 1f32 };

    (v * scale + 0.5f32) as i32
}

// Quantize a float in [-1..1] range into an N-bit fixed point snorm value
// Assumes reconstruction function (q / (2^(N-1)-1)), which is the case for fixed-function normalized fixed point conversion (except early OpenGL versions)
// Maximum reconstruction error: 1/2^N
#[inline]
pub fn quantize_snorm(v: f32, n: u32) -> i32 {
    let scale = ((1 << (n - 1)) - 1) as f32;
    let round = if v >= 0f32 { 0.5f32 } else { -0.5f32 };
    let v = if v >= -1f32 { v } else { -1f32 };
    let v = if v <= 1f32 { v } else { 1f32 };

    (v * scale + round) as i32
}

#[repr(C)]
union FloatUInt {
    fl: f32,
    ui: u32,
}

// Quantize a float into half-precision floating point value
// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest
// Representable magnitude range: [6e-5; 65504]
// Maximum relative reconstruction error: 5e-4
#[inline]
pub fn quantize_half(v: f32) -> u16 {
    let u = FloatUInt { fl: v };
    let ui = unsafe { u.ui };
    let s = ((ui >> 16) & 0x8000) as i32;
    let em = (ui & 0x7fffffff) as i32;

    // bias exponent and round to nearest; 112 is relative exponent bias (127-15)
    let mut h = (em - (112 << 23) + (1 << 12)) >> 13;

    // underflow: flush to zero; 113 encodes exponent -14
    h = if em < (113 << 23) { 0 } else { h };

    // overflow: infinity; 143 encodes exponent 16
    h = if em >= (143 << 23) { 0x7c00 } else { h };

    // NaN; note that we convert all types of NaN to qNaN
    h = if em > (255 << 23) { 0x7e00 } else { h };

    (s | h) as u16
}

// Quantize a float into a floating point value with a limited number of significant mantissa bits
// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest
// Assumes N is in a valid mantissa precision range, which is 1..23
pub fn quantize_float(v: f32, n: i32) -> f32 {
    let mut u = FloatUInt { fl: v };
    let mut ui = unsafe { u.ui };

    let mask = ((1 << (23 - n)) - 1) as i32;
    let round = ((1 << (23 - n)) >> 1) as i32;

    let e = (ui & 0x7f800000) as i32;
    let rui: u32 = ((ui as i32 + round) & !mask) as u32;

    // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
    ui = if e == 0x7f800000 { ui } else { rui };

    // flush denormals to zero
    ui = if e == 0 { 0 } else { ui };

    u.ui = ui;
    unsafe { u.fl }
}
