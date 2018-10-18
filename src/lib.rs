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
            vertex_count
        );
    }
    optimized
}

pub fn optimize_vertex_cache_in_place(indices: &mut [u32], vertex_count: usize) {
    let mut optimized: Vec<u32> = Vec::with_capacity(indices.len());
    optimized.resize(indices.len(), 0u32);
    unsafe {
        ffi::meshopt_optimizeVertexCache(
            indices.as_mut_ptr() as *mut ::std::os::raw::c_uint,
            indices.as_ptr() as *const ::std::os::raw::c_uint,
            indices.len(),
            vertex_count
        );
    }
}

pub fn optimize_vertex_cache_fifo(indices: &[u32], vertex_count: usize, cache_size: u32) -> Vec<u32> {
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

pub fn optimize_vertex_cache_fifo_in_place(indices: &mut [u32], vertex_count: usize, cache_size: u32) {
    let mut optimized: Vec<u32> = Vec::with_capacity(indices.len());
    optimized.resize(indices.len(), 0u32);
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

// Quantize a float in [0..1] range into an N-bit fixed point unorm value
// Assumes reconstruction function (q / (2^N-1)), which is the case for fixed-function normalized fixed point conversion
// Maximum reconstruction error: 1/2^(N+1)
#[inline]
pub fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << n) - 1i32) as f32;
    let v = if v >= 0f32 {
        v
    } else {
        0f32
    };
    let v = if v <= 1f32 {
        v
    } else {
        1f32
    };

    (v * scale + 0.5f32) as i32
}

// Quantize a float in [-1..1] range into an N-bit fixed point snorm value
// Assumes reconstruction function (q / (2^(N-1)-1)), which is the case for fixed-function normalized fixed point conversion (except early OpenGL versions)
// Maximum reconstruction error: 1/2^N
#[inline]
pub fn quantize_snorm(v: f32, n: u32) -> i32 {
    let scale = ((1 << (n - 1)) - 1) as f32;
    let round = if v >= 0f32 {
        0.5f32
    } else {
        -0.5f32
    };
    let v = if v >= -1f32 {
        v
    } else {
        -1f32
    };
    let v = if v <= 1f32 {
        v
    } else {
        1f32
    };

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
