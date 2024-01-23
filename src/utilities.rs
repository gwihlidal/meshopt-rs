use crate::{Error, Result};
use std::io::{Cursor, Read};

#[inline(always)]
pub fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    typed_to_bytes(std::slice::from_ref(p))
}

#[inline(always)]
pub fn typed_to_bytes<T: Sized>(typed: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(typed.as_ptr().cast(), std::mem::size_of_val(typed)) }
}

pub fn convert_indices_32_to_16(indices: &[u32]) -> Result<Vec<u16>> {
    let mut result: Vec<u16> = Vec::with_capacity(indices.len());
    for index in indices {
        if *index > 65536 {
            return Err(Error::memory(
                "index value must be <= 65536 when converting to 16-bit",
            ));
        }
        result.push(*index as u16);
    }
    Ok(result)
}

pub fn convert_indices_16_to_32(indices: &[u16]) -> Result<Vec<u32>> {
    let mut result: Vec<u32> = Vec::with_capacity(indices.len());
    for index in indices {
        result.push(u32::from(*index));
    }
    Ok(result)
}

/// Quantize a float in [0..1] range into an N-bit fixed point unorm value.
///
/// Assumes reconstruction function (q / (2^N-1)), which is the case for
/// fixed-function normalized fixed point conversion.
///
/// Maximum reconstruction error: 1/2^(N+1).
#[inline(always)]
pub fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << n) - 1i32) as f32;
    let v = if v >= 0f32 { v } else { 0f32 };
    let v = if v <= 1f32 { v } else { 1f32 };
    (v * scale + 0.5f32) as i32
}

/// Quantize a float in [-1..1] range into an N-bit fixed point snorm value.
///
/// Assumes reconstruction function (q / (2^(N-1)-1)), which is the case for
/// fixed-function normalized fixed point conversion (except early OpenGL versions).
///
/// Maximum reconstruction error: 1/2^N.
#[inline(always)]
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

/// Quantize a float into half-precision floating point value.
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
/// Representable magnitude range: [6e-5; 65504].
/// Maximum relative reconstruction error: 5e-4.
#[inline(always)]
pub fn quantize_half(v: f32) -> u16 {
    let u = FloatUInt { fl: v };
    let ui = unsafe { u.ui };
    let s = ((ui >> 16) & 0x8000) as i32;
    let em = (ui & 0x7fff_ffff) as i32;

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

/// Quantize a float into a floating point value with a limited number of significant mantissa bits.
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
/// Assumes N is in a valid mantissa precision range, which is 1..23
#[inline(always)]
pub fn quantize_float(v: f32, n: i32) -> f32 {
    let mut u = FloatUInt { fl: v };
    let mut ui = unsafe { u.ui };

    let mask = (1 << (23 - n)) - 1;
    let round = (1 << (23 - n)) >> 1;

    let e = (ui & 0x7f80_0000) as i32;
    let rui: u32 = ((ui as i32 + round) & !mask) as u32;

    // round all numbers except inf/nan; this is important to make
    // sure nan doesn't overflow into -0
    ui = if e == 0x7f80_0000 { ui } else { rui };

    // flush denormals to zero
    ui = if e == 0 { 0 } else { ui };

    u.ui = ui;
    unsafe { u.fl }
}

#[inline(always)]
pub fn rcp_safe(v: f32) -> f32 {
    if v.abs() as u32 == 0 {
        0f32
    } else {
        1f32 / v
    }
}

pub struct VertexDataAdapter<'a> {
    pub reader: Cursor<&'a [u8]>,
    pub vertex_count: usize,
    pub vertex_stride: usize,
    pub position_offset: usize,
}

impl<'a> VertexDataAdapter<'a> {
    pub fn new(
        data: &'a [u8],
        vertex_stride: usize,
        position_offset: usize,
    ) -> Result<VertexDataAdapter<'a>> {
        let vertex_count = data.len() / vertex_stride;
        if data.len() % vertex_stride != 0 {
            Err(Error::memory_dynamic(format!(
                "vertex data length ({}) must be evenly divisible by vertex_stride ({})",
                data.len(),
                vertex_stride
            )))
        } else if position_offset >= vertex_stride {
            Err(Error::memory_dynamic(format!(
                "position_offset ({}) must be smaller than vertex_stride ({})",
                position_offset, vertex_stride
            )))
        } else {
            Ok(VertexDataAdapter {
                reader: Cursor::new(data),
                vertex_count,
                vertex_stride,
                position_offset,
            })
        }
    }

    pub fn xyz_f32_at(&mut self, vertex: usize) -> Result<&[f32]> {
        if vertex <= self.vertex_count {
            Err(Error::memory_dynamic(format!(
                "vertex index ({}) must be less than total vertex count ({})",
                vertex, self.vertex_count
            )))
        } else {
            let reader_pos = self.reader.position();
            let vertex_offset = vertex * self.vertex_stride;
            self.reader
                .set_position((vertex_offset + self.position_offset) as u64);
            let mut scratch = [0u8; 12];
            self.reader.read_exact(&mut scratch)?;
            let position =
                unsafe { std::slice::from_raw_parts(scratch.as_ptr().cast::<f32>(), 3) };
            self.reader.set_position(reader_pos);
            Ok(position)
        }
    }

    pub fn pos_ptr(&self) -> *const f32 {
        let vertex_data = self.reader.get_ref();
        let vertex_data = vertex_data.as_ptr().cast::<u8>();
        let positions = unsafe { vertex_data.add(self.position_offset) };
        positions.cast()
    }
}

impl<'a> Read for VertexDataAdapter<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::result::Result<usize, std::io::Error> {
        self.reader.read(buf)
    }
}
