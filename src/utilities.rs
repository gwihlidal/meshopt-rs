use crate::{Error, Result, ffi};
use bytemuck::NoUninit;
use std::io::{Cursor, Read};
use std::mem;

/// Returns the bytes of a single value.
///
/// `T` must not contain padding or other uninitialized bytes, see [`bytemuck::NoUninit`].
#[inline(always)]
pub fn any_as_u8_slice<T: NoUninit>(p: &T) -> &[u8] {
    bytemuck::bytes_of(p)
}

/// Returns the bytes of a slice of values.
///
/// `T` must not contain padding or other uninitialized bytes, see [`bytemuck::NoUninit`].
#[inline(always)]
pub fn typed_to_bytes<T: NoUninit>(typed: &[T]) -> &[u8] {
    bytemuck::cast_slice(typed)
}

/// Largest vertex size (in bytes) the library supports.
pub(crate) const MAX_VERTEX_SIZE: usize = 256;

/// Checks that every index refers to one of `vertex_count` vertices.
///
/// Functions that take an index buffer panic if it contains an index that is out of range;
/// use this to report an error instead when processing meshes that may be invalid.
pub fn validate_indices(indices: &[u32], vertex_count: usize) -> Result<()> {
    match indices
        .iter()
        .find(|&&index| index as usize >= vertex_count)
    {
        Some(index) => Err(Error::memory_dynamic(format!(
            "index ({index}) must be less than vertex count ({vertex_count})"
        ))),
        None => Ok(()),
    }
}

/// Checks that `indices` is a triangle list referring to `vertex_count` vertices.
///
/// Functions that take a triangle list panic if the number of indices isn't a multiple of 3 or
/// if it contains an index that is out of range; use this to report an error instead when
/// processing meshes that may be invalid.
pub fn validate_triangles(indices: &[u32], vertex_count: usize) -> Result<()> {
    if !indices.len().is_multiple_of(3) {
        return Err(Error::memory_dynamic(format!(
            "index count ({}) must be a multiple of 3",
            indices.len()
        )));
    }
    validate_indices(indices, vertex_count)
}

/// Panics unless every index refers to one of `vertex_count` vertices.
///
/// The library does not validate indices, so every index buffer coming from safe code
/// has to go through this before it is handed over.
#[track_caller]
pub(crate) fn check_indices(indices: &[u32], vertex_count: usize) {
    if let Err(Error::Memory(message)) = validate_indices(indices, vertex_count) {
        panic!("{message}");
    }
}

/// Panics unless `indices` is a triangle list referring to `vertex_count` vertices.
#[track_caller]
pub(crate) fn check_triangles(indices: &[u32], vertex_count: usize) {
    if let Err(Error::Memory(message)) = validate_triangles(indices, vertex_count) {
        panic!("{message}");
    }
}

/// Returns whether `len` bytes hold `count` elements of `size` bytes that are `stride` bytes apart.
///
/// The last element doesn't need a full stride, so that an attribute can be selected from
/// interleaved vertex data by slicing it from the attribute's offset.
pub(crate) fn holds_strided(len: usize, count: usize, size: usize, stride: usize) -> bool {
    count == 0
        || (count - 1)
            .checked_mul(stride)
            .and_then(|start| start.checked_add(size))
            .is_some_and(|required| len >= required)
}

/// Panics unless `T` is a vertex type the library can process bytewise.
#[track_caller]
pub(crate) fn check_vertex_size<T>() {
    let size = mem::size_of::<T>();
    assert!(
        size > 0 && size <= MAX_VERTEX_SIZE,
        "vertex size ({size}) must be between 1 and {MAX_VERTEX_SIZE} bytes"
    );
}

pub fn convert_indices_32_to_16(indices: &[u32]) -> Result<Vec<u16>> {
    let mut result: Vec<u16> = Vec::with_capacity(indices.len());
    for index in indices {
        let Ok(index) = u16::try_from(*index) else {
            return Err(Error::memory(
                "index value must be <= 65535 when converting to 16-bit",
            ));
        };
        result.push(index);
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

/// Quantize a float into half-precision floating point value.
///
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
/// Representable magnitude range: [6e-5; 65504].
/// Maximum relative reconstruction error: 5e-4.
#[inline(always)]
pub fn quantize_half(v: f32) -> u16 {
    let ui = f32::to_bits(v);
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
///
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
/// Assumes N is in a valid mantissa precision range, which is 1..23
#[inline(always)]
pub fn quantize_float(v: f32, n: i32) -> f32 {
    let mut ui = f32::to_bits(v);

    let mask = (1 << (23 - n)) - 1;
    let round = (1 << (23 - n)) >> 1;

    let e = (ui & 0x7f80_0000) as i32;
    let rui: u32 = ((ui as i32 + round) & !mask) as u32;

    // round all numbers except inf/nan; this is important to make
    // sure nan doesn't overflow into -0
    ui = if e == 0x7f80_0000 { ui } else { rui };

    // flush denormals to zero
    ui = if e == 0 { 0 } else { ui };

    f32::from_bits(ui)
}

/// Reverse quantization of a half-precision (as defined by IEEE-754 fp16) floating point value
///
/// Preserves Inf/NaN, flushes denormals to zero
#[inline(always)]
pub fn dequantize_half(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let em = (h & 0x7fff) as u32;

    // bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
    let mut r = (em + (112 << 10)) << 13;

    // denormal: flush to zero
    if em < (1 << 10) {
        r = 0;
    }

    // infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
    // 112 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 255
    if em >= (31 << 10) {
        r += 112 << 23;
    }

    let bits = s | r;
    f32::from_bits(bits)
}

/// Computes a shared exponent suitable for mesh/cluster position quantization.
///
/// Given mesh or cluster bounds, computes a shared exponent that can be used to quantize any position
/// inside the bounds to a 24-bit integer grid. The resulting output can be stored as a compact bit-stream
/// to be decoded directly in shaders, or to be used as an input to RT BVH builders, for example via
/// `D3D12_VERTEX_FORMAT_COMPRESSED1` in DXR2 (`max_bits` = 16).
///
/// To quantize positions, compute `scale = pow(2, exponent)` and `iv = int(round(v / scale))`.
/// The resulting integer can be stored as signed 24-bit, or as an unsigned offset from a signed 24-bit
/// anchor value, shared between all positions.
///
/// * `min`/`max` specify the axis-aligned bounding box of the mesh or cluster
/// * `min_exp` specifies the minimum value for the returned exponent, limiting precision to reduce size;
///   e.g. `min_exp` = -10 limits precision to 1mm given metric units
/// * `max_bits` specifies the maximum allowed number of bits for the quantized integer range (offset from
///   anchor is an unsigned integer up to 2^`max_bits`-1)
pub fn compute_position_exponent(min: [f32; 3], max: [f32; 3], min_exp: i32, max_bits: i32) -> i32 {
    unsafe { ffi::meshopt_computePositionExponent(min.as_ptr(), max.as_ptr(), min_exp, max_bits) }
}

#[inline(always)]
pub fn rcp_safe(v: f32) -> f32 {
    if v.abs() as u32 == 0 { 0f32 } else { 1f32 / v }
}

/// Describes where vertex positions live inside a buffer of interleaved vertex data.
///
/// The layout is validated on construction and again whenever the data is handed to the
/// library, so changing the fields to something that doesn't match the data results in a panic.
pub struct VertexDataAdapter<'a> {
    pub reader: Cursor<&'a [u8]>,
    pub vertex_count: usize,
    pub vertex_stride: usize,
    pub position_offset: usize,
}

impl<'a> VertexDataAdapter<'a> {
    /// Creates an adapter over `data`, where each vertex takes `vertex_stride` bytes and stores
    /// its position as 3 floats starting `position_offset` bytes into the vertex.
    ///
    /// Fails if the layout doesn't describe a valid position stream: `vertex_stride` must be a
    /// multiple of 4 no larger than 256 that evenly divides the data, the position must fit
    /// inside the vertex, and the positions must be aligned for `f32`.
    pub fn new(
        data: &'a [u8],
        vertex_stride: usize,
        position_offset: usize,
    ) -> Result<VertexDataAdapter<'a>> {
        Self::check_layout(data, vertex_stride, position_offset)?;
        if !data.len().is_multiple_of(vertex_stride) {
            return Err(Error::memory_dynamic(format!(
                "vertex data length ({}) must be evenly divisible by vertex_stride ({})",
                data.len(),
                vertex_stride
            )));
        }

        Ok(VertexDataAdapter {
            reader: Cursor::new(data),
            vertex_count: data.len() / vertex_stride,
            vertex_stride,
            position_offset,
        })
    }

    fn check_layout(data: &[u8], vertex_stride: usize, position_offset: usize) -> Result<()> {
        const POSITION_SIZE: usize = mem::size_of::<[f32; 3]>();

        if vertex_stride == 0
            || vertex_stride > MAX_VERTEX_SIZE
            || !vertex_stride.is_multiple_of(mem::size_of::<f32>())
        {
            return Err(Error::memory_dynamic(format!(
                "vertex_stride ({vertex_stride}) must be a non-zero multiple of 4 no larger than {MAX_VERTEX_SIZE}"
            )));
        }
        if vertex_stride < POSITION_SIZE || position_offset > vertex_stride - POSITION_SIZE {
            return Err(Error::memory_dynamic(format!(
                "position at position_offset ({position_offset}) must fit in vertex_stride ({vertex_stride})"
            )));
        }
        if !data.is_empty()
            && !(data.as_ptr() as usize + position_offset).is_multiple_of(mem::align_of::<f32>())
        {
            return Err(Error::memory("vertex positions must be aligned to 4 bytes"));
        }
        Ok(())
    }

    /// The vertex data this adapter was created with.
    pub fn data(&self) -> &'a [u8] {
        self.reader.get_ref()
    }

    pub fn xyz_f32_at(&self, vertex: usize) -> Result<[f32; 3]> {
        if vertex >= self.vertex_count {
            return Err(Error::memory_dynamic(format!(
                "vertex index ({}) must be less than total vertex count ({})",
                vertex, self.vertex_count
            )));
        }
        let bytes = vertex
            .checked_mul(self.vertex_stride)
            .and_then(|start| start.checked_add(self.position_offset))
            .and_then(|start| self.data().get(start..)?.get(..mem::size_of::<[f32; 3]>()))
            .ok_or_else(|| Error::memory("vertex position is out of bounds of the vertex data"))?;
        Ok(bytemuck::pod_read_unaligned(bytes))
    }

    /// Pointer to the position of the first vertex.
    ///
    /// The pointer is valid for reading `vertex_count` positions that are `vertex_stride`
    /// bytes apart, as long as the vertex data is.
    ///
    /// Panics if the fields no longer describe a valid layout for the data; this is what every
    /// function that takes an adapter relies on.
    pub fn pos_ptr(&self) -> *const f32 {
        let data = self.data();
        if let Err(error) = Self::check_layout(data, self.vertex_stride, self.position_offset) {
            panic!("invalid vertex data adapter: {error}");
        }
        assert!(
            self.vertex_count <= data.len() / self.vertex_stride,
            "vertex_count ({}) must not exceed the {} vertices in the vertex data",
            self.vertex_count,
            data.len() / self.vertex_stride
        );
        // an empty adapter has no vertex to offset into
        if self.vertex_count == 0 {
            return std::ptr::NonNull::dangling().as_ptr();
        }
        data[self.position_offset..].as_ptr().cast()
    }
}

impl Read for VertexDataAdapter<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::result::Result<usize, std::io::Error> {
        self.reader.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Vertex, VertexDataAdapter, typed_to_bytes};

    #[test]
    fn test_xyz_f32_at() {
        let vertices = vec![
            Vertex {
                p: [1.0, 2.0, 3.0],
                n: [0.0; 3],
                t: [0.0; 2],
            },
            Vertex {
                p: [4.0, 5.0, 6.0],
                n: [0.0; 3],
                t: [0.0; 2],
            },
        ];

        let adapter = VertexDataAdapter::new(
            typed_to_bytes(&vertices),
            size_of::<Vertex>(),
            std::mem::offset_of!(Vertex, p),
        )
        .unwrap();

        let p = adapter.xyz_f32_at(0).unwrap();
        assert_eq!(p, [1.0, 2.0, 3.0]);
        let p = adapter.xyz_f32_at(1).unwrap();
        assert_eq!(p, [4.0, 5.0, 6.0]);

        adapter.xyz_f32_at(2).expect_err("should fail");
    }

    #[test]
    fn test_compute_position_exponent() {
        // expected values match meshoptimizer's own tests
        let minv = [-1.0, 2.0, 10.0];
        let maxv = [1.0, 2.5, 20.0];
        assert_eq!(compute_position_exponent(minv, maxv, -16, 16), -12);
        assert_eq!(compute_position_exponent(minv, maxv, -10, 16), -10);

        let minirr = [-std::f32::consts::PI, -std::f32::consts::E, -1.413_213_6];
        let maxirr = [std::f32::consts::PI, std::f32::consts::E, 1.413_213_6];
        assert_eq!(compute_position_exponent(minirr, maxirr, -16, 16), -13);
    }

    #[test]
    fn quantize_roundtrip() {
        for i in u16::MIN..u16::MAX {
            let f = dequantize_half(i);
            let q = quantize_half(f);
            // dont care about denormals
            if !f.is_normal() {
                continue;
            }
            assert_eq!(i, q, "quantization error for {i}: {f} -> {q}");
        }
    }

    #[test]
    fn test_adapter_rejects_invalid_layout() {
        let data = [0f32; 8];
        let bytes = typed_to_bytes(&data);

        assert!(VertexDataAdapter::new(bytes, 16, 0).is_ok());
        assert!(VertexDataAdapter::new(bytes, 16, 4).is_ok());
        // zero, unaligned or oversized stride
        assert!(VertexDataAdapter::new(bytes, 0, 0).is_err());
        assert!(VertexDataAdapter::new(&bytes[..30], 15, 0).is_err());
        assert!(VertexDataAdapter::new(&[0u8; 1024], 512, 0).is_err());
        // data isn't a whole number of vertices
        assert!(VertexDataAdapter::new(&bytes[..28], 16, 0).is_err());
        // position doesn't fit in the vertex
        assert!(VertexDataAdapter::new(bytes, 16, 8).is_err());
        assert!(VertexDataAdapter::new(bytes, 8, 0).is_err());
        // positions aren't aligned
        assert!(VertexDataAdapter::new(&bytes[1..17], 16, 0).is_err());
        assert!(VertexDataAdapter::new(bytes, 16, 2).is_err());
    }

    #[test]
    fn test_empty_adapter() {
        let adapter = VertexDataAdapter::new(&[], 16, 4).unwrap();
        assert_eq!(adapter.vertex_count, 0);
        assert!(!adapter.pos_ptr().is_null());
        adapter.xyz_f32_at(0).expect_err("should fail");
    }

    #[test]
    fn test_convert_indices_32_to_16() {
        assert_eq!(convert_indices_32_to_16(&[0, 65535]).unwrap(), [0, 65535]);
        assert!(convert_indices_32_to_16(&[65536]).is_err());
    }

    #[test]
    #[should_panic(expected = "vertex_count (3) must not exceed the 2 vertices")]
    fn test_adapter_modified_count() {
        let data = [0f32; 8];
        let mut adapter = VertexDataAdapter::new(typed_to_bytes(&data), 16, 0).unwrap();
        adapter.vertex_count = 3;
        adapter.pos_ptr();
    }

    #[test]
    #[should_panic(expected = "invalid vertex data adapter")]
    fn test_adapter_modified_stride() {
        let data = [0f32; 8];
        let mut adapter = VertexDataAdapter::new(typed_to_bytes(&data), 16, 0).unwrap();
        adapter.vertex_stride = 8;
        adapter.pos_ptr();
    }

    #[test]
    fn test_holds_strided() {
        assert!(holds_strided(0, 0, 12, 16));
        assert!(holds_strided(32, 2, 12, 16));
        // the last element only needs its own size
        assert!(holds_strided(28, 2, 12, 16));
        assert!(!holds_strided(27, 2, 12, 16));
        assert!(!holds_strided(usize::MAX, usize::MAX, 12, 16));
    }

    #[test]
    fn test_validate_indices() {
        assert!(validate_indices(&[0, 1, 2, 2], 3).is_ok());
        assert!(validate_indices(&[], 0).is_ok());
        let error = validate_indices(&[0, 1, 3], 3).unwrap_err();
        assert!(error.to_string().contains("index (3)"));

        assert!(validate_triangles(&[0, 1, 2], 3).is_ok());
        assert!(validate_triangles(&[0, 1, 2, 2], 3).is_err());
        assert!(validate_triangles(&[0, 1, 3], 3).is_err());
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_check_indices() {
        check_indices(&[0, 1, 3], 3);
    }
}
