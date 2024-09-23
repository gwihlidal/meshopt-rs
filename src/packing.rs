use crate::{quantize_half, quantize_snorm};
use float_cmp::ApproxEqUlps;

pub trait DecodePosition {
    fn decode_position(&self) -> [f32; 3];
}

impl DecodePosition for [f32; 3] {
    fn decode_position(&self) -> [f32; 3] {
        *self
    }
}

pub trait FromVertex {
    fn fill_from_vertex(&mut self, vertex: &Vertex);
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub struct PackedVertex {
    /// Unsigned 16-bit value, use `pos_offset/pos_scale` to unpack
    pub p: [u16; 4],

    /// Normalized signed 8-bit value
    pub n: [i8; 4],

    /// Unsigned 16-bit value, use `uv_offset/uv_scale` to unpack
    pub t: [u16; 2],
}

impl FromVertex for PackedVertex {
    fn fill_from_vertex(&mut self, vertex: &Vertex) {
        self.p[0] = quantize_half(vertex.p[0]);
        self.p[1] = quantize_half(vertex.p[1]);
        self.p[2] = quantize_half(vertex.p[2]);
        self.p[3] = 0u16;

        self.n[0] = quantize_snorm(vertex.n[0], 8) as i8;
        self.n[1] = quantize_snorm(vertex.n[1], 8) as i8;
        self.n[2] = quantize_snorm(vertex.n[2], 8) as i8;
        self.n[3] = 0i8;

        self.t[0] = quantize_half(vertex.t[0]);
        self.t[1] = quantize_half(vertex.t[1]);
    }
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub struct PackedVertexOct {
    pub p: [u16; 3],
    pub n: [u8; 2], // octahedron encoded normal, aliases .pw
    pub t: [u16; 2],
}

impl FromVertex for PackedVertexOct {
    fn fill_from_vertex(&mut self, vertex: &Vertex) {
        self.p[0] = quantize_half(vertex.p[0]);
        self.p[1] = quantize_half(vertex.p[1]);
        self.p[2] = quantize_half(vertex.p[2]);

        let nsum = vertex.n[0].abs() + vertex.n[1].abs() + vertex.n[2].abs();
        let nx = vertex.n[0] / nsum;
        let ny = vertex.n[1] / nsum;
        let nz = vertex.n[2];

        let nu = if nz >= 0f32 {
            nx
        } else {
            (1f32 - ny.abs()) * if nx >= 0f32 { 1f32 } else { -1f32 }
        };

        let nv = if nz >= 0f32 {
            ny
        } else {
            (1f32 - nx.abs()) * if ny >= 0f32 { 1f32 } else { -1f32 }
        };

        self.n[0] = quantize_snorm(nu, 8) as u8;
        self.n[1] = quantize_snorm(nv, 8) as u8;

        self.t[0] = quantize_half(vertex.t[0]);
        self.t[1] = quantize_half(vertex.t[1]);
    }
}

#[derive(Default, Debug, Copy, Clone, PartialOrd)]
#[repr(C)]
/// A basic Vertex type that can be used with most mesh processing functions.
/// You don't _need_ to use this type, you can use your own type by implementing
/// the `DecodePosition` trait and making a [`VertexDataAdapter`] from slices of it.
pub struct Vertex {
    pub p: [f32; 3],
    pub n: [f32; 3],
    pub t: [f32; 2],
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Vertex) -> bool {
        self.p[0].approx_eq_ulps(&other.p[0], 2)
            && self.p[1].approx_eq_ulps(&other.p[1], 2)
            && self.p[2].approx_eq_ulps(&other.p[2], 2)
            && self.n[0].approx_eq_ulps(&other.n[0], 2)
            && self.n[1].approx_eq_ulps(&other.n[1], 2)
            && self.n[2].approx_eq_ulps(&other.n[2], 2)
            && self.t[0].approx_eq_ulps(&other.t[0], 2)
            && self.t[1].approx_eq_ulps(&other.t[1], 2)
    }
}

impl Eq for Vertex {}

impl Vertex {}

impl DecodePosition for Vertex {
    fn decode_position(&self) -> [f32; 3] {
        self.p
    }
}

pub fn pack_vertices<T: FromVertex + Default + Clone>(input: &[Vertex]) -> Vec<T> {
    let mut vertices: Vec<T> = vec![T::default(); input.len()];
    for i in 0..input.len() {
        vertices[i].fill_from_vertex(&input[i]);
    }
    vertices
}
