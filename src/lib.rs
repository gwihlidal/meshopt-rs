extern crate failure;
extern crate float_cmp;

pub mod analyze;
pub mod clusterize;
pub mod encoding;
pub mod error;
pub mod ffi;
pub mod optimize;
pub mod packing;
pub mod remap;
pub mod shadow;
pub mod simplify;
pub mod stripify;
pub mod utilities;

pub use crate::analyze::*;
pub use crate::clusterize::*;
pub use crate::encoding::*;
pub use crate::error::*;
pub use crate::optimize::*;
pub use crate::packing::*;
pub use crate::remap::*;
pub use crate::shadow::*;
pub use crate::simplify::*;
pub use crate::stripify::*;
pub use crate::utilities::*;
use std::marker::PhantomData;

/// Vertex attribute stream, similar to glVertexPointer
///
/// Each element takes size bytes, with stride controlling
/// the spacing between successive elements.
#[derive(Debug, Copy, Clone)]
pub struct VertexStream<'a> {
    /// Pointer to buffer which contains vertex data.
    pub data: *const u8,
    /// Space between vertices inside the buffer (in bytes).
    pub stride: usize,
    /// The size in bytes of the vertex attribute this Stream is representing.
    pub size: usize,

    _marker: PhantomData<&'a ()>,
}

impl<'a> VertexStream<'a> {
    /// Create a `VertexStream` for a buffer consisting only of elements of type `T`.
    pub fn new<T>(ptr: *const T) -> VertexStream<'a> {
        Self::new_with_stride::<T, T>(ptr, std::mem::size_of::<T>())
    }

    /// Create a `VertexStream` for a buffer that contains elements of type `VertexType`.
    ///
    /// The buffer pointed to by `ptr` starts with one value of `T`, the next value of T
    /// is `*(ptr + stride)`.
    ///
    /// (The `VertexType` does not need to be a concrete type,
    /// it is only used here to avoid casts on the caller side).
    pub fn new_with_stride<T, VertexType>(
        ptr: *const VertexType,
        stride: usize,
    ) -> VertexStream<'a> {
        VertexStream {
            data: ptr as *const u8,
            stride,
            size: std::mem::size_of::<T>(),

            _marker: PhantomData,
        }
    }
}
