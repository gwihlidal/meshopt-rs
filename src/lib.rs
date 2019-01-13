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

/// Vertex attribute stream, similar to glVertexPointer
/// 
/// Each element takes size bytes, with stride controlling
/// the spacing between successive elements.
#[derive(Debug, Copy, Clone)]
pub struct VertexStream<'a> {
    pub data: &'a [u8],
    pub stride: usize,
}