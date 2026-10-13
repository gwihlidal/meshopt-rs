// BEGIN - Embark standard lints v5 for Rust 1.55+
// do not change or add/remove here, but one can add exceptions after this section
// for more info see: <https://github.com/EmbarkStudios/rust-ecosystem/issues/59>
#![deny(unsafe_code)]
#![warn(
    clippy::all,
    clippy::await_holding_lock,
    clippy::char_lit_as_u8,
    clippy::checked_conversions,
    clippy::dbg_macro,
    clippy::debug_assert_with_mut_call,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::doc_markdown,
    clippy::empty_enums,
    clippy::enum_glob_use,
    clippy::exit,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_deref_methods,
    clippy::explicit_into_iter_loop,
    clippy::fallible_impl_from,
    clippy::filter_map_next,
    clippy::flat_map_option,
    clippy::float_cmp_const,
    clippy::fn_params_excessive_bools,
    clippy::from_iter_instead_of_collect,
    clippy::if_let_mutex,
    clippy::implicit_clone,
    clippy::imprecise_flops,
    clippy::inefficient_to_string,
    clippy::invalid_upcast_comparisons,
    clippy::large_digit_groups,
    clippy::large_stack_arrays,
    clippy::large_types_passed_by_value,
    clippy::let_unit_value,
    clippy::linkedlist,
    clippy::lossy_float_literal,
    clippy::macro_use_imports,
    clippy::manual_ok_or,
    clippy::map_err_ignore,
    clippy::map_flatten,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::match_wild_err_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::mem_forget,
    clippy::missing_enforced_import_renames,
    clippy::mut_mut,
    clippy::mutex_integer,
    clippy::needless_borrow,
    clippy::needless_continue,
    clippy::needless_for_each,
    clippy::option_option,
    clippy::path_buf_push_overwrite,
    clippy::ptr_as_ptr,
    clippy::rc_mutex,
    clippy::ref_option_ref,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::same_functions_in_if_condition,
    clippy::semicolon_if_nothing_returned,
    clippy::single_match_else,
    clippy::string_add_assign,
    clippy::string_add,
    clippy::string_lit_as_bytes,
    clippy::todo,
    clippy::trait_duplication_in_bounds,
    clippy::unimplemented,
    clippy::unnested_or_patterns,
    clippy::unused_self,
    clippy::useless_transmute,
    clippy::verbose_file_reads,
    clippy::zero_sized_map_values,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms
)]
// END - Embark standard lints v0.5 for Rust 1.55+
// crate-specific exceptions:
// This crate is doing a lot of FFI and byte munging
#![allow(unsafe_code)]

pub mod analyze;
pub mod clusterize;
#[cfg(not(target_arch = "wasm32"))]
pub mod clusterlod;
pub mod encoding;
pub mod error;
pub mod ffi;
pub mod opacity;
pub mod optimize;
pub mod packing;
pub mod remap;
pub mod remesh;
pub mod shadow;
pub mod simplify;
pub mod spatial;
pub mod stripify;
pub mod tangents;
pub mod utilities;

#[cfg(not(target_arch = "wasm32"))]
pub use crate::clusterlod::*;
pub use crate::{
    analyze::*, clusterize::*, encoding::*, error::*, opacity::*, optimize::*, packing::*,
    remap::*, remesh::*, shadow::*, simplify::*, spatial::*, stripify::*, tangents::*,
    utilities::*,
};

pub use bytemuck;

/// Vertex attribute stream, similar to `glVertexPointer`
///
/// Each element takes `size` bytes, with `stride` controlling
/// the spacing between successive elements.
#[derive(Debug, Copy, Clone)]
pub struct VertexStream<'a> {
    /// Buffer which contains vertex data, starting at the first element.
    data: &'a [u8],
    /// Space between vertices inside the buffer (in bytes).
    stride: usize,
    /// The size in bytes of the vertex attribute this Stream is representing.
    size: usize,
}

impl<'a> VertexStream<'a> {
    /// Creates a `VertexStream` over a slice consisting only of elements of type `T`.
    ///
    /// Panics if `T` is zero-sized or larger than 256 bytes.
    pub fn new<T: bytemuck::NoUninit>(data: &'a [T]) -> VertexStream<'a> {
        let size = std::mem::size_of::<T>();
        Self::new_with_stride(bytemuck::cast_slice(data), size, size)
            .expect("element size must be between 1 and 256 bytes")
    }

    /// Creates a `VertexStream` for an attribute stored inside interleaved vertex data.
    ///
    /// `data` starts at the first element; each element takes `size` bytes and the next one
    /// starts `stride` bytes after it. To select an attribute that isn't at the start of the
    /// vertex, pass the vertex data starting at the attribute's offset.
    ///
    /// Fails if `size` is zero, larger than 256 or larger than `stride`.
    pub fn new_with_stride(data: &'a [u8], size: usize, stride: usize) -> Result<VertexStream<'a>> {
        if size == 0 || size > utilities::MAX_VERTEX_SIZE || size > stride {
            return Err(Error::memory_dynamic(format!(
                "stream element size ({size}) must be non-zero, at most {} and no larger than stride ({stride})",
                utilities::MAX_VERTEX_SIZE
            )));
        }
        Ok(VertexStream { data, stride, size })
    }

    /// Number of complete elements in the stream.
    pub fn len(&self) -> usize {
        if self.data.len() < self.size {
            0
        } else {
            (self.data.len() - self.size) / self.stride + 1
        }
    }

    /// Returns `true` if the stream has no complete elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Space between elements inside the buffer (in bytes).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The size in bytes of the vertex attribute this stream is representing.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Converts streams for the library, checking that each one holds `vertex_count` elements.
#[track_caller]
pub(crate) fn ffi_streams(
    streams: &[VertexStream<'_>],
    vertex_count: usize,
) -> Vec<ffi::meshopt_Stream> {
    assert!(
        !streams.is_empty() && streams.len() <= 16,
        "between 1 and 16 streams are supported"
    );
    streams
        .iter()
        .map(|stream| {
            assert!(
                stream.len() >= vertex_count,
                "stream has {} elements, but vertex count is {vertex_count}",
                stream.len()
            );
            ffi::meshopt_Stream {
                data: stream.data.as_ptr().cast(),
                size: stream.size,
                stride: stream.stride,
            }
        })
        .collect()
}
