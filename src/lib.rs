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
    clippy::empty_enum,
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
    clippy::match_on_vec_items,
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
    clippy::string_to_string,
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

pub use crate::{
    analyze::*, clusterize::*, encoding::*, error::*, optimize::*, packing::*, remap::*, shadow::*,
    simplify::*, stripify::*, utilities::*,
};
use std::marker::PhantomData;
use std::mem::size_of;

/// Vertex attribute stream, similar to `glVertexPointer`
///
/// Each element takes size bytes, with stride controlling
/// the spacing between successive elements.
#[derive(Debug, Copy, Clone)]
#[repr(C)] // repr(C) matches the ffi struct which should eliminate one copy when converting to ffi::meshopt_Stream
pub struct VertexStream<'a> {
    /// Pointer to buffer which contains vertex data.
    data: *const u8,
    /// Space between vertices inside the buffer (in bytes).
    stride: usize,
    /// The size in bytes of the vertex attribute this Stream is representing.
    size: usize,

    _marker: PhantomData<&'a [u8]>,
}

impl<'a> VertexStream<'a> {
    /// Create a new `VertexStream` from a slice of bytes.
    /// The `stride` parameter controls the spacing between successive elements.
    /// The `size` parameter controls the size in bytes of the vertex attribute this Stream is representing.
    ///
    /// You can also use `new_from_slice` to create a `VertexStream` from a slice of typed data.
    ///
    /// # Errors
    /// Returns an error if the `data` slice is not evenly divisible by `stride` or if `size` is greater than `stride`.
    /// # Example
    /// ```
    /// use meshopt::VertexStream;
    /// let data = vec![0u8; 12];
    /// let stream = VertexStream::new(&data, 12, 12).unwrap();
    /// ```
    ///
    pub fn new(data: &'a [u8], stride: usize, size: usize) -> Result<VertexStream<'a>> {
        let vertex_count = data.len() / stride;
        if data.len() % vertex_count != 0 {
            return Err(Error::memory_dynamic(format!(
                "vertex data length ({}) must be evenly divisible by stride ({})",
                data.len(),
                stride
            )));
        }

        if size > stride {
            return Err(Error::memory_dynamic(format!(
                "size ({}) must be less than or equal to stride ({})",
                size, stride
            )));
        }

        let data = data.as_ptr();

        Ok(Self {
            data,
            stride,
            size,
            _marker: Default::default(),
        })
    }

    /// Create a new `VertexStream` from a slice of bytes without checking the parameters.
    ///
    /// # Safety
    /// The `data` slice must be evenly divisible by `stride` and `size` must be less than or equal to `stride`.
    pub unsafe fn new_unchecked(data: &'a [u8], stride: usize, size: usize) -> VertexStream<'a> {
        let data = data.as_ptr();
        Self {
            data,
            stride,
            size,
            _marker: Default::default(),
        }
    }

    /// Create a new `VertexStream` from a slice of typed vertices.
    /// Its stride and size are calculated based on the size `T` of the slice elements.
    pub fn new_from_slice<T>(data: &[T]) -> VertexStream<'a> {
        let stride = size_of::<T>();
        let size = stride;
        let data = typed_to_bytes(data);
        let data = data.as_ptr();
        Self {
            data,
            stride,
            size,
            _marker: Default::default(),
        }
    }

    // We need getters as the fields cant be public due to safety reasons

    pub fn data(&self) -> *const u8 {
        self.data
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn size(&self) -> usize {
        self.size
    }
}
