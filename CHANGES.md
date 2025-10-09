# Changes

## 0.6.2 (2025-10-09)

* Expose generate_position_remap(_decoder)
* Fix memory safety issue with VertexDataAdapter

## 0.6.1 (2025-09-23)

* Added `SimplifyOptions::Permissive` flag for mesh simplification

## 0.6.0 (2025-09-17)

* Upgraded meshoptimizer library to 0.25 (hash 6daea4695c48338363b08022d2fb15deaef6ac09)
* Added `SimplifyOptions::Prune` and `SimplifyOptions::Regularize` flags for mesh simplification
* Added `simplify_sloppy_with_locks` and `simplify_sloppy_with_locks_decoder` API functions
* Added `build_meshlets_spatial` API function for raytracing-optimized meshlet generation
* Added `partition_clusters_with_positions` and `partition_clusters_with_decoder` API functions for position-aware cluster partitioning
* Fix internal memory safety issues with `build_meshlets_flex`

## 0.5.0 (2025-07-01)

* Upgraded meshoptimizer library to 0.23 (hash 3e9d1ff3135794f519f3237515277c8d9a3fd3f2)
* Added `build_meshlets_flex` API function
* Added `dequantize_half` utility

## 0.4.1 (2024-12-22)

* Truncate vertices and triangles when building meshlets

## 0.4.0 (2024-10-25)

* Upgraded meshoptimizer library to 0.22 (hash 4affad044571506a5724c9a6f15424f43e86f731)
* Added `simplify_with_attributes_and_locks` API function

## 0.3.0 (2024-06-26)

* Upgraded meshoptimizer library to 0.21 (hash 47aafa533b439a78b53cd2854c177db61be7e666)
* Added `SimplifyOptions::Sparse` and `SimplifyOptions::ErrorAbsolute` options
* Improved `build_meshlets` to automatically optimize meshlet triangle order for HW efficiency

## 0.2.1 (2024-04-03)

* Updated dependencies
* Added `simplify_scale` and `simplify_scale_decoder` API functions
* Added `simplify_with_locks` and `simplify_with_locks_decoder` API functions

## 0.2.0 (2024-01-23)

* Updated dependencies.
* **Breaking change**: New parameters to simplify API.

## 0.1.9 (2019-11-02)

* Updated dependencies.
* Added `dyn` to `Fail::cause()` to fix warning.
* Added missing `allocator.cpp` to source_files in `build.rs` and in `Cargo.toml` package include list.
* Made the crate buildable on WebAssembly.
* Fixed build under toolchain 'windows-gnu'.
* Updated vendoring of meshoptimizer to commit hash `7cf4a53ece15fa7526410a6d4cae059bd5593178`.

## 0.1.8 (2019-07-14)

* Updated vendoring of meshoptimizer to commit hash `212a35ea9d32ea5e0223105566b3b7deeb06071f`.
* Updated dependencies.
* Updated demo stripify code for restart index.

## 0.1.7 (2019-05-19)

* Implemented `VertexDataAdapter` and modified a number of methods to remove a heavy allocation and slow decode.
  `DecodePosition` is supported through new `*_decoder` methods.
* Updated vendoring of meshoptimizer to commit hash `7bf6e425fa158794c3da75684e8f8c7040b97cfa`.

## 0.1.6 (2019-03-29)

* Fixed usage of VertexStream and adjust data representation.
* Upgraded meshoptimizer library to 0.11.0.
* Upgraded crate dependencies.
* Added `simplify_sloppy` wrapper

## 0.1.5 (2019-01-14)

* Fixed demo example.

## 0.1.4 (2019-01-12)

* Upgraded meshoptimizer library to 0.10.0.
* Upgraded crate dependencies.
* Added proper error handling and removed asserts/unwraps.
* Derived and implemented debug in generated bindings (where possible).
* Implemented mesh encoder command line tool (matches format for meshoptimizer's wasm viewer/loader).
* Implemented support for multiple vertex attribute streams.
* Implemented generate_shadow_indices_multi
* Implemented generate_vertex_remap_multi
* Passed in vertex count to remap_vertex_buffer (needed for correctly resizing result).
* Added more documentation (and some fixes)

## 0.1.3 (2018-12-07)

* Rust 2018 Edition.

## 0.1.2 (2018-12-04)

* Upgraded meshoptimizer library.
* Added support for generating shadow indices.
* Added support for meshlet generation.

## 0.1.1 (2018-10-19)

* Support remapping meshes with a pre-existing index buffer, instead of purely unindexed data.

## 0.1.0 (2018-10-19)

* First release.
