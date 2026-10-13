use crate::{DecodePosition, VertexDataAdapter, ffi};
use bitflags::bitflags;
use std::mem;

bitflags! {
    /// Experimental: Remeshing flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct RemeshOptions : u32 {
        const None = 0;
        /// Produce a two-sided shell that wraps around surfaces of the original mesh, instead of a
        /// solid mesh.
        const Shell = 1;
        /// Compute optimal output positions that approximate the original surface as closely as
        /// possible.
        const Solve = 2;
    }
}

/// Experimental: Generates a new mesh that approximates or refines the original mesh, attempting
/// to preserve mesh appearance at a given voxel resolution.
///
/// The original topology is not preserved; features that are closer than the voxel size may be
/// merged, and small gaps in the original mesh may be closed.
///
/// Returns the new mesh, with a vertex position for each triangle corner.
///
/// * `resolution` is the dimension of the internal voxel grid and should be in the range [4, 256]
/// * `options` must be a bitmask composed of `RemeshOptions` flags; `RemeshOptions::None` is a safe
///   default
pub fn remesh(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    resolution: u32,
    options: RemeshOptions,
) -> Vec<[f32; 3]> {
    remesh_raw(
        indices,
        vertices.pos_ptr(),
        vertices.vertex_count,
        vertices.vertex_stride,
        resolution,
        options,
    )
}

/// Experimental: Generates a new mesh that approximates or refines the original mesh, attempting
/// to preserve mesh appearance at a given voxel resolution.
///
/// See [`remesh`] for details.
pub fn remesh_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
    resolution: u32,
    options: RemeshOptions,
) -> Vec<[f32; 3]> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    remesh_raw(
        indices,
        positions.as_ptr().cast(),
        positions.len(),
        mem::size_of::<[f32; 3]>(),
        resolution,
        options,
    )
}

fn remesh_raw(
    indices: &[u32],
    vertex_positions: *const f32,
    vertex_count: usize,
    vertex_positions_stride: usize,
    resolution: u32,
    options: RemeshOptions,
) -> Vec<[f32; 3]> {
    // without a destination, this only computes an upper bound on the triangle count
    let max_triangle_count = unsafe {
        ffi::meshopt_remesh(
            std::ptr::null_mut(),
            0,
            indices.as_ptr(),
            indices.len(),
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            resolution as i32,
            options.bits(),
        )
    };

    let mut result: Vec<[f32; 3]> = vec![[0.0; 3]; max_triangle_count * 3];
    let triangle_count = unsafe {
        ffi::meshopt_remesh(
            result.as_mut_ptr().cast(),
            max_triangle_count,
            indices.as_ptr(),
            indices.len(),
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            resolution as i32,
            options.bits(),
        )
    };

    debug_assert!(triangle_count <= max_triangle_count);
    result.truncate(triangle_count * 3);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    // Unit cube with outward-facing triangles
    fn cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        #[rustfmt::skip]
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            3, 6, 2, 3, 7, 6, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        (positions, indices)
    }

    fn bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in points {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        (min, max)
    }

    #[test]
    fn test_remesh_cube() {
        let (positions, indices) = cube();

        for options in [
            RemeshOptions::None,
            RemeshOptions::Solve,
            RemeshOptions::Shell,
            RemeshOptions::Shell | RemeshOptions::Solve,
        ] {
            let result = remesh_decoder(&indices, &positions, 16, options);
            assert!(!result.is_empty(), "{options:?}");
            assert_eq!(result.len() % 3, 0, "{options:?}");

            // output should stay close to the original surface; allow a couple of voxels of slack
            let (min, max) = bounds(&result);
            for k in 0..3 {
                assert!(min[k] > -0.25 && min[k] < 0.25, "{options:?}: min {min:?}");
                assert!(max[k] > 0.75 && max[k] < 1.25, "{options:?}: max {max:?}");
            }
        }
    }

    #[test]
    fn test_remesh_adapter_matches_decoder() {
        let (positions, indices) = cube();

        // interleave positions with padding to exercise the stride
        let vertices: Vec<[f32; 6]> = positions
            .iter()
            .map(|p| [9.0, p[0], p[1], p[2], 9.0, 9.0])
            .collect();
        let adapter = VertexDataAdapter::new(
            typed_to_bytes(&vertices),
            mem::size_of::<[f32; 6]>(),
            mem::size_of::<f32>(),
        )
        .unwrap();

        let a = remesh(&indices, &adapter, 16, RemeshOptions::Solve);
        let b = remesh_decoder(&indices, &positions, 16, RemeshOptions::Solve);
        assert_eq!(a, b);
    }

    #[test]
    fn test_remesh_empty() {
        let positions: Vec<[f32; 3]> = vec![];
        let result = remesh_decoder(&[], &positions, 16, RemeshOptions::None);
        assert!(result.is_empty());
    }
}
