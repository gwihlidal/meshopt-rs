use crate::{DecodePosition, VertexDataAdapter, ffi};
use bitflags::bitflags;
use std::mem;

bitflags! {
    /// Tangent generation options
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct TangentOptions : u32 {
        const None = 0;
        /// Produce tangents compatible with `MikkTSpace` (same weighting and fallbacks) at the cost
        /// of reduced quality. Not recommended unless normal maps are baked.
        const Compatible = 1;
        /// For vertices only connected to degenerate triangles, output zero tangents instead of an
        /// arbitrary fallback.
        const ZeroFallback = 2;
    }
}

/// Experimental: Computes per-corner normal vectors; normals are averaged across soft edges, using
/// crease angle to determine which edges are hard.
///
/// To apply normals to the mesh, either deindex and reindex it with the normal stream, or copy
/// normals to existing vertex data while duplicating vertices with different normal vectors (on
/// normal seams).
/// Input can be indexed or unindexed (`indices` is `None`); this does not affect the resulting
/// normals, but indexed inputs are faster to process.
///
/// Returns one normal per index, or one per vertex if the input is unindexed.
///
/// * `crease_angle` specifies the angle in radians below which normals are averaged across edges
/// * `smoothing` specifies the amount of smoothing to apply to resulting normals as a post-process;
///   larger smoothing values take longer, with the recommended range being [0..5]
pub fn generate_normals(
    indices: Option<&[u32]>,
    vertices: &VertexDataAdapter<'_>,
    crease_angle: f32,
    smoothing: f32,
) -> Vec<[f32; 3]> {
    generate_normals_raw(
        indices,
        vertices.pos_ptr(),
        vertices.vertex_count,
        vertices.vertex_stride,
        crease_angle,
        smoothing,
    )
}

/// Experimental: Computes per-corner normal vectors; normals are averaged across soft edges, using
/// crease angle to determine which edges are hard.
///
/// See [`generate_normals`] for details.
pub fn generate_normals_decoder<T: DecodePosition>(
    indices: Option<&[u32]>,
    vertices: &[T],
    crease_angle: f32,
    smoothing: f32,
) -> Vec<[f32; 3]> {
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    generate_normals_raw(
        indices,
        positions.as_ptr().cast(),
        positions.len(),
        mem::size_of::<[f32; 3]>(),
        crease_angle,
        smoothing,
    )
}

fn generate_normals_raw(
    indices: Option<&[u32]>,
    vertex_positions: *const f32,
    vertex_count: usize,
    vertex_positions_stride: usize,
    crease_angle: f32,
    smoothing: f32,
) -> Vec<[f32; 3]> {
    let index_count = indices.map_or(vertex_count, <[u32]>::len);
    let mut result: Vec<[f32; 3]> = vec![[0.0; 3]; index_count];
    unsafe {
        ffi::meshopt_generateNormals(
            result.as_mut_ptr().cast(),
            indices.map_or(std::ptr::null(), <[u32]>::as_ptr),
            index_count,
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            crease_angle,
            smoothing,
        );
    }
    result
}

/// Computes per-corner tangent vectors; for each corner, computes normalized tangent vector (xyz)
/// and orientation (w, +/-1).
///
/// Bitangent can be reconstructed via `cross(normal, tangent.xyz) * tangent.w`.
/// To apply tangents to the mesh, either deindex and reindex it with the tangent stream, or copy
/// tangents to existing vertex data while duplicating vertices with different tangent vectors (e.g.
/// on UV mirror seams).
/// Input can be indexed or unindexed (`indices` is `None`); this does not affect the resulting
/// tangents, but indexed inputs are faster to process.
///
/// Returns one tangent per index, or one per vertex if the input is unindexed.
///
/// * `normals` should contain unit normals
///
/// Panics if `positions`, `normals` and `uvs` have different lengths.
pub fn generate_tangents(
    indices: Option<&[u32]>,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    options: TangentOptions,
) -> Vec<[f32; 4]> {
    let vertex_count = positions.len();
    assert!(
        normals.len() == vertex_count && uvs.len() == vertex_count,
        "positions ({}), normals ({}) and uvs ({}) must have the same length",
        vertex_count,
        normals.len(),
        uvs.len()
    );

    let index_count = indices.map_or(vertex_count, <[u32]>::len);
    let mut result: Vec<[f32; 4]> = vec![[0.0; 4]; index_count];
    unsafe {
        ffi::meshopt_generateTangents(
            result.as_mut_ptr().cast(),
            indices.map_or(std::ptr::null(), <[u32]>::as_ptr),
            index_count,
            positions.as_ptr().cast(),
            vertex_count,
            mem::size_of::<[f32; 3]>(),
            normals.as_ptr().cast(),
            mem::size_of::<[f32; 3]>(),
            uvs.as_ptr().cast(),
            mem::size_of::<[f32; 2]>(),
            options.bits(),
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    fn assert_close(a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    // Two triangles forming a 90 degree fold along the x axis: one in the xy plane facing +z,
    // one in the xz plane facing -y, sharing vertices 0 and 1
    fn fold() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        let indices = vec![0, 1, 2, 0, 3, 1];
        (positions, indices)
    }

    #[test]
    fn test_generate_normals_flat() {
        let (positions, indices) = fold();

        // crease angle below the fold angle keeps the edge hard
        let normals = generate_normals_decoder(Some(&indices), &positions, 0.5, 0.0);
        assert_eq!(normals.len(), indices.len());
        for n in &normals[0..3] {
            assert_close(n, &[0.0, 0.0, 1.0]);
        }
        for n in &normals[3..6] {
            assert_close(n, &[0.0, -1.0, 0.0]);
        }
    }

    #[test]
    fn test_generate_normals_smooth() {
        let (positions, indices) = fold();

        // crease angle above the fold angle averages normals on the shared edge
        let normals = generate_normals_decoder(Some(&indices), &positions, 2.0, 0.0);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert_close(&normals[0], &[0.0, -h, h]);
        assert_close(&normals[1], &[0.0, -h, h]);
        assert_close(&normals[3], &[0.0, -h, h]);
        assert_close(&normals[5], &[0.0, -h, h]);

        // vertices that are not on the shared edge keep their face normal
        assert_close(&normals[2], &[0.0, 0.0, 1.0]);
        assert_close(&normals[4], &[0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_generate_normals_unindexed() {
        let (positions, indices) = fold();
        let unindexed: Vec<[f32; 3]> = indices.iter().map(|&i| positions[i as usize]).collect();

        let indexed = generate_normals_decoder(Some(&indices), &positions, 2.0, 0.0);
        let normals = generate_normals_decoder(None, &unindexed, 2.0, 0.0);
        assert_eq!(normals.len(), unindexed.len());
        for (a, b) in normals.iter().zip(&indexed) {
            assert_close(a, b);
        }
    }

    #[test]
    fn test_generate_normals_adapter() {
        let (positions, indices) = fold();
        let vertices: Vec<[f32; 5]> = positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 7.0, 7.0])
            .collect();
        let adapter =
            VertexDataAdapter::new(typed_to_bytes(&vertices), mem::size_of::<[f32; 5]>(), 0)
                .unwrap();

        let a = generate_normals(Some(&indices), &adapter, 0.5, 1.0);
        let b = generate_normals_decoder(Some(&indices), &positions, 0.5, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_generate_tangents() {
        // quad in the xy plane with uvs aligned to xy
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let tangents = generate_tangents(
            Some(&indices),
            &positions,
            &normals,
            &uvs,
            TangentOptions::None,
        );
        assert_eq!(tangents.len(), indices.len());
        for t in &tangents {
            assert_close(t, &[1.0, 0.0, 0.0, 1.0]);
        }

        // mirroring u flips the tangent and the orientation
        let mirrored: Vec<[f32; 2]> = uvs.iter().map(|uv| [1.0 - uv[0], uv[1]]).collect();
        let tangents = generate_tangents(
            Some(&indices),
            &positions,
            &normals,
            &mirrored,
            TangentOptions::Compatible,
        );
        for t in &tangents {
            assert_close(t, &[-1.0, 0.0, 0.0, -1.0]);
        }
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_generate_tangents_mismatched_streams() {
        let positions = vec![[0.0; 3]; 3];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let uvs = vec![[0.0; 2]; 2];
        generate_tangents(None, &positions, &normals, &uvs, TangentOptions::None);
    }
}
