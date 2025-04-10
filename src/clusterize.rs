use crate::ffi;
use crate::{DecodePosition, VertexDataAdapter};

pub type Bounds = ffi::meshopt_Bounds;

#[derive(Copy, Clone)]
pub struct Meshlet<'data> {
    pub vertices: &'data [u32],
    pub triangles: &'data [u8],
}

pub struct Meshlets {
    pub meshlets: Vec<ffi::meshopt_Meshlet>,
    pub vertices: Vec<u32>,
    pub triangles: Vec<u8>,
}

impl Meshlets {
    #[inline]
    pub fn len(&self) -> usize {
        self.meshlets.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.meshlets.is_empty()
    }

    fn meshlet_from_ffi(&self, meshlet: &ffi::meshopt_Meshlet) -> Meshlet<'_> {
        Meshlet {
            vertices: &self.vertices[meshlet.vertex_offset as usize
                ..meshlet.vertex_offset as usize + meshlet.vertex_count as usize],
            triangles: &self.triangles[meshlet.triangle_offset as usize
                ..meshlet.triangle_offset as usize + meshlet.triangle_count as usize * 3],
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Meshlet<'_> {
        self.meshlet_from_ffi(&self.meshlets[idx])
    }

    pub fn iter(&self) -> impl Iterator<Item = Meshlet<'_>> {
        self.meshlets
            .iter()
            .map(|meshlet| self.meshlet_from_ffi(meshlet))
    }
}

/// Splits the mesh into a set of meshlets where each meshlet has a micro index buffer
/// indexing into meshlet vertices that refer to the original vertex buffer.
///
/// The resulting data can be used to render meshes using `NVidia programmable mesh shading`
/// pipeline, or in other cluster-based renderers.
///
/// Note: `max_vertices` must be <= 256 and `max_triangles` must be <= 512 and divisible by 4.
pub fn build_meshlets(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    max_vertices: usize,
    max_triangles: usize,
    cone_weight: f32,
) -> Meshlets {
    let meshlet_count =
        unsafe { ffi::meshopt_buildMeshletsBound(indices.len(), max_vertices, max_triangles) };
    let mut meshlets: Vec<ffi::meshopt_Meshlet> =
        vec![unsafe { ::std::mem::zeroed() }; meshlet_count];

    let mut meshlet_verts: Vec<u32> = vec![0; meshlet_count * max_vertices];
    let mut meshlet_tris: Vec<u8> = vec![0; meshlet_count * max_triangles * 3];

    let count = unsafe {
        ffi::meshopt_buildMeshlets(
            meshlets.as_mut_ptr(),
            meshlet_verts.as_mut_ptr(),
            meshlet_tris.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            max_vertices,
            max_triangles,
            cone_weight,
        )
    };

    let last_meshlet = meshlets[count - 1];
    meshlet_verts
        .truncate(last_meshlet.vertex_offset as usize + last_meshlet.vertex_count as usize);
    meshlet_tris.truncate(
        last_meshlet.triangle_offset as usize
            + ((last_meshlet.triangle_count as usize * 3 + 3) & !3),
    );
    meshlets.truncate(count);

    for meshlet in meshlets.iter_mut().take(count) {
        unsafe {
            ffi::meshopt_optimizeMeshlet(
                &mut meshlet_verts[meshlet.vertex_offset as usize],
                &mut meshlet_tris[meshlet.triangle_offset as usize],
                meshlet.triangle_count as usize,
                meshlet.vertex_count as usize,
            );
        };
    }

    Meshlets {
        meshlets,
        vertices: meshlet_verts,
        triangles: meshlet_tris,
    }
}

/// Experimental: Meshlet builder with flexible cluster sizes.
///
/// Splits the mesh into a set of meshlets, similarly to `build_meshlets`, but allows to specify minimum and maximum number of triangles per meshlet.
/// Clusters between min and max triangle counts are split when the cluster size would have exceeded the expected cluster size by more than `split_factor`.
/// Additionally, allows to switch to axis aligned clusters by setting `cone_weight` to a negative value.
///
/// * `max_vertices`, `min_triangles` and `max_triangles` must not exceed implementation limits (`max_vertices` <= 256, `max_triangles` <= 512; `min_triangles` <= `max_triangles`; both `min_triangles` and `max_triangles` must be divisible by 4)
/// * `cone_weight` should be set to 0 when cone culling is not used, and a value between 0 and 1 otherwise to balance between cluster size and cone culling efficiency; additionally, `cone_weight` can be set to a negative value to prioritize axis aligned clusters (for raytracing) instead
/// * `split_factor` should be set to a non-negative value; when greater than 0, clusters that have large bounds may be split unless they are under the `min_triangles` threshold
pub fn build_meshlets_flex(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    cone_weight: f32,
    split_factor: f32,
) -> Meshlets {
    let meshlet_count =
        unsafe { ffi::meshopt_buildMeshletsBound(indices.len(), max_vertices, max_triangles) };
    let mut meshlets: Vec<ffi::meshopt_Meshlet> =
        vec![unsafe { ::std::mem::zeroed() }; meshlet_count];

    let mut meshlet_verts: Vec<u32> = vec![0; meshlet_count * max_vertices];
    let mut meshlet_tris: Vec<u8> = vec![0; meshlet_count * max_triangles * 3];

    let count = unsafe {
        ffi::meshopt_buildMeshletsFlex(
            meshlets.as_mut_ptr(),
            meshlet_verts.as_mut_ptr(),
            meshlet_tris.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            max_vertices,
            min_triangles,
            max_triangles,
            cone_weight,
            split_factor,
        )
    };

    let last_meshlet = meshlets[count - 1];
    meshlet_verts
        .truncate(last_meshlet.vertex_offset as usize + last_meshlet.vertex_count as usize);
    meshlet_tris.truncate(
        last_meshlet.triangle_offset as usize
            + ((last_meshlet.triangle_count as usize * 3 + 3) & !3),
    );
    meshlets.truncate(count);

    for meshlet in meshlets.iter_mut().take(count) {
        unsafe {
            ffi::meshopt_optimizeMeshlet(
                &mut meshlet_verts[meshlet.vertex_offset as usize],
                &mut meshlet_tris[meshlet.triangle_offset as usize],
                meshlet.triangle_count as usize,
                meshlet.vertex_count as usize,
            );
        };
    }

    Meshlets {
        meshlets,
        vertices: meshlet_verts,
        triangles: meshlet_tris,
    }
}

/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
///
/// For backface culling with orthographic projection, use the following formula to reject backfacing clusters:
///   `dot(view, cone_axis) >= cone_cutoff`
///
/// For perspective projection, use the following formula that needs cone apex in addition to axis & cutoff:
///   `dot(normalize(cone_apex - camera_position), cone_axis) >= cone_cutoff`
///
/// Alternatively, you can use the formula that doesn't need cone apex and uses bounding sphere instead:
///   `dot(normalize(center - camera_position), cone_axis) >= cone_cutoff + radius / length(center - camera_position)`
///
/// or an equivalent formula that doesn't have a singularity at `center = camera_position`:
///   `dot(center - camera_position, cone_axis) >= cone_cutoff * length(center - camera_position) + radius`
///
/// The formula that uses the apex is slightly more accurate but needs the apex; if you are already using bounding sphere
/// to do frustum/occlusion culling, the formula that doesn't use the apex may be preferable.
///
/// `index_count` should be <= 256*3 (the function assumes clusters of limited size)
pub fn compute_cluster_bounds(indices: &[u32], vertices: &VertexDataAdapter<'_>) -> Bounds {
    unsafe {
        ffi::meshopt_computeClusterBounds(
            indices.as_ptr(),
            indices.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        )
    }
}

/// Creates a view into data containing position-like information.
pub struct PositionDataAdapter<'a> {
    /// The data buffer containing position information.
    /// This should be a pointer to a buffer of size at least `position_count * position_stride`.
    pub data: &'a [u8],
    /// The number of positions in the buffer.
    pub position_count: usize,
    /// The size of each element in the buffer. can be as big as you want if indexing in some other struct
    pub position_stride: usize,
    /// The offset in bytes from the start of each element to the position data. it must contain 3xf32 of information.
    pub position_offset: usize,
}

/// Creates a view into data containing radius-like information. radius must be a non-negative f32.
/// You may use the same data for both position and radius, but the stride must be the same.
pub struct RadiusDataAdapter<'a> {
    /// The data buffer containing radius information.
    /// This should be a pointer to a buffer of size at least `radius_count * radius_stride`.
    pub data: &'a [u8],
    /// The number of radii in the buffer.
    pub radius_count: usize,
    /// The size of each element in the buffer. can be as big as you want if indexing in some other struct
    pub radius_stride: usize,
    /// The offset in bytes from the start of each element to the radius data.
    pub radius_offset: usize,
}

pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

/// Sphere bounds generator
/// Creates bounding sphere around a set of points or a set of spheres;
pub fn compute_sphere_bounds(positions: PositionDataAdapter<'_>, radius: Option<RadiusDataAdapter<'_>>) -> Sphere {
    if let Some(ref r) = radius {
        assert_eq!(positions.position_count, r.radius_count);
    }
    assert!(positions.data.len() >= positions.position_count * positions.position_stride);
    unsafe {
        let (radius_ptr, radius_stride) = match radius {
            Some(r) => {
                (r.data.as_ptr().add(r.radius_offset).cast(), r.radius_stride)
            },
            None => (std::ptr::null(), 0),
        };
        let bounds = ffi::meshopt_computeSphereBounds(
            positions.data.as_ptr().add(positions.position_offset).cast(),
            positions.position_count,
            positions.position_stride,
            radius_ptr,
            radius_stride,
        );
        
        Sphere {
            center: [bounds.center[0], bounds.center[1], bounds.center[2]],
            radius: bounds.radius,
        }
    }
}

/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
///
/// For backface culling with orthographic projection, use the following formula to reject backfacing clusters:
///   `dot(view, cone_axis) >= cone_cutoff`
///
/// For perspective projection, use the following formula that needs cone apex in addition to axis & cutoff:
///   `dot(normalize(cone_apex - camera_position), cone_axis) >= cone_cutoff`
///
/// Alternatively, you can use the formula that doesn't need cone apex and uses bounding sphere instead:
///   `dot(normalize(center - camera_position), cone_axis) >= cone_cutoff + radius / length(center - camera_position)`
///
/// or an equivalent formula that doesn't have a singularity at `center = camera_position`:
///   `dot(center - camera_position, cone_axis) >= cone_cutoff * length(center - camera_position) + radius`
///
/// The formula that uses the apex is slightly more accurate but needs the apex; if you are already using bounding sphere
/// to do frustum/occlusion culling, the formula that doesn't use the apex may be preferable.
///
/// `index_count` should be <= 256*3 (the function assumes clusters of limited size)
pub fn compute_cluster_bounds_decoder<T: DecodePosition>(
    indices: &[u32],
    vertices: &[T],
) -> Bounds {
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_computeClusterBounds(
            indices.as_ptr(),
            indices.len(),
            vertices.as_ptr().cast(),
            vertices.len() * 3,
            ::std::mem::size_of::<f32>() * 3,
        )
    }
}

pub fn compute_meshlet_bounds(meshlet: Meshlet<'_>, vertices: &VertexDataAdapter<'_>) -> Bounds {
    unsafe {
        ffi::meshopt_computeMeshletBounds(
            meshlet.vertices.as_ptr(),
            meshlet.triangles.as_ptr(),
            meshlet.triangles.len() / 3,
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
        )
    }
}

pub fn compute_meshlet_bounds_decoder<T: DecodePosition>(
    meshlet: Meshlet<'_>,
    vertices: &[T],
) -> Bounds {
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_computeMeshletBounds(
            meshlet.vertices.as_ptr(),
            meshlet.triangles.as_ptr(),
            meshlet.triangles.len() / 3,
            vertices.as_ptr().cast(),
            vertices.len() * 3,
            std::mem::size_of::<f32>() * 3,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::typed_to_bytes;
    use super::*;
    #[test]
    fn test_cluster_sphere_bounds() {
        let vbr: &[f32] = &[0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 2.0,
        1.0, 0.0, 1.0, 3.0,];

        let bounds = compute_sphere_bounds(
            PositionDataAdapter {
                data: typed_to_bytes(vbr),
                position_count: 4,
                position_stride: 4 * size_of::<f32>(),
                position_offset: 0,
            },
            None,
        );

        assert!(bounds.radius < 0.97);

        let eps: &[f32] = &[1e-3, 2e-3, 3e-3, 4e-3];

        let bounds = compute_sphere_bounds(
            PositionDataAdapter {
                data: typed_to_bytes(vbr),
                position_count: 4,
                position_stride: 4 * size_of::<f32>(),
                position_offset: 0,
            },
            Some(RadiusDataAdapter {
                data: typed_to_bytes(eps),
                radius_count: 4,
                radius_stride: size_of::<f32>(),
                radius_offset: 0,
            }),
        );

        assert!(bounds.radius < 0.87);
        assert!((bounds.center[0] - 0.5).abs() < 1e-2);
        assert!((bounds.center[1] - 0.5).abs() < 1e-2);
        assert!((bounds.center[2] - 0.5).abs() < 1e-2);

        let bounds = compute_sphere_bounds(
            PositionDataAdapter {
                data: typed_to_bytes(vbr),
                position_count: 4,
                position_stride: 4 * size_of::<f32>(),
                position_offset: 0,
            },
            Some(RadiusDataAdapter {
                data: typed_to_bytes(vbr),
                radius_count: 4,
                radius_stride: 4 * size_of::<f32>(),
                radius_offset: 3 * size_of::<f32>(),
            }),
        );

        assert!((bounds.radius - 3.0).abs() < 1e-2);
        assert!((bounds.center[0] - 1.0).abs() < 1e-2);
        assert!((bounds.center[1] - 0.0).abs() < 1e-2);
        assert!((bounds.center[2] - 1.0).abs() < 1e-2);
    }
}