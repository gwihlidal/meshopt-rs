use crate::ffi;
use crate::utilities::{check_indices, check_triangles, holds_strided};
use crate::{DecodePosition, VertexDataAdapter};
use std::mem;

pub type Bounds = ffi::meshopt_Bounds;

/// Largest number of vertices a meshlet can have, as local indices are stored as bytes.
const MAX_MESHLET_VERTICES: usize = 256;
/// Largest number of triangles the library supports in a meshlet or cluster.
const MAX_MESHLET_TRIANGLES: usize = 512;

/// Panics unless the meshlet builders support the size limits; the library sizes its buffers
/// from them.
#[track_caller]
pub(crate) fn check_meshlet_limits(
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
) {
    assert!(
        (3..=MAX_MESHLET_VERTICES).contains(&max_vertices),
        "max_vertices ({max_vertices}) must be between 3 and {MAX_MESHLET_VERTICES}"
    );
    assert!(
        (1..=max_triangles).contains(&min_triangles) && max_triangles <= MAX_MESHLET_TRIANGLES,
        "min_triangles ({min_triangles}) and max_triangles ({max_triangles}) must be between 1 and {MAX_MESHLET_TRIANGLES}"
    );
}

/// Panics unless `indices` is a triangle list small enough to be a single cluster.
#[track_caller]
fn check_cluster(indices: &[u32], vertex_count: usize) {
    assert!(
        indices.len() / 3 <= MAX_MESHLET_TRIANGLES,
        "cluster must have at most {MAX_MESHLET_TRIANGLES} triangles"
    );
    check_triangles(indices, vertex_count);
}

/// Panics unless `triangles` is a list of triangles indexing into `vertices`, which in turn refer
/// to `vertex_count` vertices.
#[track_caller]
fn check_meshlet(vertices: &[u32], triangles: &[u8], vertex_count: usize) {
    assert!(
        vertices.len() <= MAX_MESHLET_VERTICES
            && triangles.len().is_multiple_of(3)
            && triangles.len() / 3 <= MAX_MESHLET_TRIANGLES,
        "meshlet must have at most {MAX_MESHLET_VERTICES} vertices and {MAX_MESHLET_TRIANGLES} triangles"
    );
    assert!(
        triangles
            .iter()
            .all(|&index| (index as usize) < vertices.len()),
        "meshlet triangles must refer to the meshlet vertices"
    );
    check_indices(vertices, vertex_count);
}

/// Panics unless the cluster index counts add up to the number of cluster indices.
#[track_caller]
fn check_partition(cluster_indices: &[u32], cluster_index_counts: &[u32], vertex_count: usize) {
    let total: u64 = cluster_index_counts
        .iter()
        .map(|&count| u64::from(count))
        .sum();
    assert!(
        total == cluster_indices.len() as u64 && !cluster_index_counts.contains(&0),
        "cluster_index_counts must be non-zero and add up to the number of cluster indices"
    );
    check_indices(cluster_indices, vertex_count);
}

/// Internal helper to finalize meshlet data by truncating arrays and optimizing meshlets
fn finalize_meshlets(
    mut meshlets: Vec<ffi::meshopt_Meshlet>,
    mut meshlet_verts: Vec<u32>,
    mut meshlet_tris: Vec<u8>,
    count: usize,
) -> Meshlets {
    if count > 0 {
        let last_meshlet = meshlets[count - 1];
        meshlet_verts
            .truncate(last_meshlet.vertex_offset as usize + last_meshlet.vertex_count as usize);
        meshlet_tris.truncate(
            last_meshlet.triangle_offset as usize
                + ((last_meshlet.triangle_count as usize * 3 + 3) & !3),
        );
    }
    meshlets.truncate(count);

    // Optimize each meshlet
    for meshlet in meshlets.iter_mut().take(count) {
        unsafe {
            ffi::meshopt_optimizeMeshlet(
                meshlet_verts[meshlet.vertex_offset as usize..].as_mut_ptr(),
                meshlet_tris[meshlet.triangle_offset as usize..].as_mut_ptr(),
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
    check_meshlet_limits(max_vertices, max_triangles, max_triangles);
    check_triangles(indices, vertices.vertex_count);
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

    finalize_meshlets(meshlets, meshlet_verts, meshlet_tris, count)
}

/// Splits the mesh into a set of meshlets where each meshlet has a micro index buffer
/// indexing into meshlet vertices that refer to the original vertex buffer.
///
/// The resulting data can be used to render meshes using `NVidia programmable mesh shading`
/// pipeline, or in other cluster-based renderers. For maximum efficiency the index buffer being
/// converted has to be optimized for vertex cache first.
///
/// `max_vertices` and `max_triangles` must not exceed implementation limits (`max_vertices` <= 256, `max_triangles` <= 512)
pub fn build_meshlets_scan(
    indices: &[u32],
    vertex_count: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> Meshlets {
    check_meshlet_limits(max_vertices, max_triangles, max_triangles);
    check_triangles(indices, vertex_count);
    let meshlet_count =
        unsafe { ffi::meshopt_buildMeshletsBound(indices.len(), max_vertices, max_triangles) };
    let mut meshlets: Vec<ffi::meshopt_Meshlet> = vec![unsafe { mem::zeroed() }; meshlet_count];

    let mut meshlet_verts: Vec<u32> = vec![0; meshlet_count * max_vertices];
    let mut meshlet_tris: Vec<u8> = vec![0; meshlet_count * max_triangles * 3];

    let count = unsafe {
        ffi::meshopt_buildMeshletsScan(
            meshlets.as_mut_ptr(),
            meshlet_verts.as_mut_ptr(),
            meshlet_tris.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_count,
            max_vertices,
            max_triangles,
        )
    };

    finalize_meshlets(meshlets, meshlet_verts, meshlet_tris, count)
}

/// Meshlet builder with flexible cluster sizes.
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
    check_meshlet_limits(max_vertices, min_triangles, max_triangles);
    check_triangles(indices, vertices.vertex_count);
    let meshlet_count =
        unsafe { ffi::meshopt_buildMeshletsBound(indices.len(), max_vertices, min_triangles) };
    let mut meshlets: Vec<ffi::meshopt_Meshlet> = vec![unsafe { mem::zeroed() }; meshlet_count];

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

    finalize_meshlets(meshlets, meshlet_verts, meshlet_tris, count)
}

/// Spatial meshlet builder with flexible cluster sizes.
///
/// Splits the mesh into a set of meshlets, similarly to `build_meshlets`, but allows to specify minimum and maximum number of triangles per meshlet.
/// Uses a spatial approach that optimizes for SAH (Surface Area Heuristic) quality while supporting flexible cluster sizes.
///
/// * `max_vertices`, `min_triangles` and `max_triangles` must not exceed implementation limits (`max_vertices` <= 256, `max_triangles` <= 512; `min_triangles` <= `max_triangles`; both `min_triangles` and `max_triangles` must be divisible by 4)
/// * `fill_weight` allows to prioritize clusters that are closer to maximum size at some cost to SAH quality; 0.5 is a safe default
pub fn build_meshlets_spatial(
    indices: &[u32],
    vertices: &VertexDataAdapter<'_>,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    fill_weight: f32,
) -> Meshlets {
    check_meshlet_limits(max_vertices, min_triangles, max_triangles);
    check_triangles(indices, vertices.vertex_count);
    let meshlet_count =
        unsafe { ffi::meshopt_buildMeshletsBound(indices.len(), max_vertices, min_triangles) };
    let mut meshlets: Vec<ffi::meshopt_Meshlet> = vec![unsafe { mem::zeroed() }; meshlet_count];

    let mut meshlet_verts: Vec<u32> = vec![0; meshlet_count * max_vertices];
    let mut meshlet_tris: Vec<u8> = vec![0; meshlet_count * max_triangles * 3];

    let count = unsafe {
        ffi::meshopt_buildMeshletsSpatial(
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
            fill_weight,
        )
    };

    finalize_meshlets(meshlets, meshlet_verts, meshlet_tris, count)
}

/// Reorders meshlet vertices and triangles to maximize locality, with higher levels resulting in
/// smaller compressed size at the cost of optimization time.
///
/// At level 0 the result is equivalent to the meshlet optimization done by `build_meshlets`;
/// levels >= 1 may rotate triangle corners to improve compression (which can change provoking
/// vertex and affect OMM data).
///
/// * `level` should be in the range [0, 9] with 9 being the slowest; the sweet spot for compression
///   ratio is around 3
pub fn optimize_meshlet_level(vertices: &mut [u32], triangles: &mut [u8], level: u8) {
    check_meshlet(vertices, triangles, usize::MAX);
    unsafe {
        ffi::meshopt_optimizeMeshletLevel(
            vertices.as_mut_ptr(),
            vertices.len(),
            triangles.as_mut_ptr(),
            triangles.len() / 3,
            i32::from(level),
        );
    }
}

/// Extracts meshlet-local vertex and triangle indices from absolute cluster indices.
///
/// Returns `vertices` and `triangles` such that `vertices[triangles[i]] == indices[i]`, with one
/// element in `vertices` per unique vertex.
///
/// * `indices` should have at most 256 unique vertex indices
/// * `indices.len() / 3` must not exceed implementation limits (<= 512)
pub fn extract_meshlet_indices(indices: &[u32]) -> (Vec<u32>, Vec<u8>) {
    // the number of unique vertices can't exceed the number of indices
    let mut vertices: Vec<u32> = vec![0; indices.len()];
    let mut triangles: Vec<u8> = vec![0; indices.len()];
    let vertex_count = unsafe {
        ffi::meshopt_extractMeshletIndices(
            vertices.as_mut_ptr(),
            triangles.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
        )
    };
    vertices.truncate(vertex_count);
    (vertices, triangles)
}

/// Cluster partitioner
/// Partitions clusters into groups of similar size, prioritizing grouping clusters that share vertices.
///
///  `destination` must contain enough space for the resulting partition data (`cluster_index_counts.len()` elements)
///  `destination[i]` will contain the partition id for cluster i, with the total number of partitions returned by the function.
///  `cluster_indices` should have the vertex indices referenced by each cluster, stored sequentially
///  `cluster_index_counts` should have the number of indices in each cluster; sum of all `cluster_index_counts` must be equal to `cluster_indices.len()`
///  `target_partition_size` is a target size for each partition, in clusters; the resulting partitions may be smaller or larger
///
/// The returned value is the number of partitions created. (`destination` can be sliced to the size of the returned value)
pub fn partition_clusters(
    destination: &mut [u32],
    cluster_indices: &[u32],
    cluster_index_counts: &[u32],
    vertex_count: usize,
    target_partition_size: usize,
) -> usize {
    assert_eq!(destination.len(), cluster_index_counts.len());
    check_partition(cluster_indices, cluster_index_counts, vertex_count);
    let target_partition_size = target_partition_size.max(1);
    unsafe {
        ffi::meshopt_partitionClusters(
            destination.as_mut_ptr(),
            cluster_indices.as_ptr(),
            cluster_indices.len(),
            cluster_index_counts.as_ptr(),
            cluster_index_counts.len(),
            std::ptr::null(),
            vertex_count,
            0,
            target_partition_size,
        )
    }
}

/// Cluster partitioner with vertex positions
/// Partitions clusters into groups of similar size, prioritizing grouping clusters that share vertices
/// and are spatially close based on vertex positions.
///
///  `destination` must contain enough space for the resulting partition data (`cluster_index_counts.len()` elements)
///  `destination[i]` will contain the partition id for cluster i, with the total number of partitions returned by the function.
///  `cluster_indices` should have the vertex indices referenced by each cluster, stored sequentially
///  `cluster_index_counts` should have the number of indices in each cluster; sum of all `cluster_index_counts` must be equal to `cluster_indices.len()`
///  `vertices` should contain vertex position data
///  `target_partition_size` is a target size for each partition, in clusters; the resulting partitions may be smaller or larger
///
/// The returned value is the number of partitions created. (`destination` can be sliced to the size of the returned value)
pub fn partition_clusters_with_positions(
    destination: &mut [u32],
    cluster_indices: &[u32],
    cluster_index_counts: &[u32],
    vertices: &VertexDataAdapter<'_>,
    target_partition_size: usize,
) -> usize {
    assert_eq!(destination.len(), cluster_index_counts.len());
    check_partition(cluster_indices, cluster_index_counts, vertices.vertex_count);
    let target_partition_size = target_partition_size.max(1);
    unsafe {
        ffi::meshopt_partitionClusters(
            destination.as_mut_ptr(),
            cluster_indices.as_ptr(),
            cluster_indices.len(),
            cluster_index_counts.as_ptr(),
            cluster_index_counts.len(),
            vertices.pos_ptr(),
            vertices.vertex_count,
            vertices.vertex_stride,
            target_partition_size,
        )
    }
}

/// Cluster partitioner with vertex positions (decoder version)
/// Partitions clusters into groups of similar size, prioritizing grouping clusters that share vertices
/// and are spatially close based on vertex positions.
///
///  `destination` must contain enough space for the resulting partition data (`cluster_index_counts.len()` elements)
///  `destination[i]` will contain the partition id for cluster i, with the total number of partitions returned by the function.
///  `cluster_indices` should have the vertex indices referenced by each cluster, stored sequentially
///  `cluster_index_counts` should have the number of indices in each cluster; sum of all `cluster_index_counts` must be equal to `cluster_indices.len()`
///  `vertices` should contain vertex data that can decode positions
///  `target_partition_size` is a target size for each partition, in clusters; the resulting partitions may be smaller or larger
///
/// The returned value is the number of partitions created. (`destination` can be sliced to the size of the returned value)
pub fn partition_clusters_with_decoder<T: DecodePosition>(
    destination: &mut [u32],
    cluster_indices: &[u32],
    cluster_index_counts: &[u32],
    vertices: &[T],
    target_partition_size: usize,
) -> usize {
    assert_eq!(destination.len(), cluster_index_counts.len());
    check_partition(cluster_indices, cluster_index_counts, vertices.len());
    let target_partition_size = target_partition_size.max(1);
    let positions = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_partitionClusters(
            destination.as_mut_ptr(),
            cluster_indices.as_ptr(),
            cluster_indices.len(),
            cluster_index_counts.as_ptr(),
            cluster_index_counts.len(),
            positions.as_ptr().cast(),
            positions.len(),
            mem::size_of::<f32>() * 3,
            target_partition_size,
        )
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
    check_cluster(indices, vertices.vertex_count);
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
pub fn compute_sphere_bounds(
    positions: PositionDataAdapter<'_>,
    radius: Option<RadiusDataAdapter<'_>>,
) -> Sphere {
    /// Returns a pointer to `count` elements of `size` bytes in `data`, checking that they are
    /// in bounds and aligned for `f32`.
    #[track_caller]
    fn element_ptr(
        data: &[u8],
        count: usize,
        stride: usize,
        offset: usize,
        size: usize,
    ) -> *const f32 {
        let elements = data.get(offset..).unwrap_or_default();
        assert!(
            stride.is_multiple_of(mem::align_of::<f32>())
                && offset.checked_add(size).is_some_and(|end| end <= stride)
                && holds_strided(elements.len(), count, size, stride),
            "data must hold {count} elements of {size} bytes at offset {offset} and stride {stride}"
        );
        assert!(
            (elements.as_ptr() as usize).is_multiple_of(mem::align_of::<f32>()),
            "data must be aligned to 4 bytes"
        );
        elements.as_ptr().cast()
    }

    let count = positions.position_count;
    if count == 0 {
        return Sphere {
            center: [0.0; 3],
            radius: 0.0,
        };
    }
    let positions_ptr = element_ptr(
        positions.data,
        count,
        positions.position_stride,
        positions.position_offset,
        mem::size_of::<[f32; 3]>(),
    );
    let (radius_ptr, radius_stride) = match radius {
        Some(r) => {
            assert_eq!(count, r.radius_count);
            let ptr = element_ptr(
                r.data,
                count,
                r.radius_stride,
                r.radius_offset,
                mem::size_of::<f32>(),
            );
            (ptr, r.radius_stride)
        }
        None => (std::ptr::null(), 0),
    };
    let bounds = unsafe {
        ffi::meshopt_computeSphereBounds(
            positions_ptr,
            count,
            positions.position_stride,
            radius_ptr,
            radius_stride,
        )
    };

    Sphere {
        center: bounds.center,
        radius: bounds.radius,
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
    check_cluster(indices, vertices.len());
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    unsafe {
        ffi::meshopt_computeClusterBounds(
            indices.as_ptr(),
            indices.len(),
            vertices.as_ptr().cast(),
            vertices.len(),
            mem::size_of::<f32>() * 3,
        )
    }
}

pub fn compute_meshlet_bounds(meshlet: Meshlet<'_>, vertices: &VertexDataAdapter<'_>) -> Bounds {
    check_meshlet(meshlet.vertices, meshlet.triangles, vertices.vertex_count);
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
    check_meshlet(meshlet.vertices, meshlet.triangles, vertices.len());
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
            vertices.len(),
            mem::size_of::<f32>() * 3,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;
    #[test]
    fn test_cluster_sphere_bounds() {
        let vbr: &[f32] = &[
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 2.0, //
            1.0, 0.0, 1.0, 3.0,
        ];

        let bounds = compute_sphere_bounds(
            PositionDataAdapter {
                data: typed_to_bytes(vbr),
                position_count: 4,
                position_stride: 4 * mem::size_of::<f32>(),
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
                position_stride: 4 * mem::size_of::<f32>(),
                position_offset: 0,
            },
            Some(RadiusDataAdapter {
                data: typed_to_bytes(eps),
                radius_count: 4,
                radius_stride: mem::size_of::<f32>(),
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
                position_stride: 4 * mem::size_of::<f32>(),
                position_offset: 0,
            },
            Some(RadiusDataAdapter {
                data: typed_to_bytes(vbr),
                radius_count: 4,
                radius_stride: 4 * mem::size_of::<f32>(),
                radius_offset: 3 * mem::size_of::<f32>(),
            }),
        );

        assert!((bounds.radius - 3.0).abs() < 1e-2);
        assert!((bounds.center[0] - 1.0).abs() < 1e-2);
        assert!((bounds.center[1] - 0.0).abs() < 1e-2);
        assert!((bounds.center[2] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_partition_basic() {
        // 0   1   2
        //     3
        // 4 5 6 7 8
        //     9
        // 10 11  12
        let cluster_indices: &[u32] = &[
            0, 1, 3, 4, 5, 6, //
            1, 2, 3, 6, 7, 8, //
            4, 5, 6, 9, 10, 11, //
            6, 7, 8, 9, 11, 12, //
        ];

        let cluster_index_counts: &[u32] = &[6, 6, 6, 6];
        let mut partitions = vec![0u32; cluster_index_counts.len()];

        assert_eq!(
            partition_clusters(
                &mut partitions,
                cluster_indices,
                cluster_index_counts,
                13,
                1
            ),
            4
        );
        assert_eq!(partitions, [0, 1, 2, 3]);

        assert_eq!(
            partition_clusters(
                &mut partitions,
                cluster_indices,
                cluster_index_counts,
                13,
                2
            ),
            2
        );
        assert_eq!(partitions, [0, 0, 1, 1]);

        assert_eq!(
            partition_clusters(
                &mut partitions,
                cluster_indices,
                cluster_index_counts,
                13,
                4
            ),
            1
        );
        assert_eq!(partitions, [0, 0, 0, 0]);
    }

    #[test]
    fn test_partition_with_positions() {
        // 0   1   2
        //     3
        // 4 5 6 7 8
        //     9
        // 10 11  12
        let cluster_indices: &[u32] = &[
            0, 1, 3, 4, 5, 6, //
            1, 2, 3, 6, 7, 8, //
            4, 5, 6, 9, 10, 11, //
            6, 7, 8, 9, 11, 12, //
        ];

        let cluster_index_counts: &[u32] = &[6, 6, 6, 6];
        let mut partitions = vec![0u32; cluster_index_counts.len()];

        // Vertex positions arranged in a 3x4 grid
        let vertices: &[f32] = &[
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            2.0, 0.0, 0.0, // 2
            1.0, 1.0, 0.0, // 3
            0.0, 2.0, 0.0, // 4
            1.0, 2.0, 0.0, // 5
            2.0, 2.0, 0.0, // 6
            3.0, 2.0, 0.0, // 7
            4.0, 2.0, 0.0, // 8
            2.0, 3.0, 0.0, // 9
            0.0, 4.0, 0.0, // 10
            1.0, 4.0, 0.0, // 11
            2.0, 4.0, 0.0, // 12
        ];

        let vertex_adapter =
            VertexDataAdapter::new(typed_to_bytes(vertices), 3 * mem::size_of::<f32>(), 0).unwrap();

        // Test with positions - should produce better spatial partitioning
        let result = partition_clusters_with_positions(
            &mut partitions,
            cluster_indices,
            cluster_index_counts,
            &vertex_adapter,
            2,
        );

        assert_eq!(result, 2);
        assert_eq!(partitions, [0, 1, 0, 1]);
    }

    #[test]
    fn test_partition_with_decoder() {
        use crate::Vertex;

        // Same test as above but using decoder interface
        let cluster_indices: &[u32] = &[
            0, 1, 3, 4, 5, 6, 1, 2, 3, 6, 7, 8, 4, 5, 6, 9, 10, 11, 6, 7, 8, 9, 11, 12,
        ];

        let cluster_index_counts: &[u32] = &[6, 6, 6, 6];
        let mut partitions = vec![0u32; cluster_index_counts.len()];

        // Create vertices with position data
        let vertices: Vec<Vertex> = vec![
            Vertex {
                p: [0.0, 0.0, 0.0],
                ..Default::default()
            }, // 0
            Vertex {
                p: [1.0, 0.0, 0.0],
                ..Default::default()
            }, // 1
            Vertex {
                p: [2.0, 0.0, 0.0],
                ..Default::default()
            }, // 2
            Vertex {
                p: [1.0, 1.0, 0.0],
                ..Default::default()
            }, // 3
            Vertex {
                p: [0.0, 2.0, 0.0],
                ..Default::default()
            }, // 4
            Vertex {
                p: [1.0, 2.0, 0.0],
                ..Default::default()
            }, // 5
            Vertex {
                p: [2.0, 2.0, 0.0],
                ..Default::default()
            }, // 6
            Vertex {
                p: [3.0, 2.0, 0.0],
                ..Default::default()
            }, // 7
            Vertex {
                p: [4.0, 2.0, 0.0],
                ..Default::default()
            }, // 8
            Vertex {
                p: [2.0, 3.0, 0.0],
                ..Default::default()
            }, // 9
            Vertex {
                p: [0.0, 4.0, 0.0],
                ..Default::default()
            }, // 10
            Vertex {
                p: [1.0, 4.0, 0.0],
                ..Default::default()
            }, // 11
            Vertex {
                p: [2.0, 4.0, 0.0],
                ..Default::default()
            }, // 12
        ];

        let result = partition_clusters_with_decoder(
            &mut partitions,
            cluster_indices,
            cluster_index_counts,
            &vertices,
            2,
        );

        assert_eq!(result, 2);
        assert_eq!(partitions, [0, 1, 0, 1]);
    }

    #[test]
    fn test_meshlets_scan() {
        // Two tetrahedrons
        let ib: &[u32] = &[
            0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2, //
            4, 5, 6, 4, 6, 7, 4, 7, 5, 5, 7, 6,
        ];

        // 4 vertices per meshlet splits the tetrahedrons apart
        let meshlets = build_meshlets_scan(ib, 8, 4, 8);
        assert_eq!(meshlets.len(), 2);

        for (meshlet, expected) in meshlets.iter().zip([[0, 1, 2, 3], [4, 5, 6, 7]]) {
            assert_eq!(meshlet.triangles.len(), 12);

            let mut vertices = meshlet.vertices.to_vec();
            vertices.sort_unstable();
            assert_eq!(vertices, expected);
        }
    }

    #[test]
    fn test_meshlets_flex() {
        // Two tetrahedrons far apart
        let vb: &[f32] = &[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
            10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0,
        ];

        let ib: &[u32] = &[
            0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2, //
            4, 5, 6, 4, 6, 7, 4, 7, 5, 5, 7, 6,
        ];

        let vertices =
            VertexDataAdapter::new(typed_to_bytes(vb), mem::size_of::<[f32; 3]>(), 0).unwrap();

        // Up to 2 meshlets with min_triangles = 4
        assert_eq!(
            unsafe { ffi::meshopt_buildMeshletsBound(ib.len(), 16, 4) },
            2
        );

        let meshlets = build_meshlets_flex(ib, &vertices, 16, 4, 8, 0.0, 0.0);
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets.meshlets[0].triangle_count, 8);
        assert_eq!(meshlets.meshlets[0].vertex_count, 8);

        let meshlets = build_meshlets_flex(ib, &vertices, 16, 4, 8, 0.0, 10.0);
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets.meshlets[0].triangle_count, 8);
        assert_eq!(meshlets.meshlets[0].vertex_count, 8);

        let meshlets = build_meshlets_flex(ib, &vertices, 16, 4, 8, 0.0, 1.0);
        assert_eq!(meshlets.len(), 2);
        assert_eq!(meshlets.meshlets[0].triangle_count, 4);
        assert_eq!(meshlets.meshlets[0].vertex_count, 4);
        assert_eq!(meshlets.meshlets[1].triangle_count, 4);
        assert_eq!(meshlets.meshlets[1].vertex_count, 4);
    }

    #[test]
    fn test_meshlets_spatial() {
        // Two tetrahedrons far apart (copied from vendor test)
        let vb: &[f32] = &[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 11.0, 0.0,
            0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0,
        ];

        let ib: &[u32] = &[
            0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2, 4, 5, 6, 4, 6, 7, 4, 7, 5, 5, 7, 6,
        ];

        let vertices =
            VertexDataAdapter::new(typed_to_bytes(vb), 3 * mem::size_of::<f32>(), 0).unwrap();

        // Up to 2 meshlets with min_triangles=4
        assert_eq!(
            unsafe { ffi::meshopt_buildMeshletsBound(ib.len(), 16, 4) },
            2
        );

        // With strict limits, we should get one meshlet (max_triangles=8) or two (max_triangles=4)
        let meshlets = build_meshlets_spatial(ib, &vertices, 16, 8, 8, 0.0);
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets.meshlets[0].triangle_count, 8);
        assert_eq!(meshlets.meshlets[0].vertex_count, 8);

        let meshlets = build_meshlets_spatial(ib, &vertices, 16, 4, 4, 0.0);
        assert_eq!(meshlets.len(), 2);
        assert_eq!(meshlets.meshlets[0].triangle_count, 4);
        assert_eq!(meshlets.meshlets[0].vertex_count, 4);
        assert_eq!(meshlets.meshlets[1].triangle_count, 4);
        assert_eq!(meshlets.meshlets[1].vertex_count, 4);

        // With max_vertices=4 we should get two meshlets since we can't accommodate both
        let meshlets = build_meshlets_spatial(ib, &vertices, 4, 4, 8, 0.0);
        assert_eq!(meshlets.len(), 2);
        assert_eq!(meshlets.meshlets[0].triangle_count, 4);
        assert_eq!(meshlets.meshlets[0].vertex_count, 4);
        assert_eq!(meshlets.meshlets[1].triangle_count, 4);
        assert_eq!(meshlets.meshlets[1].vertex_count, 4);
    }

    #[test]
    fn test_extract_meshlet_indices() {
        // 15 and 1039 collide in low 10 bits (from meshoptimizer's extractMeshlet test)
        let indices = [0, 7, 15, 1039, 7, 15];

        let (vertices, triangles) = extract_meshlet_indices(&indices);
        assert_eq!(vertices, [0, 7, 15, 1039]);
        assert_eq!(triangles, [0, 1, 2, 3, 1, 2]);

        for (&t, &i) in triangles.iter().zip(&indices) {
            assert_eq!(vertices[t as usize], i);
        }
    }

    #[test]
    fn test_optimize_meshlet_level() {
        // 8x8 quad grid
        const N: u32 = 8;
        let vb: Vec<[f32; 3]> = (0..=N)
            .flat_map(|y| (0..=N).map(move |x| [x as f32, y as f32, 0.0]))
            .collect();
        let ib: Vec<u32> = (0..N)
            .flat_map(|y| {
                (0..N).flat_map(move |x| {
                    let v = y * (N + 1) + x;
                    [v, v + 1, v + N + 1, v + 1, v + N + 2, v + N + 1]
                })
            })
            .collect();

        let vertices =
            VertexDataAdapter::new(typed_to_bytes(&vb), mem::size_of::<[f32; 3]>(), 0).unwrap();
        let meshlets = build_meshlets(&ib, &vertices, 128, 128, 0.0);
        assert_eq!(meshlets.len(), 1);

        // global triangles, rotated so that the smallest index comes first, in sorted order
        let canonical = |vertices: &[u32], triangles: &[u8]| {
            let mut result: Vec<[u32; 3]> = triangles
                .chunks(3)
                .map(|t| {
                    let mut t = [0, 1, 2].map(|k| vertices[t[k] as usize]);
                    let min = (0..3).min_by_key(|&k| t[k]).unwrap();
                    t.rotate_left(min);
                    t
                })
                .collect();
            result.sort_unstable();
            result
        };

        let meshlet = meshlets.get(0);
        let expected = canonical(meshlet.vertices, meshlet.triangles);

        for level in [0, 1, 3, 9] {
            let mut meshlet_vertices = meshlet.vertices.to_vec();
            let mut meshlet_triangles = meshlet.triangles.to_vec();
            optimize_meshlet_level(&mut meshlet_vertices, &mut meshlet_triangles, level);

            assert_eq!(
                canonical(&meshlet_vertices, &meshlet_triangles),
                expected,
                "level {level}"
            );
        }
    }

    #[test]
    fn test_sphere_bounds_empty() {
        // from issue #74: offsets past the end of the data with no elements
        let data = [255u8; 3];
        let sphere = compute_sphere_bounds(
            PositionDataAdapter {
                data: &data,
                position_count: 0,
                position_stride: 65536,
                position_offset: 4,
            },
            Some(RadiusDataAdapter {
                data: &data,
                radius_count: 0,
                radius_stride: 65536,
                radius_offset: 4,
            }),
        );
        assert_eq!(sphere.radius, 0.0);
    }

    #[test]
    #[should_panic(expected = "data must hold 2 elements")]
    fn test_sphere_bounds_short_data() {
        let positions = [0f32; 5];
        compute_sphere_bounds(
            PositionDataAdapter {
                data: typed_to_bytes(&positions),
                position_count: 2,
                position_stride: 12,
                position_offset: 0,
            },
            None,
        );
    }

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_build_meshlets_invalid_index() {
        build_meshlets_scan(&[0, 1, 3], 3, 64, 126);
    }

    #[test]
    #[should_panic(expected = "max_vertices (257)")]
    fn test_build_meshlets_invalid_limits() {
        build_meshlets_scan(&[0, 1, 2], 3, 257, 126);
    }

    #[test]
    #[should_panic(expected = "meshlet triangles must refer to the meshlet vertices")]
    fn test_optimize_meshlet_invalid_triangle() {
        optimize_meshlet_level(&mut [0, 1, 2], &mut [0, 1, 3], 1);
    }

    #[test]
    #[should_panic(expected = "add up to the number of cluster indices")]
    fn test_partition_clusters_invalid_counts() {
        partition_clusters(&mut [0, 0], &[0, 1, 2, 2, 1, 3], &[3, 6], 4, 2);
    }
}
