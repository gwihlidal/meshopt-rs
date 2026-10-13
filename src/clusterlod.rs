use crate::clusterize::check_meshlet_limits;
use crate::utilities::{check_triangles, holds_strided};
use crate::{SimplifyVertex, VertexDataAdapter, ffi};
use std::os::raw::{c_int, c_void};
use std::{mem, slice};

pub type ClodConfig = ffi::clodConfig;
pub type ClodBounds = ffi::clodBounds;
pub type ClodGroup = ffi::clodGroup;
pub type ClodNode = ffi::clodNode;

/// Input mesh for [`clod_build`].
pub struct ClodMesh<'a> {
    /// Input triangle indices.
    pub indices: &'a [u32],
    /// Input vertex positions.
    pub vertices: &'a VertexDataAdapter<'a>,
    /// Input vertex attributes; used for attribute-aware simplification and permissive simplification.
    pub vertex_attributes: &'a [f32],
    pub vertex_attributes_stride: usize,
    /// Input vertex locks; allows to preserve additional seams (when not using `attribute_protect_mask`)
    /// or lock vertices via `SimplifyVertex` flags.
    pub vertex_lock: Option<&'a [SimplifyVertex]>,
    /// Attribute weights for attribute-aware simplification; maps to `simplify_with_attributes_and_locks` parameters.
    pub attribute_weights: &'a [f32],
    /// Attribute mask to flag attribute discontinuities for permissive simplification; mask (1<<K) corresponds to attribute K.
    pub attribute_protect_mask: u32,
}

impl<'a> ClodMesh<'a> {
    /// Creates a mesh with positions only; attributes, locks and the protect mask can be set afterwards.
    pub fn new(indices: &'a [u32], vertices: &'a VertexDataAdapter<'a>) -> Self {
        ClodMesh {
            indices,
            vertices,
            vertex_attributes: &[],
            vertex_attributes_stride: 0,
            vertex_lock: None,
            attribute_weights: &[],
            attribute_protect_mask: 0,
        }
    }
}

/// Cluster produced by [`clod_build`].
#[derive(Debug, Copy, Clone)]
pub struct ClodCluster<'a> {
    /// Index of more refined group (with more triangles) that produced this cluster during simplification,
    /// or -1 for original geometry.
    pub refined: i32,
    /// Cluster bounds; should only be used for culling, as `bounds.error` is not monotonic across DAG.
    pub bounds: ClodBounds,
    /// Cluster indices; refer to the original mesh vertex buffer.
    pub indices: &'a [u32],
    /// Cluster vertex count; `indices` has `vertex_count` unique entries.
    pub vertex_count: usize,
}

/// Default configuration optimized for rasterization.
pub fn clod_default_config(max_triangles: usize) -> ClodConfig {
    unsafe { ffi::clodDefaultConfig(max_triangles) }
}

/// Default configuration optimized for raytracing.
pub fn clod_default_config_rt(max_triangles: usize) -> ClodConfig {
    unsafe { ffi::clodDefaultConfigRT(max_triangles) }
}

/// Builds cluster LOD hierarchy, calling `output` as new clusters and groups are generated.
///
/// `output` gets called for each group in sequence; the returned value gets saved for clusters
/// emitted from this group (`ClodCluster::refined`).
///
/// Panics if `config.simplify_dilate_borders` is set, as border dilation mutates vertex positions in place,
/// if the meshlet limits in `config` are not ones `build_meshlets_flex` supports or `config.partition_size`
/// is 0, if `mesh.indices` is not a triangle list that refers to `mesh.vertices`, if `mesh.vertex_attributes`
/// or `mesh.vertex_lock` is too small for the number of vertices, or if `mesh.vertex_attributes_stride`
/// is not a multiple of 4 with room for `mesh.attribute_weights` and `mesh.attribute_protect_mask`.
pub fn clod_build<F>(config: ClodConfig, mesh: &ClodMesh<'_>, output: F)
where
    F: FnMut(ClodGroup, &[ClodCluster<'_>]) -> i32,
{
    assert!(
        !config.simplify_dilate_borders,
        "simplify_dilate_borders is not supported, as vertex positions are immutable"
    );
    check_meshlet_limits(
        config.max_vertices,
        config.min_triangles,
        config.max_triangles,
    );
    assert!(config.partition_size > 0, "partition_size must not be 0");

    let vertex_count = mesh.vertices.vertex_count;
    check_triangles(mesh.indices, vertex_count);

    // the library reads the attributes that have a weight or a bit in the protect mask, which
    // has one bit per float of the stride
    let stride = mesh.vertex_attributes_stride;
    let stride_floats = stride / mem::size_of::<f32>();
    let used_floats = mesh
        .attribute_weights
        .len()
        .max((u32::BITS - mesh.attribute_protect_mask.leading_zeros()) as usize);
    assert!(
        stride.is_multiple_of(mem::size_of::<f32>())
            && stride_floats < u32::BITS as usize
            && used_floats <= stride_floats,
        "vertex_attributes_stride ({stride}) must be a multiple of 4 smaller than 128 with room for {used_floats} attributes"
    );
    assert!(
        used_floats == 0
            || holds_strided(
                mem::size_of_val(mesh.vertex_attributes),
                vertex_count,
                used_floats * mem::size_of::<f32>(),
                stride
            ),
        "vertex_attributes ({}) must have room for {vertex_count} vertices at stride {stride}",
        mesh.vertex_attributes.len()
    );
    assert!(
        mesh.vertex_lock
            .is_none_or(|lock| lock.len() >= vertex_count),
        "vertex_lock must have a value for each of the {vertex_count} vertices"
    );

    // nothing to output, and the library expects to have vertices to work with
    if mesh.indices.is_empty() {
        return;
    }

    let ffi_mesh = ffi::clodMesh {
        indices: mesh.indices.as_ptr(),
        index_count: mesh.indices.len(),
        vertex_count,
        vertex_positions: mesh.vertices.pos_ptr(),
        vertex_positions_stride: mesh.vertices.vertex_stride,
        vertex_attributes: mesh.vertex_attributes.as_ptr(),
        vertex_attributes_stride: stride,
        vertex_lock: mesh
            .vertex_lock
            .map_or(std::ptr::null(), |lock| lock.as_ptr().cast()),
        attribute_weights: mesh.attribute_weights.as_ptr(),
        attribute_count: mesh.attribute_weights.len(),
        attribute_protect_mask: mesh.attribute_protect_mask,
    };

    let mut output = output;
    unsafe {
        ffi::clodBuild(
            config,
            ffi_mesh,
            (&raw mut output).cast(),
            Some(output_callback::<F>),
        );
    }
}

unsafe extern "C" fn output_callback<F>(
    output_context: *mut c_void,
    group: ClodGroup,
    clusters: *const ffi::clodCluster,
    cluster_count: usize,
) -> c_int
where
    F: FnMut(ClodGroup, &[ClodCluster<'_>]) -> i32,
{
    let output = unsafe { &mut *output_context.cast::<F>() };
    let clusters = unsafe { raw_slice(clusters, cluster_count) }
        .iter()
        .map(|cluster| ClodCluster {
            refined: cluster.refined,
            bounds: cluster.bounds,
            indices: unsafe { raw_slice(cluster.indices, cluster.index_count) },
            vertex_count: cluster.vertex_count,
        })
        .collect::<Vec<_>>();

    output(group, &clusters)
}

/// Like `slice::from_raw_parts`, but empty slices can come with any pointer.
unsafe fn raw_slice<'a, T>(data: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(data, len) }
    }
}

/// Extracts meshlet-local indices from cluster indices produced by [`clod_build`].
///
/// Returns `vertices` and `triangles` such that `vertices[triangles[i]] == indices[i]`; the number of
/// unique vertices will be equal to `ClodCluster::vertex_count`.
pub fn clod_local_indices(indices: &[u32]) -> (Vec<u32>, Vec<u8>) {
    let mut vertices: Vec<u32> = vec![0; indices.len()];
    let mut triangles: Vec<u8> = vec![0; indices.len()];
    let vertex_count = unsafe {
        ffi::clodLocalIndices(
            vertices.as_mut_ptr(),
            triangles.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
        )
    };
    vertices.truncate(vertex_count);
    (vertices, triangles)
}

/// Builds a spatial hierarchy over the groups produced by [`clod_build`] for accelerating cluster visibility.
///
/// `level_count` is the number of levels in the group DAG (max(`ClodGroup::depth`) + 1).
///
/// The result is a forest with one tree per DAG level:
/// - nodes `0..level_count` are the roots of per-level trees, each covering groups with `ClodGroup::depth` == level
/// - remaining nodes are the tree nodes; each internal node has up to `node_width` children, and each leaf node refers to a single group
///
/// `node_width` is at least 2. Panics if one of the levels has no groups.
pub fn clod_build_hierarchy(
    groups: &[ClodGroup],
    node_width: usize,
    level_count: usize,
) -> Vec<ClodNode> {
    let node_width = node_width.max(2);
    // each level has a root, which the library takes from the groups of that level
    let mut levels = vec![false; level_count];
    for group in groups {
        if let Some(level) = usize::try_from(group.depth)
            .ok()
            .and_then(|depth| levels.get_mut(depth))
        {
            *level = true;
        }
    }
    if let Some(level) = levels.iter().position(|&used| !used) {
        panic!("level {level} must have at least one group");
    }

    let node_count = unsafe { ffi::clodBuildHierarchyBound(groups.len(), node_width, level_count) };
    let mut nodes: Vec<ClodNode> = vec![unsafe { mem::zeroed() }; node_count];
    let count = unsafe {
        ffi::clodBuildHierarchy(
            nodes.as_mut_ptr(),
            groups.as_ptr(),
            groups.len(),
            node_width,
            level_count,
        )
    };
    nodes.truncate(count);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_to_bytes;

    // Grid of size x size quads with a gentle bump so that simplification has some error to accumulate
    fn grid(size: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        for y in 0..=size {
            for x in 0..=size {
                let (fx, fy) = (x as f32 / size as f32, y as f32 / size as f32);
                positions.push([fx, fy, (fx * 3.0).sin() * (fy * 3.0).cos() * 0.1]);
            }
        }

        let row = (size + 1) as u32;
        let mut indices = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let i = y as u32 * row + x as u32;
                indices.extend_from_slice(&[i, i + 1, i + row + 1, i, i + row + 1, i + row]);
            }
        }

        (positions, indices)
    }

    fn adapter(positions: &[[f32; 3]]) -> VertexDataAdapter<'_> {
        VertexDataAdapter::new(typed_to_bytes(positions), mem::size_of::<[f32; 3]>(), 0).unwrap()
    }

    struct Output {
        groups: Vec<ClodGroup>,
        // per cluster: (group, refined, indices, vertex_count)
        clusters: Vec<(usize, i32, Vec<u32>, usize)>,
    }

    fn build(config: ClodConfig, mesh: &ClodMesh<'_>) -> Output {
        let mut output = Output {
            groups: Vec::new(),
            clusters: Vec::new(),
        };

        clod_build(config, mesh, |group, clusters| {
            let id = output.groups.len();
            output.groups.push(group);
            for cluster in clusters {
                output.clusters.push((
                    id,
                    cluster.refined,
                    cluster.indices.to_vec(),
                    cluster.vertex_count,
                ));
            }
            id as i32
        });

        output
    }

    #[test]
    fn test_clod_build() {
        let (positions, indices) = grid(64);
        let vertices = adapter(&positions);
        let config = clod_default_config(64);

        let output = build(config, &ClodMesh::new(&indices, &vertices));
        let levels = output.groups.last().unwrap().depth as usize + 1;
        assert!(levels > 1);

        // groups are generated in order of increasing depth, and the last group is terminal
        assert!(output.groups.windows(2).all(|w| w[0].depth <= w[1].depth));
        assert_eq!(output.groups.last().unwrap().simplified.error, f32::MAX);

        let mut original_triangles = 0;
        for (group, refined, cluster_indices, vertex_count) in &output.clusters {
            assert_eq!(cluster_indices.len() % 3, 0);
            assert!(cluster_indices.len() / 3 <= config.max_triangles);
            assert!(*vertex_count <= config.max_vertices);

            let mut unique = cluster_indices.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), *vertex_count);

            if *refined == -1 {
                original_triangles += cluster_indices.len() / 3;
            } else {
                // clusters refer to groups that were output earlier
                assert!((*refined as usize) < *group);
            }
        }
        assert_eq!(original_triangles, indices.len() / 3);

        let nodes = clod_build_hierarchy(&output.groups, 8, levels);
        let bound = unsafe { ffi::clodBuildHierarchyBound(output.groups.len(), 8, levels) };
        assert!(nodes.len() > levels && nodes.len() <= bound);

        // every group is referenced by exactly one leaf
        let mut leaves = vec![0; output.groups.len()];
        for node in &nodes {
            if node.group >= 0 {
                leaves[node.group as usize] += 1;
            } else {
                assert!(node.child_count as usize <= 8);
            }
        }
        assert!(leaves.iter().all(|&count| count == 1));
    }

    #[test]
    fn test_clod_build_attributes_and_locks() {
        let (positions, indices) = grid(32);
        let vertices = adapter(&positions);
        let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[1]]).collect();

        // locking all vertices prevents simplification, so every group at depth 0 is terminal
        let locks = vec![SimplifyVertex::Lock; positions.len()];
        let mesh = ClodMesh {
            vertex_attributes: uvs.as_flattened(),
            vertex_attributes_stride: mem::size_of::<[f32; 2]>(),
            vertex_lock: Some(&locks),
            attribute_weights: &[0.5, 0.5],
            attribute_protect_mask: 0b11,
            ..ClodMesh::new(&indices, &vertices)
        };

        let output = build(clod_default_config(64), &mesh);
        assert!(output.groups.iter().all(|g| g.depth == 0));
        assert!(output.groups.iter().all(|g| g.simplified.error == f32::MAX));
        assert!(output.clusters.iter().all(|c| c.1 == -1));
    }

    #[test]
    fn test_clod_local_indices() {
        let (positions, indices) = grid(16);
        let vertices = adapter(&positions);
        let output = build(clod_default_config(64), &ClodMesh::new(&indices, &vertices));

        for (_, _, cluster_indices, vertex_count) in &output.clusters {
            let (local_vertices, triangles) = clod_local_indices(cluster_indices);
            assert_eq!(local_vertices.len(), *vertex_count);
            assert_eq!(triangles.len(), cluster_indices.len());
            for (i, &index) in cluster_indices.iter().enumerate() {
                assert_eq!(local_vertices[triangles[i] as usize], index);
            }
        }
    }

    #[test]
    #[should_panic(expected = "simplify_dilate_borders")]
    fn test_clod_build_dilate_borders() {
        let (positions, indices) = grid(4);
        let vertices = adapter(&positions);
        let config = ClodConfig {
            simplify_dilate_borders: true,
            ..clod_default_config(64)
        };
        clod_build(config, &ClodMesh::new(&indices, &vertices), |_, _| -1);
    }

    // Runs `f`, which must panic with a message that contains `expected`

    #[test]
    #[should_panic(expected = "must be less than vertex count")]
    fn test_clod_build_invalid_index() {
        let (positions, mut indices) = grid(4);
        indices[7] = positions.len() as u32;
        let vertices = adapter(&positions);
        build(clod_default_config(64), &ClodMesh::new(&indices, &vertices));
    }

    #[test]
    #[should_panic(expected = "vertex_attributes (49) must have room for 25 vertices")]
    fn test_clod_build_short_attributes() {
        let (positions, indices) = grid(4);
        let vertices = adapter(&positions);
        let mesh = ClodMesh {
            vertex_attributes: &[0.0; 49],
            vertex_attributes_stride: 8,
            attribute_weights: &[0.5, 0.5],
            ..ClodMesh::new(&indices, &vertices)
        };
        build(clod_default_config(64), &mesh);
    }

    #[test]
    fn test_clod_build_empty() {
        let (positions, _) = grid(4);
        let vertices = adapter(&positions);
        let output = build(clod_default_config(64), &ClodMesh::new(&[], &vertices));
        assert!(output.groups.is_empty());
    }

    #[test]
    #[should_panic(expected = "level 1 must have at least one group")]
    fn test_clod_build_hierarchy_empty_level() {
        let group = ClodGroup {
            depth: 0,
            simplified: ClodBounds {
                center: [0.0; 3],
                radius: 1.0,
                error: 0.5,
            },
        };
        clod_build_hierarchy(&[group], 8, 2);
    }
}
