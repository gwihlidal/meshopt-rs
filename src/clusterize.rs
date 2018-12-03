use ffi;
use DecodePosition;

pub type Bounds = ffi::meshopt_Bounds;
pub type Meshlet = ffi::meshopt_Meshlet;

pub fn build_meshlets(indices: &[u32], vertex_count: usize, max_vertices: usize, max_triangles: usize) -> Vec<Meshlet> {
    let meshlet_count = unsafe {
        ffi::meshopt_buildMeshletsBound(indices.len(), max_vertices, max_triangles)
    };
    let mut meshlets: Vec<Meshlet> = vec![unsafe { ::std::mem::zeroed() }; meshlet_count];
    let count = unsafe {
        ffi::meshopt_buildMeshlets(meshlets.as_mut_ptr(), indices.as_ptr(), indices.len(), vertex_count, max_vertices, max_triangles)
    };
    meshlets.resize(count, unsafe { ::std::mem::zeroed() });
    meshlets
}

pub fn compute_cluster_bounds<T: DecodePosition>(indices: &[u32], vertices: &[T]) -> Bounds {
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let positions = vertices.as_ptr() as *const f32;
    let ffi_bounds = unsafe {
        ffi::meshopt_computeClusterBounds(
            indices.as_ptr(),
            indices.len(),
            positions,
            vertices.len() * 3,
            ::std::mem::size_of::<f32>() * 3,
        )
    };
    ffi_bounds
}

pub fn compute_meshlet_bounds<T: DecodePosition>(meshlet: &Meshlet, vertices: &[T]) -> Bounds {
    let vertices = vertices
        .iter()
        .map(|vertex| vertex.decode_position())
        .collect::<Vec<[f32; 3]>>();
    let positions = vertices.as_ptr() as *const f32;
    let ffi_bounds = unsafe {
        // TODO: Should change mesh optimizer take meshlet by reference
        ffi::meshopt_computeMeshletBounds(
            *meshlet,
            positions,
            vertices.len() * 3,
            ::std::mem::size_of::<f32>() * 3,
        )
    };
    ffi_bounds
}
