extern crate float_cmp;
extern crate meshopt;
extern crate tobj;

use float_cmp::ApproxEqUlps;
use std::mem;
use std::path::Path;

const CACHE_SIZE: usize = 16;

trait FromVertex {
    fn from_vertex(&mut self, vertex: &Vertex);
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
struct PackedVertex {
    p: [u16; 4],
    n: [u8; 4],
    t: [u16; 2],
}

impl FromVertex for PackedVertex {
    fn from_vertex(&mut self, vertex: &Vertex) {
        self.p[0] = meshopt::quantize_half(vertex.p[0]) as u16;
        self.p[1] = meshopt::quantize_half(vertex.p[1]) as u16;
        self.p[2] = meshopt::quantize_half(vertex.p[2]) as u16;
        self.p[3] = 0u16;

        self.n[0] = meshopt::quantize_snorm(vertex.n[0], 8) as u8;
        self.n[1] = meshopt::quantize_snorm(vertex.n[1], 8) as u8;
        self.n[2] = meshopt::quantize_snorm(vertex.n[2], 8) as u8;
        self.n[3] = 0u8;

        self.t[0] = meshopt::quantize_half(vertex.t[0]) as u16;
        self.t[1] = meshopt::quantize_half(vertex.t[1]) as u16;
    }
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
struct PackedVertexOct {
    p: [u16; 3],
    n: [u8; 2], // octahedron encoded normal, aliases .pw
    t: [u16; 2],
}

impl FromVertex for PackedVertexOct {
    fn from_vertex(&mut self, vertex: &Vertex) {
        self.p[0] = meshopt::quantize_half(vertex.p[0]) as u16;
        self.p[1] = meshopt::quantize_half(vertex.p[1]) as u16;
        self.p[2] = meshopt::quantize_half(vertex.p[2]) as u16;

        let nsum = vertex.n[0].abs() + vertex.n[1].abs() + vertex.n[2].abs();
        let nx = vertex.n[0] / nsum;
        let ny = vertex.n[1] / nsum;
        let nz = vertex.n[2];

        let nu = if nz >= 0f32 {
            nx
        } else {
            (1f32 - ny.abs()) * if nx >= 0f32 {
                1f32
            } else {
                -1f32
            }
        };

        let nv = if nz >= 0f32 {
            ny
        } else {
            (1f32 - nx.abs()) * if ny >= 0f32 {
                1f32
            } else {
                -1f32
            }
        };

        self.n[0] = meshopt::quantize_snorm(nu, 8) as u8;
        self.n[1] = meshopt::quantize_snorm(nv, 8) as u8;

        self.t[0] = meshopt::quantize_half(vertex.t[0]) as u16;
        self.t[1] = meshopt::quantize_half(vertex.t[1]) as u16;
    }
}

#[derive(Default, Debug, Copy, Clone, PartialOrd)]
#[repr(C)]
struct Vertex {
    p: [f32; 3],
    n: [f32; 3],
    t: [f32; 2],
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Vertex) -> bool {
        self.p[0].approx_eq_ulps(&other.p[0], 2)
            && self.p[1].approx_eq_ulps(&other.p[1], 2)
            && self.p[2].approx_eq_ulps(&other.p[2], 2)
            && self.n[0].approx_eq_ulps(&other.n[0], 2)
            && self.n[1].approx_eq_ulps(&other.n[1], 2)
            && self.n[2].approx_eq_ulps(&other.n[2], 2)
            && self.t[0].approx_eq_ulps(&other.t[0], 2)
            && self.t[1].approx_eq_ulps(&other.t[1], 2)
    }
}

impl Eq for Vertex {}

impl Vertex {
        }

impl meshopt::DecodePosition for Vertex {
    fn decode_position(&self) -> [f32; 3] {
        self.p
    }
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
#[repr(C)]
struct Triangle {
    v: [Vertex; 3],
}

impl Triangle {
    #[allow(dead_code)]
    fn rotate(&mut self) -> bool {
        if self.v[1] < self.v[2] && self.v[0] > self.v[1] {
            // 1 is minimum, rotate 012 => 120
            let tv = self.v[0].clone();
            self.v[0] = self.v[1];
            self.v[1] = self.v[2];
            self.v[2] = tv;
        } else if self.v[0] > self.v[2] && self.v[1] > self.v[2] {
            // 2 is minimum, rotate 012 => 201
            let tv = self.v[2].clone();
            self.v[2] = self.v[1];
            self.v[1] = self.v[0];
            self.v[0] = tv;
        }
        self.v[0] != self.v[1] && self.v[0] != self.v[2] && self.v[1] != self.v[2]
    }
}

#[derive(Default, Debug, Clone)]
struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    #[allow(dead_code)]
    fn is_valid(&self) -> bool {
        if self.indices.len() % 3 != 0 {
            return false;
        } else {
            for i in 0..self.indices.len() {
                if self.indices[i] as usize >= self.vertices.len() {
                    return false;
                }
            }
        }
        true
    }

    fn load_obj(path: &Path) -> Self {
        let obj = tobj::load_obj(path);
        assert!(obj.is_ok());
        let (models, _) = obj.unwrap();

        assert!(models.len() == 1);

        let mut merged_vertices: Vec<Vertex> = Vec::new();

        let mut total_indices = 0;

        for (i, m) in models.iter().enumerate() {
            let mut vertices: Vec<Vertex> = Vec::new();

            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("Size of model[{}].indices: {}", i, mesh.indices.len());
            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);

            total_indices += mesh.indices.len();

            for i in 0..mesh.indices.len() {
                let index = mesh.indices[i] as usize;

                // pos = [x, y, z]
                let p = [
                    mesh.positions[index * 3],
                    mesh.positions[index * 3 + 1],
                    mesh.positions[index * 3 + 2],
                ];

                let n = if !mesh.normals.is_empty() {
                    // normal = [x, y, z]
                    [
                        mesh.normals[index * 3],
                        mesh.normals[index * 3 + 1],
                        mesh.normals[index * 3 + 2],
                    ]
                } else {
                    [0f32, 0f32, 0f32]
                };

                let t = if !mesh.texcoords.is_empty() {
                    // tex coord = [u, v];
                    [mesh.texcoords[index * 2], mesh.texcoords[index * 2 + 1]]
                } else {
                    [0f32, 0f32]
                };

                vertices.push(Vertex { p, n, t });
            }

            merged_vertices.append(&mut vertices);
        }

        let (total_vertices, vertex_remap) = generate_vertex_remap(total_indices, &merged_vertices);

        let mut mesh = Self::default();

        mesh.indices.resize(total_indices, 0u32);
        unsafe {
            meshopt::ffi::meshopt_remapIndexBuffer(
                mesh.indices.as_ptr() as *mut ::std::os::raw::c_uint,
                ::std::ptr::null(),
                total_indices,
                vertex_remap.as_ptr() as *const ::std::os::raw::c_uint,
            );
        }

        mesh.vertices.resize(total_vertices, Vertex::default());
        unsafe {
            meshopt::ffi::meshopt_remapVertexBuffer(
                mesh.vertices.as_ptr() as *mut ::std::os::raw::c_void,
                merged_vertices.as_ptr() as *const ::std::os::raw::c_void,
                total_indices,
                mem::size_of::<Vertex>(),
                vertex_remap.as_ptr() as *const ::std::os::raw::c_uint,
            );
        }

        println!(
            "# {:?}: {} vertices, {} triangles",
            path,
            mesh.vertices.len(),
            mesh.indices.len() / 3
        );

        mesh
    }

    #[allow(dead_code)]
    fn save_obj(&self, _path: &Path) {}

    #[allow(dead_code)]
    fn create_plane(size: u32) -> Self {
        let mut mesh = Self {
            vertices: Vec::with_capacity((size as usize + 1) * (size as usize + 1)),
            indices: Vec::with_capacity(size as usize * size as usize * 6),
        };

        for y in 0..(size + 1) {
            for x in 0..(size + 1) {
                mesh.vertices.push(Vertex {
                    p: [x as f32, y as f32, 0f32],
                    n: [0f32, 0f32, 1f32],
                    t: [x as f32 / size as f32, y as f32 / size as f32],
                });
            }
        }

        for y in 0..size {
            for x in 0..size {
                mesh.indices.push((y + 0) * (size + 1) + (x + 0));
                mesh.indices.push((y + 0) * (size + 1) + (x + 1));
                mesh.indices.push((y + 1) * (size + 1) + (x + 0));

                mesh.indices.push((y + 1) * (size + 1) + (x + 0));
                mesh.indices.push((y + 0) * (size + 1) + (x + 1));
                mesh.indices.push((y + 1) * (size + 1) + (x + 1));
            }
        }

        println!(
            "# tessellated plane: {} vertices, {} triangles",
            mesh.vertices.len(),
            mesh.indices.len() / 3
        );
        mesh
    }

    #[allow(dead_code)]
    fn encode_index(&self) {
        unimplemented!();
    }

    #[allow(dead_code)]
    fn stripify(&self) {
        unimplemented!();
    }

    #[allow(dead_code)]
    fn deindex(&self) -> Vec<Triangle> {
        let tri_count = self.indices.len() / 3;
        let mut result = Vec::with_capacity(tri_count);

        for i in 0..tri_count {
            let i0 = self.indices[i * 3 + 0];
            let i1 = self.indices[i * 3 + 1];
            let i2 = self.indices[i * 3 + 2];
            let mut tri = Triangle {
                v: [
                    self.vertices[i0 as usize].clone(),
                    self.vertices[i1 as usize].clone(),
                    self.vertices[i2 as usize].clone(),
                ],
            };

            // skip degenerate triangles since some algorithms don't preserve them
            if tri.rotate() {
                result.push(tri);
            }
        }

        result
    }
}

fn generate_vertex_remap<T>(index_count: usize, vertices: &[T]) -> (usize, Vec<u32>) {
    let mut remap: Vec<u32> = Vec::new();
    remap.resize(index_count, 0u32);
    let vertex_count = unsafe {
        meshopt::ffi::meshopt_generateVertexRemap(
            remap.as_ptr() as *mut ::std::os::raw::c_uint, // vb
            ::std::ptr::null(),                            // ib
            index_count,
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            index_count,
            mem::size_of::<T>(),
        )
    };

    (vertex_count, remap)
}

fn pack_vertices<T>(input: &[T]) -> Vec<u8> {
    let conservative_size =
        unsafe { meshopt::ffi::meshopt_encodeVertexBufferBound(input.len(), mem::size_of::<T>()) };

    println!(
        "Conservative size is: {}, sizeof is: {}",
        conservative_size,
        mem::size_of::<T>()
    );

    let mut encoded_data: Vec<u8> = Vec::new();
    encoded_data.resize(conservative_size, 0u8);

    let encoded_size = unsafe {
        meshopt::ffi::meshopt_encodeVertexBuffer(
            encoded_data.as_ptr() as *mut ::std::os::raw::c_uchar,
            encoded_data.len(),
            input.as_ptr() as *const ::std::os::raw::c_void,
            input.len(),
            mem::size_of::<T>(),
        )
    };

    encoded_data.resize(encoded_size, 0u8);
    println!("Encoded size is: {}", encoded_size);
    encoded_data

    /*assert_eq!(encoded_data.len() % mem::size_of::<T>(), 0);
    
    let typed_data = unsafe {
        let typed_count = encoded_data.len() / mem::size_of::<T>();
        let typed_ptr = encoded_data.as_mut_ptr() as *mut T;
        Vec::from_raw_parts(typed_ptr,
                            typed_count,
                            typed_count)
    };
    
    mem::forget(encoded_data);
    typed_data*/
}

fn encode_index_coverage() {
    //unimplemented!();
}

fn encode_vertex_coverage() {
    let mut vertices: Vec<PackedVertexOct> = Vec::with_capacity(4);

    vertices.push(PackedVertexOct {
        p: [0, 0, 0],
        n: [0, 0],
        t: [0, 0],
    });

    vertices.push(PackedVertexOct {
        p: [300, 0, 0],
        n: [0, 0],
        t: [500, 0],
    });

    vertices.push(PackedVertexOct {
        p: [0, 300, 0],
        n: [0, 0],
        t: [0, 500],
    });

    vertices.push(PackedVertexOct {
        p: [300, 300, 0],
        n: [0, 0],
        t: [500, 500],
    });

    let _encoded = pack_vertices(&vertices);
}

fn process_coverage() {
    encode_index_coverage();
    encode_vertex_coverage();
}

fn optimize_mesh(mesh: &Mesh, name: &str, opt: fn(mesh: &mut Mesh)) {
    let mut copy = mesh.clone();
    opt(&mut copy);

    let vcs =
        meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), CACHE_SIZE as u32, 0, 0);

    let vfs =
        meshopt::analyze_vertex_fetch(&copy.indices, copy.vertices.len(), mem::size_of::<Vertex>());

    let os = meshopt::analyze_overdraw(&copy.indices, &copy.vertices);

    let vcs_nv = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 32, 32, 32);

    let vcs_amd = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 14, 64, 128);

    let vcs_intel = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 128, 0, 0);

    println!(
        "{}: ACMR {} ATVR {} (NV {} AMD {} Intel {}) Overfetch {} Overdraw {}",
        name,
        vcs.acmr,
        vcs.atvr,
        vcs_nv.atvr,
        vcs_amd.atvr,
        vcs_intel.atvr,
        vfs.overfetch,
        os.overdraw
    );
}

fn opt_none(_: &mut Mesh) {}

fn opt_random_shuffle(_mesh: &mut Mesh) {
    //
}

fn opt_cache(_mesh: &mut Mesh) {
    //
}

fn opt_cache_fifo(_mesh: &mut Mesh) {
    //
}

fn opt_overdraw(_mesh: &mut Mesh) {
    //
}
fn opt_fetch(_mesh: &mut Mesh) {
    //
}
fn opt_fetch_remap(_mesh: &mut Mesh) {
    //
}
fn opt_complete(_mesh: &mut Mesh) {
    //

fn pack_vertex<T: FromVertex + Default>(mesh: &Mesh, name: &str) {
    let mut vertices: Vec<T> = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let mut packed_vertex = T::default();
        packed_vertex.from_vertex(&vertex);
        vertices.push(packed_vertex);
    }
    pack_mesh(&mut vertices, &mesh.vertices);

    let compressed_size = compress(&mut vertices);

    println!(
        "VtxPack{}  : {} bits/vertex (post-deflate {} bits/vertices)",
        name,
        (vertices.len() * mem::size_of::<T>() * 8) as f64 / mesh.vertices.len() as f64,
        (compressed_size * 8) as f64 / mesh.vertices.len() as f64);
}

fn pack_mesh<T, U>(output: &mut [T], input: &[U]) {

}

fn compress<T>(vertices: &mut [T]) -> usize {
    0
}

fn main() {
    let mesh = Mesh::load_obj(&Path::new("examples/pirate.obj"));

    optimize_mesh(&mesh, "Original", opt_none);
    optimize_mesh(&mesh, "Random", opt_random_shuffle);
    optimize_mesh(&mesh, "Cache", opt_cache);
    optimize_mesh(&mesh, "CacheFifo", opt_cache_fifo);
    optimize_mesh(&mesh, "Overdraw", opt_overdraw);
    optimize_mesh(&mesh, "Fetch", opt_fetch);
    optimize_mesh(&mesh, "FetchMap", opt_fetch_remap);
    optimize_mesh(&mesh, "Complete", opt_complete);

    process_coverage();
}
