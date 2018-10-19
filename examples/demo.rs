extern crate float_cmp;
extern crate libc;
extern crate meshopt;
extern crate miniz_oxide_c_api;
extern crate rand;
extern crate tobj;

use float_cmp::ApproxEqUlps;
use rand::{thread_rng, Rng};
use std::mem;
use std::path::{Path, PathBuf};
use std::io::prelude::*;
use std::io::BufWriter;
use std::fs::File;

const CACHE_SIZE: usize = 16;

fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    unsafe {
        ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
    }
}

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
            (1f32 - ny.abs()) * if nx >= 0f32 { 1f32 } else { -1f32 }
        };

        let nv = if nz >= 0f32 {
            ny
        } else {
            (1f32 - nx.abs()) * if ny >= 0f32 { 1f32 } else { -1f32 }
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

impl Vertex {}

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

impl Ord for Triangle {
    fn cmp(&self, other: &Triangle) -> ::std::cmp::Ordering {
        let lhs = any_as_u8_slice(&self);
        let rhs = any_as_u8_slice(&other);
        lhs.cmp(&rhs)
    }
}

#[derive(Default, Debug, Clone)]
struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl PartialEq for Mesh {
    fn eq(&self, other: &Mesh) -> bool {
        let mut lt = self.deindex();
        let mut rt = other.deindex();
        lt.sort();
        rt.sort();
        lt == rt
    }
}

impl Eq for Mesh {}

impl Mesh {
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
            //println!("model[{}].name = \'{}\'", i, m.name);
            //println!("Size of model[{}].indices: {}", i, mesh.indices.len());
            //println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);

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

        let (total_vertices, vertex_remap) = meshopt::generate_vertex_remap(total_indices, &merged_vertices);

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

    fn save_obj(&self, path: &Path) -> std::io::Result<()> {
        let mut buffer = File::create(path)?;

        for i in 0..self.vertices.len() {
            write!(buffer, "v {} {} {}\n", self.vertices[i].p[0], self.vertices[i].p[1], self.vertices[i].p[2])?;
            write!(buffer, "vn {} {} {}\n", self.vertices[i].n[0], self.vertices[i].n[1], self.vertices[i].n[2])?;
            write!(buffer, "vt {} {} {}\n", self.vertices[i].t[0], self.vertices[i].t[1], 0f32)?;
    }

        for i in (0..self.indices.len()).step_by(3) {
            let i0 = self.indices[i + 0] + 1;
            let i1 = self.indices[i + 1] + 1;
            let i2 = self.indices[i + 2] + 1;
            write!(buffer, "f {}/{}/{} {}/{}/{} {}/{}/{}\n", i0, i0, i0, i1, i1, i1, i2, i2, i2);
        }

        Ok(())
    }

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

fn pack_vertices<T>(input: &[T]) -> Vec<u8> {
    let conservative_size =
        unsafe { meshopt::ffi::meshopt_encodeVertexBufferBound(input.len(), mem::size_of::<T>()) };

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
    encoded_data
}

fn encode_index_coverage() {
    println!("encode_index_coverage: unimplemented");
    //unimplemented!();
    /*
    // note: 4 6 5 triangle here is a combo-breaker:
    // we encode it without rotating, a=next, c=next - this means we do *not* bump next to 6
    // which means that the next triangle can't be encoded via next sequencing!
    const unsigned int indices[] = {0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9};
    const size_t index_count = sizeof(indices) / sizeof(indices[0]);
    const size_t vertex_count = 10;
    
    std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(index_count, vertex_count));
    buffer.resize(meshopt_encodeIndexBuffer(&buffer[0], buffer.size(), indices, index_count));
    
    // check that encode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(i);
        size_t result = meshopt_encodeIndexBuffer(i == 0 ? 0 : &shortbuffer[0], i, indices, index_count);
        (void)result;
    
        if (i == buffer.size())
            assert(result == buffer.size());
        else
            assert(result == 0);
    }
    
    // check that decode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    unsigned int destination[index_count];
    
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(buffer.begin(), buffer.begin() + i);
        int result = meshopt_decodeIndexBuffer(destination, index_count, i == 0 ? 0 : &shortbuffer[0], i);
        (void)result;
    
        if (i == buffer.size())
            assert(result == 0);
        else
            assert(result < 0);
    }
    
    // check that decoder doesn't accept extra bytes after a valid stream
    {
        std::vector<unsigned char> largebuffer(buffer);
        largebuffer.push_back(0);
    
        int result = meshopt_decodeIndexBuffer(destination, index_count, &largebuffer[0], largebuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    
    // check that decoder doesn't accept malformed headers
    {
        std::vector<unsigned char> brokenbuffer(buffer);
        brokenbuffer[0] = 0;
    
        int result = meshopt_decodeIndexBuffer(destination, index_count, &brokenbuffer[0], brokenbuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    */}

fn encode_vertex_coverage() {
    println!("encode_vertex_coverage: unimplemented");

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

    /*
    // check that encode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(i);
        size_t result = meshopt_encodeVertexBuffer(i == 0 ? 0 : &shortbuffer[0], i, vertices, vertex_count, sizeof(PV));
        (void)result;
    
        if (i == buffer.size())
            assert(result == buffer.size());
        else
            assert(result == 0);
    }
    
    // check that decode is memory-safe; note that we reallocate the buffer for each try to make sure ASAN can verify buffer access
    PV destination[vertex_count];
    
    for (size_t i = 0; i <= buffer.size(); ++i)
    {
        std::vector<unsigned char> shortbuffer(buffer.begin(), buffer.begin() + i);
        int result = meshopt_decodeVertexBuffer(destination, vertex_count, sizeof(PV), i == 0 ? 0 : &shortbuffer[0], i);
        (void)result;
    
        if (i == buffer.size())
            assert(result == 0);
        else
            assert(result < 0);
    }
    
    // check that decoder doesn't accept extra bytes after a valid stream
    {
        std::vector<unsigned char> largebuffer(buffer);
        largebuffer.push_back(0);
    
        int result = meshopt_decodeVertexBuffer(destination, vertex_count, sizeof(PV), &largebuffer[0], largebuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    
    // check that decoder doesn't accept malformed headers
    {
        std::vector<unsigned char> brokenbuffer(buffer);
        brokenbuffer[0] = 0;
    
        int result = meshopt_decodeVertexBuffer(destination, vertex_count, sizeof(PV), &brokenbuffer[0], brokenbuffer.size());
        (void)result;
    
        assert(result < 0);
    }
    */
}

fn process_coverage() {
    encode_index_coverage();
    encode_vertex_coverage();
}

fn optimize_mesh(mesh: &Mesh, name: &str, opt: fn(mesh: &mut Mesh)) {
    let mut copy = mesh.clone();

    assert_eq!(mesh, &copy);
    assert!(copy.is_valid());

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
        "{:9}: ACMR {:.6} ATVR {:.6} (NV {:.6} AMD {:.6} Intel {:.6}) Overfetch {:.6} Overdraw {:.6}",
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

fn opt_none(_: &mut Mesh) {
    // no-op
}

fn opt_random_shuffle(mesh: &mut Mesh) {
    let face_count = mesh.indices.len() / 3;
    let mut faces: Vec<usize> = (0..face_count).map(|x| x).collect();
    thread_rng().shuffle(&mut faces);

    let mut result: Vec<u32> = Vec::with_capacity(mesh.indices.len());
    faces.iter().for_each(|face| {
        result.push(mesh.indices[faces[*face as usize] * 3 + 0]);
        result.push(mesh.indices[faces[*face as usize] * 3 + 1]);
        result.push(mesh.indices[faces[*face as usize] * 3 + 2]);
    });

    mesh.indices = result;
}

fn opt_cache(mesh: &mut Mesh) {
    meshopt::optimize_vertex_cache_in_place(&mut mesh.indices, mesh.vertices.len());
}

fn opt_cache_fifo(mesh: &mut Mesh) {
    meshopt::optimize_vertex_cache_fifo_in_place(
        &mut mesh.indices,
        mesh.vertices.len(),
        CACHE_SIZE as u32,
    );
}

fn opt_overdraw(mesh: &mut Mesh) {
    // use worst-case ACMR threshold so that overdraw optimizer can sort *all* triangles
    // warning: this significantly deteriorates the vertex cache efficiency so it is not advised; look at `opt_complete` for the recommended method
    let threshold = 3f32;
    meshopt::optimize_overdraw_in_place(&mut mesh.indices, &mesh.vertices, threshold);
}

fn opt_fetch(mesh: &mut Mesh) {
    meshopt::optimize_vertex_fetch_in_place(&mut mesh.indices, &mut mesh.vertices);
}

fn opt_fetch_remap(mesh: &mut Mesh) {
    let remap = meshopt::optimize_vertex_fetch_remap(&mesh.indices, mesh.vertices.len());
    mesh.indices = meshopt::remap_index_buffer(Some(&mesh.indices), mesh.indices.len(), &remap);
    mesh.vertices = meshopt::remap_vertex_buffer(&mesh.vertices, &remap);
}

fn opt_complete(mesh: &mut Mesh) {
    // vertex cache optimization should go first as it provides starting order for overdraw
    meshopt::optimize_vertex_cache_in_place(&mut mesh.indices, mesh.vertices.len());

    // reorder indices for overdraw, balancing overdraw and vertex cache efficiency
    let threshold = 1.05f32; // allow up to 5% worse ACMR to get more reordering opportunities for overdraw
    meshopt::optimize_overdraw_in_place(&mut mesh.indices, &mesh.vertices, threshold);

    // vertex fetch optimization should go last as it depends on the final index order
    let final_size =
        meshopt::optimize_vertex_fetch_in_place(&mut mesh.indices, &mut mesh.vertices);
    mesh.vertices.resize(final_size, Default::default());
}

fn stripify(mesh: &Mesh) {
    let strip = meshopt::stripify(&mesh.indices, mesh.vertices.len());
    let mut copy = mesh.clone();
    copy.indices = meshopt::unstripify(&strip);

    assert!(copy.is_valid());
    assert_eq!(mesh, &copy);

    let vcs = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), CACHE_SIZE as u32, 0, 0);
    let vcs_nv = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 32, 32, 32);
    let vcs_amd = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 14, 64, 128);
    let vcs_intel = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 128, 0, 0);

    println!("Stripify : ACMR {:.6} ATVR {:.6} (NV {:.6} AMD {:.6} Intel {:.6}); {} strip indices ({:.1}%)",
           vcs.acmr,
           vcs.atvr,
           vcs_nv.atvr,
           vcs_amd.atvr,
           vcs_intel.atvr,
           strip.len() as i32,
           strip.len() as f64 / mesh.indices.len() as f64 * 100f64);
 }

fn simplify(mesh: &Mesh) {
    println!("simplify: unimplemented");
    /*
    static const size_t lod_count = 5;
    
    double start = timestamp();
    
    // generate 4 LOD levels (1-4), with each subsequent LOD using 70% triangles
    // note that each LOD uses the same (shared) vertex buffer
    std::vector<unsigned int> lods[lod_count];
    
    lods[0] = mesh.indices;
    
    for (size_t i = 1; i < lod_count; ++i)
    {
        std::vector<unsigned int>& lod = lods[i];
    
        float threshold = powf(0.7f, float(i));
        size_t target_index_count = size_t(mesh.indices.size() * threshold) / 3 * 3;
        float target_error = 1e-3f;
    
        // we can simplify all the way from base level or from the last result
        // simplifying from the base level sometimes produces better results, but simplifying from last level is faster
        const std::vector<unsigned int>& source = lods[i - 1];
    
        lod.resize(source.size());
        lod.resize(meshopt_simplify(&lod[0], &source[0], source.size(), &mesh.vertices[0].px, mesh.vertices.size(), sizeof(Vertex), std::min(source.size(), target_index_count), target_error));
    }
    
    double middle = timestamp();
    
    // optimize each individual LOD for vertex cache & overdraw
    for (size_t i = 0; i < lod_count; ++i)
    {
        std::vector<unsigned int>& lod = lods[i];
    
        meshopt_optimizeVertexCache(&lod[0], &lod[0], lod.size(), mesh.vertices.size());
        meshopt_optimizeOverdraw(&lod[0], &lod[0], lod.size(), &mesh.vertices[0].px, mesh.vertices.size(), sizeof(Vertex), 1.0f);
    }
    
    // concatenate all LODs into one IB
    // note: the order of concatenation is important - since we optimize the entire IB for vertex fetch,
    // putting coarse LODs first makes sure that the vertex range referenced by them is as small as possible
    // some GPUs process the entire range referenced by the index buffer region so doing this optimizes the vertex transform
    // cost for coarse LODs
    // this order also produces much better vertex fetch cache coherency for coarse LODs (since they're essentially optimized first)
    // somewhat surprisingly, the vertex fetch cache coherency for fine LODs doesn't seem to suffer that much.
    size_t lod_index_offsets[lod_count] = {};
    size_t lod_index_counts[lod_count] = {};
    size_t total_index_count = 0;
    
    for (int i = lod_count - 1; i >= 0; --i)
    {
        lod_index_offsets[i] = total_index_count;
        lod_index_counts[i] = lods[i].size();
    
        total_index_count += lods[i].size();
    }
    
    std::vector<unsigned int> indices(total_index_count);
    
    for (size_t i = 0; i < lod_count; ++i)
    {
        memcpy(&indices[lod_index_offsets[i]], &lods[i][0], lods[i].size() * sizeof(lods[i][0]));
    }
    
    std::vector<Vertex> vertices = mesh.vertices;
    
    // vertex fetch optimization should go last as it depends on the final index order
    // note that the order of LODs above affects vertex fetch results
    meshopt_optimizeVertexFetch(&vertices[0], &indices[0], indices.size(), &vertices[0], vertices.size(), sizeof(Vertex));
    
    double end = timestamp();
    
    printf("%-9s: %d triangles => %d LOD levels down to %d triangles in %.2f msec, optimized in %.2f msec\n",
           "Simplify",
           int(lod_index_counts[0]) / 3, int(lod_count), int(lod_index_counts[lod_count - 1]) / 3,
           (middle - start) * 1000, (end - middle) * 1000);
    
    // for using LOD data at runtime, in addition to vertices and indices you have to save lod_index_offsets/lod_index_counts.
    
    {

        let vcs =
        meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), CACHE_SIZE as u32, 0, 0);

    let vfs =
        meshopt::analyze_vertex_fetch(&copy.indices, copy.vertices.len(), mem::size_of::<Vertex>());

    let vcs_nv = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 32, 32, 32);

    let vcs_amd = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 14, 64, 128);

    let vcs_intel = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 128, 0, 0);




        meshopt_VertexCacheStatistics vcs0 = meshopt_analyzeVertexCache(&indices[lod_index_offsets[0]], lod_index_counts[0], vertices.size(), kCacheSize, 0, 0);
        meshopt_VertexFetchStatistics vfs0 = meshopt_analyzeVertexFetch(&indices[lod_index_offsets[0]], lod_index_counts[0], vertices.size(), sizeof(Vertex));
        meshopt_VertexCacheStatistics vcsN = meshopt_analyzeVertexCache(&indices[lod_index_offsets[lod_count - 1]], lod_index_counts[lod_count - 1], vertices.size(), kCacheSize, 0, 0);
        meshopt_VertexFetchStatistics vfsN = meshopt_analyzeVertexFetch(&indices[lod_index_offsets[lod_count - 1]], lod_index_counts[lod_count - 1], vertices.size(), sizeof(Vertex));
    
        typedef PackedVertexOct PV;
    
        std::vector<PV> pv(vertices.size());
        packMesh(pv, vertices);
    
        std::vector<unsigned char> vbuf(meshopt_encodeVertexBufferBound(vertices.size(), sizeof(PV)));
        vbuf.resize(meshopt_encodeVertexBuffer(&vbuf[0], vbuf.size(), &pv[0], vertices.size(), sizeof(PV)));
    
        std::vector<unsigned char> ibuf(meshopt_encodeIndexBufferBound(indices.size(), vertices.size()));
        ibuf.resize(meshopt_encodeIndexBuffer(&ibuf[0], ibuf.size(), &indices[0], indices.size()));
    
        printf("%-9s  ACMR %f...%f Overfetch %f..%f Codec VB %.1f bits/vertex IB %.1f bits/triangle\n",
               "",
               vcs0.acmr, vcsN.acmr, vfs0.overfetch, vfsN.overfetch,
               double(vbuf.size()) / double(vertices.size()) * 8,
               double(ibuf.size()) / double(indices.size() / 3) * 8);
    }
    */
}

fn encode_index(mesh: &Mesh) {
    let encoded = meshopt::encode_index_buffer(&mesh.indices, mesh.vertices.len());
    let decoded = meshopt::decode_index_buffer::<u32>(&encoded, mesh.indices.len());
    let compressed = compress(&encoded);
    for i in (0..mesh.indices.len()).step_by(3) {
        assert!(
            (decoded[i + 0] == mesh.indices[i + 0]
                && decoded[i + 1] == mesh.indices[i + 1]
                && decoded[i + 2] == mesh.indices[i + 2])
                || (decoded[i + 1] == mesh.indices[i + 0]
                    && decoded[i + 2] == mesh.indices[i + 1]
                    && decoded[i + 0] == mesh.indices[i + 2])
                || (decoded[i + 2] == mesh.indices[i + 0]
                    && decoded[i + 0] == mesh.indices[i + 1]
                    && decoded[i + 1] == mesh.indices[i + 2])
        );
    }

    if mesh.vertices.len() <= 65536 {
        let decoded2 = meshopt::decode_index_buffer::<u16>(&encoded, mesh.indices.len());
        for i in (0..mesh.indices.len()).step_by(3) {
            assert!(
                decoded[i + 0] == decoded2[i + 0] as u32
                    && decoded[i + 1] == decoded2[i + 1] as u32
                    && decoded[i + 2] == decoded2[i + 2] as u32
            );
        }
    }

    println!(
        "IdxCodec : {:.1} bits/triangle (post-deflate {:.1} bits/triangle);",
        (encoded.len() * 8) as f64 / (mesh.indices.len() / 3) as f64,
        (compressed.len() * 8) as f64 / (mesh.indices.len() / 3) as f64
    );
}

fn encode_vertex<T: Clone + Default + Eq>(mesh: &Mesh, name: &str) {
    let mut packed: Vec<T> = Vec::new();
    packed.resize(mesh.vertices.len(), Default::default());
    pack_mesh(&mut packed, &mesh.vertices);

    let encoded = meshopt::encode_vertex_buffer(&packed);
    let decoded = meshopt::decode_vertex_buffer(&encoded, mesh.vertices.len());
    assert!(packed == decoded);

    let compressed = compress(&encoded);
    
    println!("VtxCodec{:1}: {:.1} bits/vertex (post-deflate {:.1} bits/vertex);",
        name,
        (encoded.len() * 8) as f64 / (mesh.vertices.len()) as f64,
        (compressed.len() * 8) as f64 / (mesh.vertices.len()) as f64);
}

fn pack_vertex<T: FromVertex + Clone + Default>(mesh: &Mesh, name: &str) {
    let mut vertices: Vec<T> = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let mut packed_vertex = T::default();
        packed_vertex.from_vertex(&vertex);
        vertices.push(packed_vertex);
    }
    pack_mesh(&mut vertices, &mesh.vertices);

    let compressed = compress(&mut vertices);

    println!(
        "VtxPack{}  : {:.1} bits/vertex (post-deflate {:.1} bits/vertices)",
        name,
        (vertices.len() * mem::size_of::<T>() * 8) as f64 / mesh.vertices.len() as f64,
        (compressed.len() * 8) as f64 / mesh.vertices.len() as f64
    );
}

fn pack_mesh<T, U>(output: &mut [T], input: &[U]) {
    println!("pack_mesh: unimplemented");
}

fn compress<T: Clone + Default>(data: &[T]) -> Vec<u8> {
    let input_size = data.len() * mem::size_of::<T>();
    let compress_bound = miniz_oxide_c_api::mz_compressBound(input_size as u32);
    let mut compress_buffer: Vec<u8> = Vec::new();
    compress_buffer.resize(compress_bound as usize, 0u8);
    let flags = miniz_oxide_c_api::tdefl_create_comp_flags_from_zip_params(
        6, //miniz_oxide_c_api::MZ_DEFAULT_LEVEL,
        15,
        miniz_oxide_c_api::MZ_DEFAULT_STRATEGY,
    );
    let compress_size = unsafe {
        miniz_oxide_c_api::tdefl_compress_mem_to_mem(
            compress_buffer.as_mut_ptr() as *mut ::libc::c_void,
            compress_buffer.len(),
            data.as_ptr() as *const ::libc::c_void,
            input_size,
            flags as i32,
        )
    };
    compress_buffer.resize(compress_size as usize, 0u8);
    compress_buffer
}

fn process(path: Option<PathBuf>) {
    let mesh = match path {
        Some(path) => {
            Mesh::load_obj(&path)
        },
        None => {
            Mesh::create_plane(200)
        }
    };

    optimize_mesh(&mesh, "Original", opt_none);
    optimize_mesh(&mesh, "Random", opt_random_shuffle);
    optimize_mesh(&mesh, "Cache", opt_cache);
    optimize_mesh(&mesh, "CacheFifo", opt_cache_fifo);
    optimize_mesh(&mesh, "Overdraw", opt_overdraw);
    optimize_mesh(&mesh, "Fetch", opt_fetch);
    optimize_mesh(&mesh, "FetchMap", opt_fetch_remap);
    optimize_mesh(&mesh, "Complete", opt_complete);

    let mut copy = mesh.clone();
    meshopt::optimize_vertex_cache_in_place(&mut copy.indices, copy.vertices.len());
    meshopt::optimize_vertex_fetch_in_place(&mut copy.indices, &mut copy.vertices);

    //copy.save_obj(Path::new("H:/Test.obj"));

    stripify(&copy);

    encode_index(&copy);
    pack_vertex::<PackedVertex>(&copy, "");
    encode_vertex::<PackedVertex>(&copy, "");
    encode_vertex::<PackedVertexOct>(&copy, "0");

    simplify(&mesh);
}

fn main() {
    //process(None);
    process(Some(Path::new("examples/pirate.obj").to_path_buf()));
}
