extern crate libc;
extern crate meshopt;
extern crate miniz_oxide;
extern crate rand;
extern crate tobj;

use meshopt::*;
use rand::{seq::SliceRandom, thread_rng};
use std::fs::File;
use std::io::prelude::*;
use std::mem;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const CACHE_SIZE: usize = 16;

fn elapsed_to_ms(elapsed: Duration) -> f32 {
    elapsed.subsec_nanos() as f32 / 1_000_000.0 + elapsed.as_secs() as f32 * 1_000.0
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
        let lhs = meshopt::utilities::any_as_u8_slice(&self);
        let rhs = meshopt::utilities::any_as_u8_slice(&other);
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

        for (_, m) in models.iter().enumerate() {
            let mut vertices: Vec<Vertex> = Vec::new();
            let mesh = &m.mesh;
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

        let (total_vertices, vertex_remap) = meshopt::generate_vertex_remap(&merged_vertices, None);

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
    fn save_obj(&self, path: &Path) -> std::io::Result<()> {
        let mut buffer = File::create(path)?;

        for i in 0..self.vertices.len() {
            write!(
                buffer,
                "v {} {} {}\n",
                self.vertices[i].p[0], self.vertices[i].p[1], self.vertices[i].p[2]
            )?;
            write!(
                buffer,
                "vn {} {} {}\n",
                self.vertices[i].n[0], self.vertices[i].n[1], self.vertices[i].n[2]
            )?;
            write!(
                buffer,
                "vt {} {} {}\n",
                self.vertices[i].t[0], self.vertices[i].t[1], 0f32
            )?;
        }

        for i in (0..self.indices.len()).step_by(3) {
            let i0 = self.indices[i + 0] + 1;
            let i1 = self.indices[i + 1] + 1;
            let i2 = self.indices[i + 2] + 1;
            write!(
                buffer,
                "f {}/{}/{} {}/{}/{} {}/{}/{}\n",
                i0, i0, i0, i1, i1, i1, i2, i2, i2
            );
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

fn optimize_mesh(mesh: &Mesh, name: &str, opt: fn(mesh: &mut Mesh)) {
    let mut copy = mesh.clone();

    assert_eq!(mesh, &copy);
    assert!(copy.is_valid());

    let optimize_start = Instant::now();
    opt(&mut copy);
    let optimize_elapsed = optimize_start.elapsed();

    let vcs =
        meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), CACHE_SIZE as u32, 0, 0);

    let vfs =
        meshopt::analyze_vertex_fetch(&copy.indices, copy.vertices.len(), mem::size_of::<Vertex>());

    let os = meshopt::analyze_overdraw(&copy.indices, &copy.vertices);

    let vcs_nv = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 32, 32, 32);

    let vcs_amd = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 14, 64, 128);

    let vcs_intel = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 128, 0, 0);

    println!(
        "{:9}: ACMR {:.6} ATVR {:.6} (NV {:.6} AMD {:.6} Intel {:.6}) Overfetch {:.6} Overdraw {:.6} in {:.2} msec",
        name,
        vcs.acmr,
        vcs.atvr,
        vcs_nv.atvr,
        vcs_amd.atvr,
        vcs_intel.atvr,
        vfs.overfetch,
        os.overdraw,
        elapsed_to_ms(optimize_elapsed),
    );
}

fn opt_none(_: &mut Mesh) {
    // no-op
}

fn opt_random_shuffle(mesh: &mut Mesh) {
    let face_count = mesh.indices.len() / 3;
    let mut faces: Vec<usize> = (0..face_count).map(|x| x).collect();
    let mut rng = thread_rng();
    faces.shuffle(&mut rng);

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
    let final_size = meshopt::optimize_vertex_fetch_in_place(&mut mesh.indices, &mut mesh.vertices);
    mesh.vertices.resize(final_size, Default::default());
}

fn stripify(mesh: &Mesh) {
    let process_start = Instant::now();
    let strip = meshopt::stripify(&mesh.indices, mesh.vertices.len());
    let process_elapsed = process_start.elapsed();

    let mut copy = mesh.clone();
    copy.indices = meshopt::unstripify(&strip);

    assert!(copy.is_valid());
    assert_eq!(mesh, &copy);

    let vcs =
        meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), CACHE_SIZE as u32, 0, 0);
    let vcs_nv = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 32, 32, 32);
    let vcs_amd = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 14, 64, 128);
    let vcs_intel = meshopt::analyze_vertex_cache(&copy.indices, copy.vertices.len(), 128, 0, 0);

    println!("Stripify : ACMR {:.6} ATVR {:.6} (NV {:.6} AMD {:.6} Intel {:.6}); {} strip indices ({:.1}%) in {:.2} msec",
        vcs.acmr,
        vcs.atvr,
        vcs_nv.atvr,
        vcs_amd.atvr,
        vcs_intel.atvr,
        strip.len() as i32,
        strip.len() as f64 / mesh.indices.len() as f64 * 100f64,
        elapsed_to_ms(process_elapsed),
    );
}

fn shadow(mesh: &Mesh) {
    let process_start = Instant::now();
    let mut shadow_indices = meshopt::generate_shadow_indices(&mesh.indices, &mesh.vertices);
    let process_elapsed = process_start.elapsed();

    // While you can't optimize the vertex data after shadow IB was constructed, you can and should optimize
    // the shadow IB for vertex cache. This is valuable even if the original indices array was optimized for
    // vertex cache!
    meshopt::optimize_vertex_cache_in_place(&mut shadow_indices, mesh.vertices.len());

    let vcs =
        meshopt::analyze_vertex_cache(&mesh.indices, mesh.vertices.len(), CACHE_SIZE as u32, 0, 0);
    let vcss = meshopt::analyze_vertex_cache(
        &shadow_indices,
        mesh.vertices.len(),
        CACHE_SIZE as u32,
        0,
        0,
    );

    let mut shadow_flags: Vec<usize> = vec![0; mesh.vertices.len()];
    let mut shadow_vertices: usize = 0;
    for shadow_index in shadow_indices {
        shadow_vertices += 1 - shadow_flags[shadow_index as usize];
        shadow_flags[shadow_index as usize] = 1;
    }

    println!("ShadowIB : ACMR {:.6} ({:.2}x improvement); {} shadow vertices ({:.2}x improvement) in {:.2} msec",
	       vcss.acmr,
           vcs.vertices_transformed as f64 / vcss.vertices_transformed as f64,
           shadow_vertices,
           mesh.vertices.len() as f64 / shadow_vertices as f64,
           elapsed_to_ms(process_elapsed)
    );
}

fn meshlets(mesh: &Mesh) {
    let max_vertices = 64;
    let max_triangles = 126;

    let process_start = Instant::now();
    let meshlets = meshopt::build_meshlets(
        &mesh.indices,
        mesh.vertices.len(),
        max_vertices,
        max_triangles,
    );
    let process_elapsed = process_start.elapsed();

    let mut avg_vertices = 0f64;
    let mut avg_triangles = 0f64;
    let mut not_full = 0usize;

    for meshlet in &meshlets {
        avg_vertices += meshlet.vertex_count as f64;
        avg_triangles += meshlet.triangle_count as f64;
        not_full += if (meshlet.vertex_count as usize) < max_vertices {
            1
        } else {
            0
        };
    }

    avg_vertices /= meshlets.len() as f64;
    avg_triangles /= meshlets.len() as f64;

    println!("Meshlets : {} meshlets (avg vertices {:.1}, avg triangles {:.1}, not full {}) in {:.2} msec",
        meshlets.len(),
        avg_vertices,
        avg_triangles,
        not_full,
        elapsed_to_ms(process_elapsed));

    let camera: [f32; 3] = [100.0, 100.0, 100.0];

    let mut rejected = 0;
    let mut rejected_s8 = 0;
    let mut rejected_alt = 0;
    let mut rejected_alt_s8 = 0;
    let mut accepted = 0;
    let mut accepted_s8 = 0;

    let test_start = Instant::now();
    for meshlet in &meshlets {
        let bounds = meshopt::compute_meshlet_bounds(&meshlet, &mesh.vertices);

        // trivial accept: we can't ever backface cull this meshlet
        if bounds.cone_cutoff >= 1f32 {
            accepted += 1;
        }

        if bounds.cone_cutoff_s8 >= 127 {
            accepted_s8 += 1;
        }

        // perspective projection: dot(normalize(cone_apex - camera_position), cone_axis) > cone_cutoff
        let mview: [f32; 3] = [
            bounds.cone_apex[0] - camera[0],
            bounds.cone_apex[1] - camera[1],
            bounds.cone_apex[2] - camera[2],
        ];

        let mviewlength = (mview[0] * mview[0] + mview[1] * mview[1] + mview[2] * mview[2]).sqrt();

        if mview[0] * bounds.cone_axis[0]
            + mview[1] * bounds.cone_axis[1]
            + mview[2] * bounds.cone_axis[2]
            >= bounds.cone_cutoff * mviewlength
        {
            rejected += 1;
        }

        if mview[0] * (bounds.cone_axis_s8[0] as f32 / 127.0)
            + mview[1] * (bounds.cone_axis_s8[1] as f32 / 127.0)
            + mview[2] * (bounds.cone_axis_s8[2] as f32 / 127.0)
            >= (bounds.cone_cutoff_s8 as f32 / 127.0) * mviewlength
        {
            rejected_s8 += 1;
        }

        // alternative formulation for perspective projection that doesn't use apex (and uses cluster bounding sphere instead):
        // dot(normalize(center - camera_position), cone_axis) > cone_cutoff + radius / length(center - camera_position)
        let cview: [f32; 3] = [
            bounds.center[0] - camera[0],
            bounds.center[1] - camera[1],
            bounds.center[2] - camera[2],
        ];

        let cviewlength = (cview[0] * cview[0] + cview[1] * cview[1] + cview[2] * cview[2]).sqrt();

        if cview[0] * bounds.cone_axis[0]
            + cview[1] * bounds.cone_axis[1]
            + cview[2] * bounds.cone_axis[2]
            >= bounds.cone_cutoff * cviewlength + bounds.radius
        {
            rejected_alt += 1;
        }

        if cview[0] * (bounds.cone_axis_s8[0] as f32 / 127.0)
            + cview[1] * (bounds.cone_axis_s8[1] as f32 / 127.0)
            + cview[2] * (bounds.cone_axis_s8[2] as f32 / 127.0)
            >= (bounds.cone_cutoff_s8 as f32 / 127.0) * cviewlength + bounds.radius
        {
            rejected_alt_s8 += 1;
        }
    }
    let test_elapsed = test_start.elapsed();

    println!("ConeCull : rejected apex {} ({:.1}%) / center {} ({:.1}%), trivially accepted {} ({:.1}%) in {:.2} msec",
	       rejected,
           rejected as f64 / (meshlets.len() as f64) * 100.0,
	       rejected_alt,
           rejected_alt as f64 / (meshlets.len() as f64) * 100.0,
	       accepted,
           accepted as f64 / (meshlets.len() as f64) * 100.0,
	       elapsed_to_ms(test_elapsed));

    println!("ConeCull8: rejected apex {} ({:.1}%) / center {} ({:.1}%), trivially accepted {} ({:.1}%) in {:.2} msec",
	       rejected_s8,
           rejected_s8 as f64 / (meshlets.len() as f64) * 100.0,
	       rejected_alt_s8,
           rejected_alt_s8 as f64 / (meshlets.len() as f64) * 100.0,
	       accepted_s8,
           accepted_s8 as f64 / (meshlets.len() as f64) * 100.0,
           elapsed_to_ms(test_elapsed));
}

fn simplify(mesh: &Mesh) {
    let lod_count = 5;

    let process_start = Instant::now();

    // generate 4 LOD levels (1-4), with each subsequent LOD using 70% triangles
    // note that each LOD uses the same (shared) vertex buffer
    let mut lods: Vec<Vec<u32>> = Vec::with_capacity(lod_count);
    lods.push(mesh.indices.clone());
    for i in 1..lod_count {
        let threshold = 0.7f32.powf(i as f32);
        let target_index_count = (mesh.indices.len() as f32 * threshold) as usize / 3 * 3;
        let target_error = 1e-3f32;
        let mut lod: Vec<u32>;
        {
            // we can simplify all the way from base level or from the last result
            // simplifying from the base level sometimes produces better results, but simplifying from last level is faster
            let src = &lods[lods.len() - 1];
            lod = meshopt::simplify(
                &src,
                &mesh.vertices,
                ::std::cmp::min(src.len(), target_index_count),
                target_error,
            );
        }
        lods.push(lod);
    }

    let process_elapsed = process_start.elapsed();
    let optimize_start = Instant::now();

    // optimize each individual LOD for vertex cache & overdraw
    for mut lod in &mut lods {
        meshopt::optimize_vertex_cache_in_place(&mut lod, mesh.vertices.len());
        meshopt::optimize_overdraw_in_place(&mut lod, &mesh.vertices, 1f32);
    }

    // concatenate all LODs into one IB
    // note: the order of concatenation is important - since we optimize the entire IB for vertex fetch,
    // putting coarse LODs first makes sure that the vertex range referenced by them is as small as possible
    // some GPUs process the entire range referenced by the index buffer region so doing this optimizes the vertex transform
    // cost for coarse LODs
    // this order also produces much better vertex fetch cache coherency for coarse LODs (since they're essentially optimized first)
    // somewhat surprisingly, the vertex fetch cache coherency for fine LODs doesn't seem to suffer that much.
    let mut lod_offsets: Vec<usize> = Vec::new();
    lod_offsets.resize(lod_count, 0);

    let mut lod_counts: Vec<usize> = Vec::new();
    lod_counts.resize(lod_count, 0);

    let mut total_index_count: usize = 0;
    for i in (0..lod_count).rev() {
        lod_offsets[i] = total_index_count;
        lod_counts[i] = lods[i].len();
        total_index_count += lod_counts[i];
    }

    let mut indices: Vec<u32> = Vec::new();
    indices.resize(total_index_count, 0u32);
    for i in 0..lod_count {
        let lod = &lods[i];
        let offset = lod_offsets[i];
        indices.splice(offset..(offset + lod.len()), lod.iter().cloned());
    }

    // vertex fetch optimization should go last as it depends on the final index order
    // note that the order of LODs above affects vertex fetch results
    let mut vertices = mesh.vertices.clone();
    let next_vertex = meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);
    vertices.resize(next_vertex, Default::default());

    let optimize_elapsed = optimize_start.elapsed();

    println!(
        "{:9}: {} triangles => {} LOD levels down to {} triangles in {:.2} msec, optimized in {:.2} msec",
        "Simplify",
        lod_counts[0] / 3,
        lod_count,
        lod_counts[lod_count - 1] / 3,
        elapsed_to_ms(process_elapsed),
        elapsed_to_ms(optimize_elapsed),
    );

    // for using LOD data at runtime, in addition to vertices and indices you have to save lod_index_offsets/lod_index_counts.
    let offset_n = lod_count - 1;

    let vcs_0 = meshopt::analyze_vertex_cache(
        &indices[lod_offsets[0]..(lod_offsets[0] + lod_counts[0])],
        vertices.len(),
        CACHE_SIZE as u32,
        0,
        0,
    );

    let vfs_0 = meshopt::analyze_vertex_fetch(
        &indices[lod_offsets[0]..(lod_offsets[0] + lod_counts[0])],
        vertices.len(),
        mem::size_of::<Vertex>(),
    );

    let vcs_n = meshopt::analyze_vertex_cache(
        &indices[lod_offsets[offset_n]..(lod_offsets[offset_n] + lod_counts[offset_n])],
        vertices.len(),
        CACHE_SIZE as u32,
        0,
        0,
    );

    let vfs_n = meshopt::analyze_vertex_fetch(
        &indices[lod_offsets[offset_n]..(lod_offsets[offset_n] + lod_counts[offset_n])],
        vertices.len(),
        mem::size_of::<Vertex>(),
    );

    let packed = pack_vertices::<PackedVertexOct>(&vertices);
    let encoded_vertices = meshopt::encode_vertex_buffer(&packed);
    let encoded_indices = meshopt::encode_index_buffer(&indices, vertices.len());

    println!("{:9}  ACMR {:.6}...{:.6} Overfetch {:.6}..{:.6} Codec VB {:.1} bits/vertex IB {:.1} bits/triangle",
        "",
        vcs_0.acmr,
        vcs_n.acmr,
        vfs_0.overfetch,
        vfs_n.overfetch,
        encoded_vertices.len() as f64 / vertices.len() as f64 * 8f64,
        encoded_indices.len() as f64 / (indices.len() as f64 / 3f64) * 8f64
    );
}

fn encode_index(mesh: &Mesh) {
    let encode_start = Instant::now();
    let encoded = meshopt::encode_index_buffer(&mesh.indices, mesh.vertices.len());
    let encode_elapsed = encode_start.elapsed();

    let decode_start = Instant::now();
    let decoded = meshopt::decode_index_buffer::<u32>(&encoded, mesh.indices.len());
    let decode_elapsed = decode_start.elapsed();

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
        "IdxCodec : {:.1} bits/triangle (post-deflate {:.1} bits/triangle); encode {:.2} msec, decode {:.2} msec ({:.2} GB/s)",
        (encoded.len() * 8) as f64 / (mesh.indices.len() / 3) as f64,
        (compressed.len() * 8) as f64 / (mesh.indices.len() / 3) as f64,
        elapsed_to_ms(encode_elapsed),
        elapsed_to_ms(decode_elapsed),
        ((decoded.len() * 4) as f64 / (1 << 30) as f64) / (elapsed_to_ms(decode_elapsed) as f64 / 1000.0),
    );
}

fn encode_vertex<T: FromVertex + Clone + Default + Eq>(mesh: &Mesh, name: &str) {
    let packed = pack_vertices::<T>(&mesh.vertices);

    let encode_start = Instant::now();
    let encoded = meshopt::encode_vertex_buffer(&packed);
    let encode_elapsed = encode_start.elapsed();

    let decode_start = Instant::now();
    let decoded = meshopt::decode_vertex_buffer(&encoded, mesh.vertices.len());
    let decode_elapsed = decode_start.elapsed();

    assert!(packed == decoded);

    let compressed = compress(&encoded);

    println!(
        "VtxCodec{:1}: {:.1} bits/vertex (post-deflate {:.1} bits/vertex); encode {:.2} msec, decode {:.2} msec ({:.2} GB/s)",
        name,
        (encoded.len() * 8) as f64 / (mesh.vertices.len()) as f64,
        (compressed.len() * 8) as f64 / (mesh.vertices.len()) as f64,
        elapsed_to_ms(encode_elapsed),
        elapsed_to_ms(decode_elapsed),
        ((decoded.len() * 4) as f64 / (1 << 30) as f64) / (elapsed_to_ms(decode_elapsed) as f64 / 1000.0),
    );
}

fn pack_mesh<T: FromVertex + Clone + Default>(mesh: &Mesh, name: &str) {
    let vertices = pack_vertices::<T>(&mesh.vertices);
    let compressed = compress(&vertices);

    println!(
        "VtxPack{}  : {:.1} bits/vertex (post-deflate {:.1} bits/vertices)",
        name,
        (vertices.len() * mem::size_of::<T>() * 8) as f64 / mesh.vertices.len() as f64,
        (compressed.len() * 8) as f64 / mesh.vertices.len() as f64
    );
}

fn compress<T: Clone + Default>(data: &[T]) -> Vec<u8> {
    use miniz_oxide::deflate::compress_to_vec;
    let bytes: &[u8] = typed_to_bytes(data);
    compress_to_vec(bytes, 6 /* 0-10 compression level */)
}

fn process(path: Option<PathBuf>, export: bool) {
    let mesh = match path {
        Some(ref path) => Mesh::load_obj(&path),
        None => {
            let mesh = Mesh::create_plane(200);
            if export {
                mesh.save_obj(Path::new("examples/plane.obj")).unwrap();
            }
            mesh
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

    if export {
        match path {
            Some(ref path) => {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let new_path = format!("examples/{}_opt.obj", stem);
                copy.save_obj(Path::new(&new_path)).unwrap();
            }
            None => {
                copy.save_obj(Path::new("examples/plane_opt.obj")).unwrap();
            }
        }
    }

    stripify(&copy);
    meshlets(&copy);
    shadow(&copy);

    encode_index(&copy);
    pack_mesh::<PackedVertex>(&copy, "");
    encode_vertex::<PackedVertex>(&copy, "");
    encode_vertex::<PackedVertexOct>(&copy, "0");

    simplify(&mesh);
}

fn main() {
    let export = false;
    process(None, export);
    process(Some(Path::new("examples/pirate.obj").to_path_buf()), export);
}
