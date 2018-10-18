extern crate float_cmp;
extern crate meshopt;
extern crate tobj;
extern crate miniz_oxide_c_api;

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

impl PartialEq for Mesh {
    fn eq(&self, other: &Mesh) -> bool {
        if self.vertices != other.vertices {
            return false;
        }

        if self.indices != other.indices {
            return false;
        }

        /*
            std::vector<Triangle> lt, rt;
            deindexMesh(lt, lhs);
            deindexMesh(rt, rhs);

            std::sort(lt.begin(), lt.end());
            std::sort(rt.begin(), rt.end());

            return lt.size() == rt.size() && memcmp(&lt[0], &rt[0], lt.size() * sizeof(Triangle)) == 0;
        */

        true
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
    */
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
    /*
    std::vector<unsigned int> faces(mesh.indices.size() / 3);

	for (size_t i = 0; i < faces.size(); ++i)
		faces[i] = static_cast<unsigned int>(i);

	std::random_shuffle(faces.begin(), faces.end());

	std::vector<unsigned int> result(mesh.indices.size());

	for (size_t i = 0; i < faces.size(); ++i)
	{
		result[i * 3 + 0] = mesh.indices[faces[i] * 3 + 0];
		result[i * 3 + 1] = mesh.indices[faces[i] * 3 + 1];
		result[i * 3 + 2] = mesh.indices[faces[i] * 3 + 2];
	}

	mesh.indices.swap(result);
    */
}

fn opt_cache(mesh: &mut Mesh) {
    meshopt::optimize_vertex_cache_in_place(&mut mesh.indices, mesh.vertices.len());
}

fn opt_cache_fifo(mesh: &mut Mesh) {
    meshopt::optimize_vertex_cache_fifo_in_place(&mut mesh.indices, mesh.vertices.len(), CACHE_SIZE as u32);
}

fn opt_overdraw(_mesh: &mut Mesh) {
    //
    // use worst-case ACMR threshold so that overdraw optimizer can sort *all* triangles
	// warning: this significantly deteriorates the vertex cache efficiency so it is not advised; look at optComplete for the recommended method
	//const float kThreshold = 3.f;
	//meshopt_optimizeOverdraw(&mesh.indices[0], &mesh.indices[0], mesh.indices.size(), &mesh.vertices[0].px, mesh.vertices.size(), sizeof(Vertex), kThreshold);
}

fn opt_fetch(_mesh: &mut Mesh) {
    //meshopt_optimizeVertexFetch(&mesh.vertices[0], &mesh.indices[0], mesh.indices.size(), &mesh.vertices[0], mesh.vertices.size(), sizeof(Vertex));
}

fn opt_fetch_remap(_mesh: &mut Mesh) {
    // this produces results equivalent to optFetch, but can be used to remap multiple vertex streams
	//std::vector<unsigned int> remap(mesh.vertices.size());
	//meshopt_optimizeVertexFetchRemap(&remap[0], &mesh.indices[0], mesh.indices.size(), mesh.vertices.size());

	//meshopt_remapIndexBuffer(&mesh.indices[0], &mesh.indices[0], mesh.indices.size(), &remap[0]);
	//meshopt_remapVertexBuffer(&mesh.vertices[0], &mesh.vertices[0], mesh.vertices.size(), sizeof(Vertex), &remap[0]);
}

fn opt_complete(mesh: &mut Mesh) {
    // vertex cache optimization should go first as it provides starting order for overdraw
    meshopt::optimize_vertex_cache_in_place(&mut mesh.indices, mesh.vertices.len());

	// reorder indices for overdraw, balancing overdraw and vertex cache efficiency
	//const float kThreshold = 1.05f; // allow up to 5% worse ACMR to get more reordering opportunities for overdraw
	//meshopt_optimizeOverdraw(&mesh.indices[0], &mesh.indices[0], mesh.indices.size(), &mesh.vertices[0].px, mesh.vertices.size(), sizeof(Vertex), kThreshold);

	// vertex fetch optimization should go last as it depends on the final index order
	//meshopt_optimizeVertexFetch(&mesh.vertices[0], &mesh.indices[0], mesh.indices.size(), &mesh.vertices[0], mesh.vertices.size(), sizeof(Vertex));
}

fn stripify(_mesh: &Mesh) {
    //
    /*
    double start = timestamp();
	std::vector<unsigned int> strip(mesh.indices.size() / 3 * 4);
	strip.resize(meshopt_stripify(&strip[0], &mesh.indices[0], mesh.indices.size(), mesh.vertices.size()));
	double end = timestamp();

	Mesh copy = mesh;
	copy.indices.resize(meshopt_unstripify(&copy.indices[0], &strip[0], strip.size()));

	assert(isMeshValid(copy));
	assert(areMeshesEqual(mesh, copy));

	meshopt_VertexCacheStatistics vcs = meshopt_analyzeVertexCache(&copy.indices[0], mesh.indices.size(), mesh.vertices.size(), kCacheSize, 0, 0);
	meshopt_VertexCacheStatistics vcs_nv = meshopt_analyzeVertexCache(&copy.indices[0], mesh.indices.size(), mesh.vertices.size(), 32, 32, 32);
	meshopt_VertexCacheStatistics vcs_amd = meshopt_analyzeVertexCache(&copy.indices[0], mesh.indices.size(), mesh.vertices.size(), 14, 64, 128);
	meshopt_VertexCacheStatistics vcs_intel = meshopt_analyzeVertexCache(&copy.indices[0], mesh.indices.size(), mesh.vertices.size(), 128, 0, 0);

	printf("Stripify : ACMR %f ATVR %f (NV %f AMD %f Intel %f); %d strip indices (%.1f%%) in %.2f msec\n",
	       vcs.acmr, vcs.atvr, vcs_nv.atvr, vcs_amd.atvr, vcs_intel.atvr,
	       int(strip.size()), double(strip.size()) / double(mesh.indices.size()) * 100,
	       (end - start) * 1000);
    */
}

fn simplify(mesh: &Mesh) {
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
    //
    /*
    double start = timestamp();

	std::vector<unsigned char> buffer(meshopt_encodeIndexBufferBound(mesh.indices.size(), mesh.vertices.size()));
	buffer.resize(meshopt_encodeIndexBuffer(&buffer[0], buffer.size(), &mesh.indices[0], mesh.indices.size()));

	double middle = timestamp();

	// using meshopt_Buffer instead of std::vector to avoid memset overhead
	meshopt_Buffer<unsigned int> result(mesh.indices.size());
	int res = meshopt_decodeIndexBuffer(&result[0], mesh.indices.size(), &buffer[0], buffer.size());
	assert(res == 0);
	(void)res;

	double end = timestamp();

	size_t csize = compress(buffer);

	for (size_t i = 0; i < mesh.indices.size(); i += 3)
	{
		assert(
		    (result[i + 0] == mesh.indices[i + 0] && result[i + 1] == mesh.indices[i + 1] && result[i + 2] == mesh.indices[i + 2]) ||
		    (result[i + 1] == mesh.indices[i + 0] && result[i + 2] == mesh.indices[i + 1] && result[i + 0] == mesh.indices[i + 2]) ||
		    (result[i + 2] == mesh.indices[i + 0] && result[i + 0] == mesh.indices[i + 1] && result[i + 1] == mesh.indices[i + 2]));
	}

	if (mesh.vertices.size() <= 65536)
	{
		meshopt_Buffer<unsigned short> result2(mesh.indices.size());
		int res2 = meshopt_decodeIndexBuffer(&result2[0], mesh.indices.size(), &buffer[0], buffer.size());
		assert(res2 == 0);
		(void)res2;

		for (size_t i = 0; i < mesh.indices.size(); i += 3)
		{
			assert(result[i + 0] == result2[i + 0] && result[i + 1] == result2[i + 1] && result[i + 2] == result2[i + 2]);
		}
	}

	printf("IdxCodec : %.1f bits/triangle (post-deflate %.1f bits/triangle); encode %.2f msec, decode %.2f msec (%.2f GB/s)\n",
	       double(buffer.size() * 8) / double(mesh.indices.size() / 3),
	       double(csize * 8) / double(mesh.indices.size() / 3),
	       (middle - start) * 1000,
	       (end - middle) * 1000,
	       (double(result.size * 4) / (1 << 30)) / (end - middle));
    */
}

fn encode_vertex<T>(mesh: &Mesh, name: &str) {
    //
    /*
    std::vector<PV> pv(mesh.vertices.size());
	packMesh(pv, mesh.vertices);

	double start = timestamp();

	std::vector<unsigned char> vbuf(meshopt_encodeVertexBufferBound(mesh.vertices.size(), sizeof(PV)));
	vbuf.resize(meshopt_encodeVertexBuffer(&vbuf[0], vbuf.size(), &pv[0], mesh.vertices.size(), sizeof(PV)));

	double middle = timestamp();

	// using meshopt_Buffer instead of std::vector to avoid memset overhead
	meshopt_Buffer<PV> result(mesh.vertices.size());
	int res = meshopt_decodeVertexBuffer(&result[0], mesh.vertices.size(), sizeof(PV), &vbuf[0], vbuf.size());
	assert(res == 0);
	(void)res;

	double end = timestamp();

	assert(memcmp(&pv[0], &result[0], pv.size() * sizeof(PV)) == 0);

	size_t csize = compress(vbuf);

	printf("VtxCodec%1s: %.1f bits/vertex (post-deflate %.1f bits/vertex); encode %.2f msec, decode %.2f msec (%.2f GB/s)\n", pvn,
	       double(vbuf.size() * 8) / double(mesh.vertices.size()),
	       double(csize * 8) / double(mesh.vertices.size()),
	       (middle - start) * 1000,
	       (end - middle) * 1000,
	       (double(result.size * sizeof(PV)) / (1 << 30)) / (end - middle));
    */
}

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
    /*
    std::vector<unsigned char> cbuf(tdefl_compress_bound(data.size() * sizeof(T)));
	unsigned int flags = tdefl_create_comp_flags_from_zip_params(MZ_DEFAULT_LEVEL, 15, MZ_DEFAULT_STRATEGY);
	return tdefl_compress_mem_to_mem(&cbuf[0], cbuf.size(), &data[0], data.size() * sizeof(T), flags);
    */
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

    let copy = mesh.clone();
	//meshopt_optimizeVertexCache(&copy.indices[0], &copy.indices[0], copy.indices.size(), copy.vertices.size());
	//meshopt_optimizeVertexFetch(&copy.vertices[0], &copy.indices[0], copy.indices.size(), &copy.vertices[0], copy.vertices.size(), sizeof(Vertex));

	stripify(&copy);

	encode_index(&copy);
    pack_vertex::<PackedVertex>(&copy, "");
    encode_vertex::<PackedVertex>(&copy, "");
    encode_vertex::<PackedVertexOct>(&copy, "0");

	simplify(&mesh);
}
