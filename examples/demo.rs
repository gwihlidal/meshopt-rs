extern crate meshopt;
extern crate tobj;

use std::path::Path;
use std::mem;

const CACHE_SIZE: usize = 16;

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
struct PackedVertex {
    p: [u16; 4],
    n: [u8; 4],
    t: [u16; 2],
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
struct PackedVertexOct {
    p: [u16; 3],
    n: [u8; 2], // octahedron encoded normal, aliases .pw
    t: [u16; 2],
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
struct Vertex {
    p: [f32; 3],
    n: [f32; 3],
    t: [f32; 2],
}

impl Vertex {
    fn pack(&self) -> PackedVertex {
        unimplemented!();

        PackedVertex {
            ..Default::default()
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
struct Triangle {
    v: [Vertex; 3],
}

impl Triangle {
    fn rotate(&self) -> bool {
        unimplemented!();
    }
}

#[derive(Default, Debug, Clone)]
struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

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
                let p = [mesh.positions[index * 3], mesh.positions[index * 3 + 1], mesh.positions[index * 3 + 2]];

                let n = if !mesh.normals.is_empty() {
                    // normal = [x, y, z]
                    [mesh.normals[index * 3], mesh.normals[index * 3 + 1], mesh.normals[index * 3 + 2]]
                } else {
                    [0f32, 0f32, 0f32]
                };

                let t = if !mesh.texcoords.is_empty() {
                    // tex coord = [u, v];
                    [mesh.texcoords[index * 2], mesh.texcoords[index * 2 + 1]]
                } else {
                    [0f32, 0f32]
                };

                vertices.push(Vertex {
                    p,
                    n,
                    t,
                });
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
                vertex_remap.as_ptr() as *const ::std::os::raw::c_uint
            );
        }

        mesh.vertices.resize(total_vertices, Vertex::default());
        unsafe {
            meshopt::ffi::meshopt_remapVertexBuffer(
                mesh.vertices.as_ptr() as *mut ::std::os::raw::c_void,
                merged_vertices.as_ptr() as *const ::std::os::raw::c_void,
                total_indices,
                mem::size_of::<Vertex>(),
                vertex_remap.as_ptr() as *const ::std::os::raw::c_uint
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

    fn save_obj(&self, path: &Path) {}

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

    fn encode_index(&self) {
        unimplemented!();
    }

    fn stripify(&self) {
        unimplemented!();
    }

    fn deindex(&self) -> Vec<Triangle> {
        let tri_count = self.indices.len() / 3;
        let mut result = Vec::with_capacity(tri_count);

        for i in 0..tri_count {
            let i0 = self.indices[i * 3 + 0];
            let i1 = self.indices[i * 3 + 1];
            let i2 = self.indices[i * 3 + 2];
            let tri = Triangle {
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
            ::std::ptr::null(), // ib
            index_count,
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            index_count,
            mem::size_of::<T>()
        )
    };

    (vertex_count, remap)
}

fn pack_vertices<T>(input: &[T]) -> Vec<u8> {
    let conservative_size = unsafe {
        meshopt::ffi::meshopt_encodeVertexBufferBound(input.len(), mem::size_of::<T>())
    };

    println!("Conservative size is: {}, sizeof is: {}", conservative_size, mem::size_of::<T>());

    let mut encoded_data: Vec<u8> = Vec::new();
    encoded_data.resize(conservative_size, 0u8);

    let encoded_size = unsafe {
        meshopt::ffi::meshopt_encodeVertexBuffer(
            encoded_data.as_ptr() as *mut ::std::os::raw::c_uchar,
            encoded_data.len(),
            input.as_ptr() as *const ::std::os::raw::c_void,
            input.len(),
            mem::size_of::<T>()
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


    let encoded = pack_vertices(&vertices);

    
}

fn process_coverage() {
    encode_index_coverage();
    encode_vertex_coverage();
}

fn main() {
    println!("This is the demo");
    let mesh = Mesh::load_obj(&Path::new("examples/pirate.obj"));

    process_coverage();
}
