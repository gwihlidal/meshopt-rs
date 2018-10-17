extern crate meshopt;
extern crate tobj;

use std::path::Path;
use std::mem;

const CACHE_SIZE: usize = 16;

#[derive(Default, Debug, Clone)]
#[repr(C)]
struct PackedVertex {
    p: [u16; 4],
    n: [u8; 4],
    t: [u16; 2],
}

#[derive(Default, Debug, Clone)]
#[repr(C)]
struct PackedVertexOct {
    p: [u16; 3],
    n: [u8; 2], // octahedron encoded normal, aliases .pw
    t: [u16; 2],
}

#[derive(Default, Debug, Clone)]
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

#[derive(Default, Debug, Clone)]
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
        false
    }

    fn load_obj(path: &Path) -> Self {
        let obj = tobj::load_obj(path);
        assert!(obj.is_ok());
        let (models, _) = obj.unwrap();

        let mut mesh = Self::default();

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

    let encoded_size = unsafe {
        meshopt::ffi::meshopt_encodeVertexBufferBound(vertices.len(), mem::size_of::<PackedVertexOct>())
    };

    println!("Encoded size is: {}", encoded_size);

    let mut encoded_data: Vec<u8> = Vec::new();
    encoded_data.resize(encoded_size, 0u8);

    let encoded_size2 = unsafe {
        meshopt::ffi::meshopt_encodeVertexBuffer(
            encoded_data.as_ptr() as *mut ::std::os::raw::c_uchar,
            encoded_data.len(),
            vertices.as_ptr() as *const ::std::os::raw::c_void,
            vertices.len(),
            mem::size_of::<PackedVertexOct>()
        )
    };

    encoded_data.resize(encoded_size2, 0u8);
    println!("Encoded size2 is: {}", encoded_size2);
    

    /*let encoded_size = unsafe {
        spv_data.as_ptr() as *const ::std::os::raw::c_void,
        meshopt::ffi::meshopt_encodeVertexBuffer();
    }


    pub fn meshopt_encodeVertexBuffer(
        buffer: *mut ::std::os::raw::c_uchar,
        buffer_size: usize,
        vertices: *const ::std::os::raw::c_void,
        vertex_count: usize,
        vertex_size: usize,
    ) -> usize;*/
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
