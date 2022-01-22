use meshopt::{
    any_as_u8_slice, quantize_snorm, quantize_unorm, rcp_safe, EncodeHeader, EncodeObject,
    PackedVertex, Vertex,
};

use std::{fs::File, io::Write, path::PathBuf};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "meshencoder")]
struct Options {
    /// Input file
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input: PathBuf,

    /// Output file
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: PathBuf,

    /// No optimization (just encoding)
    #[structopt(short = "u", long = "unoptimized")]
    unoptimized: bool,
}

#[derive(Debug)]
struct Object {
    material: String,
    index_offset: usize,
    index_count: usize,
}

#[allow(clippy::identity_op)]
fn main() {
    let options = Options::from_args();

    if options.unoptimized {
        println!("Encoding [unoptimized] {:?}", &options.input);
    } else {
        println!("Encoding {:?}", &options.input);
    }

    let obj_file = tobj::load_obj(
        &options.input,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    );
    let (models, materials) = obj_file.unwrap();
    let materials = materials.unwrap();

    let mut merged_positions: Vec<f32> = Vec::new();
    let mut merged_coords: Vec<f32> = Vec::new();
    let mut merged_vertices: Vec<Vertex> = Vec::new();
    let mut merged_indices: Vec<u32> = Vec::new();

    let mut objects: Vec<Object> = Vec::new();

    for (_, m) in models.iter().enumerate() {
        let mesh = &m.mesh;

        let material = match mesh.material_id {
            Some(id) => materials[id as usize].name.to_owned(),
            None => String::new(),
        };

        let mut vertices: Vec<Vertex> = Vec::new();
        let vertex_start = merged_vertices.len();
        let index_start = merged_indices.len();

        for i in 0..mesh.indices.len() {
            let index = mesh.indices[i] as usize;

            // pos = [x, y, z]
            let p = [
                mesh.positions[index * 3 + 0],
                mesh.positions[index * 3 + 1],
                mesh.positions[index * 3 + 2],
            ];

            merged_positions.push(p[0]);
            merged_positions.push(p[1]);
            merged_positions.push(p[2]);

            // normal = [x, y, z]
            let n = if !mesh.normals.is_empty() {
                [
                    mesh.normals[index * 3 + 0],
                    mesh.normals[index * 3 + 1],
                    mesh.normals[index * 3 + 2],
                ]
            } else {
                [0f32, 0f32, 0f32]
            };

            // tex coord = [u, v];
            let t = if !mesh.texcoords.is_empty() {
                [mesh.texcoords[index * 2], mesh.texcoords[index * 2 + 1]]
            } else {
                [0f32, 0f32]
            };

            merged_coords.push(t[0]);
            merged_coords.push(t[1]);

            vertices.push(Vertex { p, n, t });
            merged_indices.push((vertex_start + index) as u32);
        }

        merged_vertices.append(&mut vertices);

        objects.push(Object {
            material,
            index_offset: index_start,
            index_count: mesh.indices.len(),
        });
    }

    let pos_bits = 14;
    let uv_bits = 12;

    let (pos_offset, pos_scale) = meshopt::calc_pos_offset_and_scale(&merged_positions);
    let (uv_offset, uv_scale) = meshopt::calc_uv_offset_and_scale(&merged_coords);

    let pos_scale_inv = rcp_safe(pos_scale);
    let uv_scale_inv = [rcp_safe(uv_scale[0]), rcp_safe(uv_scale[1])];

    let quantized_vertices: Vec<PackedVertex> = merged_vertices
        .iter()
        .map(|v| {
            let p_0 = quantize_unorm((v.p[0] - pos_offset[0]) * pos_scale_inv, pos_bits) as u16;
            let p_1 = quantize_unorm((v.p[1] - pos_offset[1]) * pos_scale_inv, pos_bits) as u16;
            let p_2 = quantize_unorm((v.p[2] - pos_offset[2]) * pos_scale_inv, pos_bits) as u16;

            let n_0 = quantize_snorm(v.n[0], 8) as i8;
            let n_1 = quantize_snorm(v.n[1], 8) as i8;
            let n_2 = quantize_snorm(v.n[2], 8) as i8;

            let t_0 = quantize_unorm((v.t[0] - uv_offset[0]) * uv_scale_inv[0], uv_bits) as u16;
            let t_1 = quantize_unorm((v.t[1] - uv_offset[1]) * uv_scale_inv[1], uv_bits) as u16;

            PackedVertex {
                p: [p_0, p_1, p_2, 0],
                n: [n_0, n_1, n_2, 0],
                t: [t_0, t_1],
            }
        })
        .collect();

    let (vertex_count, vertex_remap) = meshopt::generate_vertex_remap(&quantized_vertices, None);

    let mut remapped_indices =
        meshopt::remap_index_buffer(None, merged_indices.len(), &vertex_remap);

    let mut remapped_vertices =
        meshopt::remap_vertex_buffer(&quantized_vertices, vertex_count, &vertex_remap);

    if !options.unoptimized {
        for object in &objects {
            meshopt::optimize_vertex_cache_in_place(
                &mut remapped_indices
                    [object.index_offset..(object.index_offset + object.index_count)],
                remapped_vertices.len(),
            );
        }

        meshopt::optimize_vertex_fetch_in_place(&mut remapped_indices, &mut remapped_vertices);
    }

    let encoded_vertices = meshopt::encode_vertex_buffer(&remapped_vertices).unwrap();
    let encoded_indices =
        meshopt::encode_index_buffer(&remapped_indices, remapped_vertices.len()).unwrap();

    let header = EncodeHeader {
        magic: *b"OPTM",
        group_count: objects.len() as u32,
        vertex_count: vertex_count as u32,
        index_count: merged_indices.len() as u32,
        vertex_data_size: encoded_vertices.len() as u32,
        index_data_size: encoded_indices.len() as u32,
        pos_offset,
        pos_scale: pos_scale / ((1 << pos_bits) - 1) as f32,
        uv_offset,
        uv_scale: [
            uv_scale[0] / ((1 << uv_bits) - 1) as f32,
            uv_scale[1] / ((1 << uv_bits) - 1) as f32,
        ],
        reserved: [0, 0],
    };

    let mut output = File::create(&options.output).unwrap();

    output.write_all(any_as_u8_slice(&header)).unwrap();

    for object in &objects {
        let object = EncodeObject {
            index_offset: object.index_offset as u32,
            index_count: object.index_count as u32,
            material_length: object.material.len() as u32,
            reserved: 0,
        };
        output.write_all(any_as_u8_slice(&object)).unwrap();
    }

    for object in &objects {
        output.write_all(object.material.as_bytes()).unwrap();
    }

    output.write_all(&encoded_vertices).unwrap();
    output.write_all(&encoded_indices).unwrap();

    println!("   Serialized encoded mesh to {:?}", &options.output);
}
