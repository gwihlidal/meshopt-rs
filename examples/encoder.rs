extern crate gltf;
extern crate meshopt;
extern crate tobj;

use meshopt::any_as_u8_slice;
use meshopt::{quantize_snorm, quantize_unorm};
use meshopt::{EncodeHeader, EncodeObject};
use meshopt::{PackedVertex, Vertex};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

struct Object {
    material: String,
    index_offset: usize,
    index_count: usize,
}

fn main() {
    //let obj = tobj::load_obj(Path::new("examples/pirate_opt.obj"));
    let obj = tobj::load_obj(Path::new("examples/multi.obj"));
    assert!(obj.is_ok());
    let (models, _materials) = obj.unwrap();

    let mut merged_positions: Vec<f32> = Vec::new();
    let mut merged_vertices: Vec<Vertex> = Vec::new();
    let mut merged_indices: Vec<u32> = Vec::new();
    let mut objects: Vec<Object> = Vec::new();

    for (i, m) in models.iter().enumerate() {
        let mesh = &m.mesh;

        println!("model[{}].name = \'{}\'", i, m.name);
        println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

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

            let n = if !mesh.normals.is_empty() {
                // normal = [x, y, z]
                [
                    mesh.normals[index * 3 + 0],
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
            merged_indices.push((vertex_start + index) as u32);
        }

        merged_vertices.append(&mut vertices);

        objects.push(Object {
            material: String::from("MAT_HERE"),
            index_offset: index_start,
            index_count: mesh.indices.len(),
        });
    }

    let pos_bits = 14;
    let uv_bits = 12;

    let (pos_offset, pos_scale_inv) = meshopt::calc_pos_offset_and_scale_inverse(&merged_positions);
    let (uv_offset, uv_scale_inv) = meshopt::calc_uv_offset_and_scale_inverse(&merged_positions);

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

    let (_, vertex_remap) = meshopt::generate_vertex_remap(&quantized_vertices, None);

    let mut remapped_indices =
        meshopt::remap_index_buffer(None, merged_indices.len(), &vertex_remap);
    let mut remapped_vertices = meshopt::remap_vertex_buffer(&quantized_vertices, &vertex_remap);

    for object in &objects {
        meshopt::optimize_vertex_cache_in_place(
            &mut remapped_indices[object.index_offset..(object.index_offset + object.index_count)],
            remapped_vertices.len(),
        );
    }

    meshopt::optimize_vertex_fetch_in_place(&mut remapped_indices, &mut remapped_vertices);

    let encoded_vertices = meshopt::encode_vertex_buffer(&remapped_vertices).unwrap();
    let encoded_indices =
        meshopt::encode_index_buffer(&remapped_indices, remapped_vertices.len()).unwrap();

    let header = EncodeHeader::new(
        objects.len() as u32,
        merged_vertices.len() as u32,
        merged_indices.len() as u32,
        encoded_vertices.len() as u32,
        encoded_indices.len() as u32,
        pos_offset,
        pos_scale_inv,
        uv_offset,
        uv_scale_inv,
        pos_bits as u32,
        uv_bits as u32,
    );

    let mut output = File::create("examples/multi.optmesh").unwrap();
    //let mut output = File::create("examples/pirate_opt.optmesh").unwrap();

    output.write(any_as_u8_slice(&header)).unwrap();

    for object in &objects {
        let object = EncodeObject {
            index_offset: object.index_offset as u32,
            index_count: object.index_count as u32,
            material_length: object.material.len() as u32,
            reserved: 0,
        };
        output.write(any_as_u8_slice(&object)).unwrap();
    }

    for object in &objects {
        output.write(object.material.as_bytes()).unwrap();
    }

    output.write(&encoded_vertices).unwrap();
    output.write(&encoded_indices).unwrap();
}
