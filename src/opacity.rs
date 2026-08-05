use crate::ffi;
use std::mem;

/// Opacity micromap state format.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpacityStates {
    /// 2-state format, 1 bit per micro triangle: 0 transparent, 1 opaque.
    Two = 2,
    /// 4-state format, 2 bits per micro triangle: 0 transparent, 1 opaque,
    /// 2 unknown-transparent, 3 unknown-opaque.
    Four = 4,
}

/// Per-entry and per-triangle micromap assignment produced by [`opacity_map_measure`].
pub struct OpacityMapMeasure {
    /// Subdivision level of each micromap entry.
    pub levels: Vec<u8>,
    /// Triangle that each entry should be rasterized from, indexing the input triangle list.
    pub sources: Vec<u32>,
    /// Entry index for each input triangle, one element per triangle.
    pub indices: Vec<i32>,
}

/// Experimental: Computes a subdivision level for each input triangle, as well as deduplicating
/// the triangles that reference the same UVs to reduce rasterization requests.
///
/// `levels` and `sources` in the result have one element per micromap entry, and `indices` has one
/// element per input triangle.
///
/// * `vertex_uvs` should have a texture coordinate for each vertex
/// * `max_level` specifies the maximum subdivision level (0..12)
/// * `target_edge` can be 0. When >0, triangle subdivision is adaptive and targets `target_edge`^2 texel area
pub fn opacity_map_measure(
    indices: &[u32],
    vertex_uvs: &[[f32; 2]],
    texture_width: u32,
    texture_height: u32,
    max_level: u8,
    target_edge: f32,
) -> OpacityMapMeasure {
    let triangle_count = indices.len() / 3;
    let mut levels: Vec<u8> = vec![0; triangle_count];
    let mut sources: Vec<u32> = vec![0; triangle_count];
    let mut omm_indices: Vec<i32> = vec![0; triangle_count];

    let count = unsafe {
        ffi::meshopt_opacityMapMeasure(
            levels.as_mut_ptr(),
            sources.as_mut_ptr(),
            omm_indices.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            vertex_uvs.as_ptr().cast(),
            vertex_uvs.len(),
            mem::size_of::<[f32; 2]>(),
            texture_width,
            texture_height,
            i32::from(max_level),
            target_edge,
        )
    };

    levels.truncate(count);
    sources.truncate(count);

    OpacityMapMeasure {
        levels,
        sources,
        indices: omm_indices,
    }
}

/// Experimental: Returns the size in bytes of the opacity data for a single micromap entry.
///
/// `level` specifies the subdivision level (0..12).
pub fn opacity_map_entry_size(level: u8, states: OpacityStates) -> usize {
    unsafe { ffi::meshopt_opacityMapEntrySize(i32::from(level), states as i32) }
}

/// Experimental: Computes the data offset of each micromap entry, returning the offsets along with
/// the total size in bytes of the opacity data buffer.
pub fn opacity_map_layout(levels: &[u8], states: OpacityStates) -> (Vec<u32>, usize) {
    let mut offsets: Vec<u32> = Vec::with_capacity(levels.len());
    let mut data_size = 0;

    for &level in levels {
        offsets.push(data_size as u32);
        data_size += opacity_map_entry_size(level, states);
    }

    (offsets, data_size)
}

/// Alpha texture sampled while rasterizing opacity state.
#[derive(Debug, Copy, Clone)]
pub struct AlphaTexture<'a> {
    /// Alpha channel of the first pixel onwards, encoded as `UNORM8`.
    pub data: &'a [u8],
    /// Distance in bytes between consecutive pixels, e.g. 4 for `RGBA` input.
    pub stride: usize,
    /// Distance in bytes between consecutive rows, e.g. 4 * `width` for tightly packed `RGBA` input.
    pub pitch: usize,
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
}

impl AlphaTexture<'_> {
    /// Number of bytes that `data` must hold to cover every sampled pixel.
    ///
    /// The last pixel sampled is at `(height - 1) * pitch + (width - 1) * stride`.
    pub fn required_len(&self) -> usize {
        if self.width == 0 || self.height == 0 {
            0
        } else {
            (self.height as usize - 1) * self.pitch + (self.width as usize - 1) * self.stride + 1
        }
    }
}

/// Experimental: Rasterizes opacity state for a single triangle entry by sampling the alpha
/// texture, using bilinear filtering and 0.5 alpha cutoff.
///
/// * `result` must have room for the output opacity data (see [`opacity_map_entry_size`])
/// * `level` specifies the subdivision level (0..12)
/// * `uvs` holds the texture coordinates of each triangle corner. Micromap data is sensitive to the corner order
///
/// A smaller mip level can be used to trade rasterization cost for quality. Mip 0 is recommended
/// for 4-state micromaps so that the raytraced output is unchanged.
///
/// Index and meshlet compression can rotate triangle corners, so run the micromap stages after the
/// index order has been finalized.
///
/// Panics if `result` or the texture data is too small.
pub fn opacity_map_rasterize(
    result: &mut [u8],
    level: u8,
    states: OpacityStates,
    uvs: &[[f32; 2]; 3],
    texture: &AlphaTexture<'_>,
) {
    let entry_size = opacity_map_entry_size(level, states);
    assert!(
        result.len() >= entry_size,
        "result ({}) must have room for the opacity data of a level {level} entry ({entry_size})",
        result.len()
    );

    let required = texture.required_len();
    assert!(
        texture.data.len() >= required,
        "texture data ({}) must cover {}x{} pixels at stride {} and pitch {} ({required})",
        texture.data.len(),
        texture.width,
        texture.height,
        texture.stride,
        texture.pitch
    );

    unsafe {
        ffi::meshopt_opacityMapRasterize(
            result.as_mut_ptr(),
            i32::from(level),
            states as i32,
            uvs[0].as_ptr(),
            uvs[1].as_ptr(),
            uvs[2].as_ptr(),
            texture.data.as_ptr(),
            texture.stride,
            texture.pitch,
            texture.width,
            texture.height,
        );
    }
}

/// Experimental: Compacts and deduplicates opacity data, merging identical micromap entries and
/// replacing micromap states with special indices (-4..-1) when possible.
///
/// `data`, `levels` and `offsets` are updated in place and truncated to the compacted entry count,
/// which is returned. `omm_indices` is rewritten with the new entry index for each triangle, or
/// with a special index in -4..-1.
///
/// * `levels` and `offsets` must have one element per entry, as produced by [`opacity_map_layout`]
/// * `omm_indices` must have one element per triangle
pub fn opacity_map_compact(
    data: &mut Vec<u8>,
    levels: &mut Vec<u8>,
    offsets: &mut Vec<u32>,
    omm_indices: &mut [i32],
    states: OpacityStates,
) -> usize {
    assert_eq!(
        levels.len(),
        offsets.len(),
        "levels and offsets must have one element per entry"
    );

    let count = unsafe {
        ffi::meshopt_opacityMapCompact(
            data.as_mut_ptr(),
            data.len(),
            levels.as_mut_ptr(),
            offsets.as_mut_ptr(),
            levels.len(),
            omm_indices.as_mut_ptr(),
            omm_indices.len(),
            states as i32,
        )
    };

    levels.truncate(count);
    offsets.truncate(count);

    let data_size = if count == 0 {
        0
    } else {
        offsets[count - 1] as usize + opacity_map_entry_size(levels[count - 1], states)
    };
    data.truncate(data_size);

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize_unorm;

    const TRIANGLE_COUNT: usize = 6;
    const TEXTURE_SIZE: u32 = 32;

    #[rustfmt::skip]
    const INDICES: [u32; TRIANGLE_COUNT * 3] = [
        // 4 corner triangles
        0, 4, 7,
        1, 5, 4,
        2, 6, 5,
        3, 7, 6,
        // 2 center triangles (2x larger). The first is flipped from its correct orientation so
        // that both produce the same micromap, which exercises compaction
        4, 6, 5,
        4, 6, 7,
    ];

    #[rustfmt::skip]
    const UVS: [[f32; 2]; 8] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [1.0, 0.5],
        [0.5, 1.0],
        [0.0, 0.5],
    ];

    /// Alpha texture holding a centered circle.
    fn circle_texture() -> Vec<u8> {
        let center = TEXTURE_SIZE as f32 * 0.5;
        let radius = 10.0;

        (0..TEXTURE_SIZE)
            .flat_map(|y| {
                (0..TEXTURE_SIZE).map(move |x| {
                    let dx = x as f32 + 0.5 - center;
                    let dy = y as f32 + 0.5 - center;
                    let dc = radius - dx.hypot(dy);
                    quantize_unorm(dc + 0.5, 8) as u8
                })
            })
            .collect()
    }

    fn build(states: OpacityStates) -> (usize, Vec<i32>, Vec<u8>, Vec<u8>) {
        let texture_data = circle_texture();
        let texture = AlphaTexture {
            data: &texture_data,
            stride: 1,
            pitch: TEXTURE_SIZE as usize,
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
        };

        let measure = opacity_map_measure(&INDICES, &UVS, TEXTURE_SIZE, TEXTURE_SIZE, 4, 2.5);
        assert!(measure.levels.len() <= TRIANGLE_COUNT);
        assert_eq!(measure.levels.len(), measure.sources.len());
        assert_eq!(measure.indices.len(), TRIANGLE_COUNT);
        assert_eq!(measure.levels, [2, 2, 2, 2, 3, 3]);

        let mut levels = measure.levels;
        let (mut offsets, data_size) = opacity_map_layout(&levels, states);
        let mut data = vec![0u8; data_size];
        let mut omm_indices = measure.indices;

        for (i, &source) in measure.sources.iter().enumerate() {
            let tri = source as usize;
            assert!(tri < TRIANGLE_COUNT);

            let offset = offsets[i] as usize;
            let uvs = [
                UVS[INDICES[tri * 3] as usize],
                UVS[INDICES[tri * 3 + 1] as usize],
                UVS[INDICES[tri * 3 + 2] as usize],
            ];
            opacity_map_rasterize(&mut data[offset..], levels[i], states, &uvs, &texture);
        }

        let count = opacity_map_compact(
            &mut data,
            &mut levels,
            &mut offsets,
            &mut omm_indices,
            states,
        );

        assert_eq!(levels.len(), count);
        assert_eq!(offsets.len(), count);
        for &index in &omm_indices {
            assert!(index < 0 || (index as usize) < count);
        }

        (count, omm_indices, levels, data)
    }

    #[test]
    fn test_opacity_map_two_state() {
        let (count, omm_indices, levels, data) = build(OpacityStates::Two);

        // the 4 corner triangles compact to a special index, and the 2 center triangles share
        // a single entry
        assert_eq!(count, 1);
        assert_eq!(omm_indices, [-1, -1, -1, -1, 0, 0]);
        assert_eq!(levels, [3]);
        assert_eq!(data.len(), opacity_map_entry_size(3, OpacityStates::Two));
    }

    #[test]
    fn test_opacity_map_four_state() {
        let (count, omm_indices, levels, data) = build(OpacityStates::Four);

        assert_eq!(count, 1);
        assert_eq!(omm_indices, [-1, -1, -1, -1, 0, 0]);
        assert_eq!(levels, [3]);
        assert_eq!(data.len(), opacity_map_entry_size(3, OpacityStates::Four));
    }

    #[test]
    fn test_opacity_map_entry_size() {
        // a level N entry has 4^N micro triangles, at 1 or 2 bits each
        for level in 0..=6 {
            let micro_triangles = 1usize << (level * 2);
            assert_eq!(
                opacity_map_entry_size(level, OpacityStates::Two),
                (micro_triangles / 8).max(1)
            );
            assert_eq!(
                opacity_map_entry_size(level, OpacityStates::Four),
                (micro_triangles / 4).max(1)
            );
        }
    }

    #[test]
    fn test_opacity_map_rasterize_level0() {
        let texture_data = circle_texture();
        let texture = AlphaTexture {
            data: &texture_data,
            stride: 1,
            pitch: TEXTURE_SIZE as usize,
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
        };

        // 4-state values are 0 transparent, 1 opaque, 2 unknown-transparent, 3 unknown-opaque
        let cases = [
            ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 2),
            ([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], 0),
            ([[0.5, 0.5], [0.5, 0.0], [0.0, 0.5]], 3),
        ];

        for (uvs, expected) in cases {
            let mut result = [0u8];
            opacity_map_rasterize(&mut result, 0, OpacityStates::Four, &uvs, &texture);
            assert_eq!(result[0], expected);
        }
    }

    /// Rasterizing from an interleaved, row padded texture must match the tightly packed case.
    #[test]
    fn test_opacity_map_rasterize_strided() {
        let packed = circle_texture();

        // alpha in the 4th byte of each pixel, plus 5 pixels of row padding
        let stride = 4;
        let pitch = TEXTURE_SIZE as usize * stride + 5 * stride;
        let mut interleaved = vec![0u8; TEXTURE_SIZE as usize * pitch];
        for y in 0..TEXTURE_SIZE as usize {
            for x in 0..TEXTURE_SIZE as usize {
                interleaved[y * pitch + x * stride + 3] = packed[y * TEXTURE_SIZE as usize + x];
            }
        }

        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut expected = [0u8; 4];
        opacity_map_rasterize(
            &mut expected,
            1,
            OpacityStates::Four,
            &uvs,
            &AlphaTexture {
                data: &packed,
                stride: 1,
                pitch: TEXTURE_SIZE as usize,
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
            },
        );

        let mut result = [0u8; 4];
        opacity_map_rasterize(
            &mut result,
            1,
            OpacityStates::Four,
            &uvs,
            &AlphaTexture {
                data: &interleaved[3..],
                stride,
                pitch,
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
            },
        );

        assert_eq!(result, expected);
        // the texture is not uniform, so this is not a trivially equal pair of zeroed buffers
        assert_ne!(result, [0u8; 4]);
    }

    /// Compacts a lone entry whose micro triangles all share `value`, returning its special index.
    fn special_index(level: u8, states: OpacityStates, value: u8) -> i32 {
        let mut data = vec![value; opacity_map_entry_size(level, states)];
        let mut levels = vec![level];
        let mut offsets = vec![0u32];
        let mut omm_indices = [0i32];

        let count = opacity_map_compact(
            &mut data,
            &mut levels,
            &mut offsets,
            &mut omm_indices,
            states,
        );

        // a uniform entry is replaced entirely by the special index
        assert_eq!(count, 0);
        assert!(data.is_empty());
        assert!(levels.is_empty());
        assert!(offsets.is_empty());

        omm_indices[0]
    }

    #[test]
    fn test_opacity_map_compact_special_index() {
        for level in 0..=2 {
            assert_eq!(special_index(level, OpacityStates::Two, 0x00), -1);
            assert_eq!(special_index(level, OpacityStates::Two, 0xff), -2);

            assert_eq!(special_index(level, OpacityStates::Four, 0x00), -1);
            assert_eq!(special_index(level, OpacityStates::Four, 0x55), -2);
            assert_eq!(special_index(level, OpacityStates::Four, 0xaa), -3);
            assert_eq!(special_index(level, OpacityStates::Four, 0xff), -4);
        }
    }

    #[test]
    fn test_opacity_map_layout() {
        let levels = [2, 3, 1];
        let (offsets, data_size) = opacity_map_layout(&levels, OpacityStates::Four);

        let sizes: Vec<usize> = levels
            .iter()
            .map(|&level| opacity_map_entry_size(level, OpacityStates::Four))
            .collect();

        assert_eq!(offsets, [0, sizes[0] as u32, (sizes[0] + sizes[1]) as u32]);
        assert_eq!(data_size, sizes.iter().sum::<usize>());
    }
}
